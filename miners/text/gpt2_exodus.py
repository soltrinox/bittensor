#!/bin/python3
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
""" The Exodus miner.

Example:
    $ python miners/text/gpt2_exodus.py

"""

import argparse
import bittensor
import math
import torch
import traceback
import os
import sys
import yaml

from termcolor import colored
from typing import List
from qqdm import qqdm, format_str
from loguru import logger; logger = logger.opt(colors=True)
from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Nucleus(nn.Module):

    def __init__(self, config ):
        super(Nucleus, self).__init__()
        self.config = config
        encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, self.config.nucleus.nhead, self.config.nucleus.nhid, self.config.nucleus.dropout ,batch_first=True)
        self.transformer = TransformerEncoder( encoder_layers, self.config.nucleus.nlayers)
        self.encoder = nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
        self.decoder = nn.Linear(  bittensor.__network_dim__, bittensor.__vocab_size__ )
        self.loss_fct = nn.CrossEntropyLoss()
        self.chain_weights = torch.zeros( [0] , requires_grad=True)
        self.init_weights()

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):    
        r""" Add custom params to the parser.
        """
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200)
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default=2)
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default=10)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def local_forward(self, inputs: torch.int64, training : bool = True) -> SimpleNamespace:
        """ Forward pass through GPT2 nucleus.
            Args:
                inputs (:obj:`torch.int64` of shape :obj:`(batch_size, block_size)`, `required`): 
                    Batch_size length x list of text sentences.
                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes a CLM loss.

            Returns:
                SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.
                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions produced using local_context. 
                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        GPT MLM loss using local_context.
                }
        """
        output = SimpleNamespace()

        # local_hidden: hidden layer encoding of sequence with local_context.
        # local_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        token = self.encoder( inputs )
        output.local_hidden = self.transformer( token )
        logger.info(output.local_hidden.shape)

        if training :
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.local_target = self.decoder( output.local_hidden )
            logger.info(output.local_target.shape)

            # local_target_loss: MLM loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            shift_logits = output.local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()     
            logger.info([shift_labels.shape,shift_logits.shape])
       
            output.local_target_loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
            
        return output

    def remote_forward(self, inputs: torch.int64, training: bool) -> SimpleNamespace:
        """ Forward pass inputs and labels through the GPT2 module and into the remote network.
        Args:
            inputs (:obj:`torch.int64` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.
            training (:obj:`bool')`, `optional`, defaults to True):
                Switch to True if this forward pass computes an MLM loss.
        Returns:
            self.local_forward() + SimpleNamespace ( 
                remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                    Hidden layer encoding produced using the remote_context.
                remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                    GPT MLM Target predictions using the remote_context.
                remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                    GPT MLM loss using the remote_context.
                distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                    Distillation loss between local_context and remote_context.
            )
        """
        # Run local model
        # output = SimpleNamespace
        output = self.local_forward(inputs, training)

        # remote_context: joined responses from a dendrite.forward_text call.
        # remote_context.shape = [batch_size, sequence_len (or block_size), bittensor.__network_dim__]
        output.remote_hidden = self.remote( inputs )

        # distillation_loss : distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss(output.local_hidden, output.remote_hidden.detach())


        if training :
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.remote_target = self.decoder(output.remote_hidden)

            # remote_target_loss: MLM loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            shift_logits = output.remote_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()            
            output.remote_target_loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )
        
        return output

    def remote(self, inputs: torch.int64 ) -> torch.float32:
        """ Forwards the inputs through the network, selects the topk peers based on self.chain_weights.
        Args:
            inputs (:obj:`torch.int64` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                Batch_size length list of text sentences.
        Returns:
            outputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                Joined hidden layer responses from peers.
        """
        # ---- Topk Weights ----
        topk_weights, topk_uids = self.chain_weights.topk( self.config.nucleus.topk, dim=0 )

        # ---- Filter endpoints ----
        endpoints = [bittensor.neuron.metagraph.endpoints[uid] for uid in topk_uids]

        # ---- Query network ----
        responses, _ = bittensor.neuron.dendrite.forward_text( 
            endpoints = endpoints, 
            inputs = [inputs for _ in endpoints] 
        )


        # ---- Join based on weights ----
        joining_weights = F.softmax( topk_weights, dim = 0 )
        output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to(self.config.miner.device)
        for index, response in enumerate( responses ): 
            output += response * joining_weights[ index ]

        # ---- Return response -----
        return output

class Miner:

    def __init__( self, config: 'bittensor.config' = None ):
        r""" Initializes a miner with the passed config.
        """
        if config == None: config = Miner.config()
        self.config = config; Miner.check_config( self.config ); print ( self.config )
        
        # Miner training device.
        self.device = torch.device(
            device = self.config.miner.device
        )

        # Dataset of text.
        self.dataset = bittensor.dataloader (
            config = self.config
        )

        # Trainable machine learning model.
        self.nucleus = Nucleus(
            config = self.config,
        ).to( self.device )

        # Torch optimizer.
        self.optimizer = torch.optim.SGD(
            [ {"params": self.nucleus.parameters()}],
            lr = self.config.miner.learning_rate,
            weight_decay = self.config.miner.weight_decay,
        )

    @staticmethod
    def config() -> 'bittensor.Config':
        r""" Fills a config namespace object with defaults or information from the command line.
        """
        # ---- Add miner args.
        parser = argparse.ArgumentParser()
        parser.add_argument('--miner.config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=3e-2)
        parser.add_argument('--miner.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
        parser.add_argument('--miner.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--miner.n_epochs', type=int, help='Number of training epochs.', default=sys.maxsize )
        parser.add_argument('--miner.epoch_length', type=int, help='Iterations of training per epoch', default=500)
        parser.add_argument('--miner.batch_size_train', type=int, help='Training batch size.', default=2)
        parser.add_argument('--miner.reload', action='store_true', help='''Reload training from previous trial run.''', default=False )
        parser.add_argument('--miner.restart_on_failure',  action='store_true', help='''Restart miner on unknown error.''', default=False)
        parser.add_argument('--miner.compute_remote_gradients', action='store_true', help='''Does the miner compute and return gradients from backward queries.''', default=False)
        parser.add_argument('--miner.accumulate_remote_gradients', action='store_true', help='''Does the miner accumulate remote gradients from backward queries.''', default=False)
        parser.add_argument('--miner.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='gpt2_exodus')
        parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--miner.use_upnpc', action='store_true', help='''Turns on port forwarding on your router using upnpc.''', default=False)
        bittensor.add_args( parser )
        Nucleus.add_args( parser )  

        # ---- Loads config_file and updates defaults
        config_file_path = vars(parser.parse_known_args()[0])['miner.config']
        if config_file_path:
            config_file_path = os.path.expanduser(config_file_path)
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f) 
                    print('Config File Detected at {} updating defaults'.format(config_file_path))
                    parser.set_defaults(**params_config)
            except Exception as e:
                print('Error in loading: {} using default parser settings'.format(e))

        return bittensor.config( parser )

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        bittensor.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name + "-" + config.wallet.hotkey, config.miner.name ))
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    def run( self ):
        r""" Miner main loop.
        """
        bit = bittensor.init (  
                config = self.config,
                root_dir = self.config.miner.full_path,
                axon_forward_callback = self.forward,
                axon_backward_callback = self.backward,
            )
        # ---- Build Bittensor neuron ----
        with bit:
            # ---- Init run state ----
            self.epoch = 0
            self.global_step = 0
            self.epoch_loss = math.inf/2
            self.best_epoch_loss = math.inf

            # ---- Optionally reload from previous run ----
            if self.config.miner.reload:
                self.reload()
            else:
                self.checkpoint()

            # --- Run until n_epochs ----
            while self.epoch < self.config.miner.n_epochs:
                try:
                    # ---- Train state ----
                    self.run_epoch()

                    # ---- Set weights on chain ----
                    self.set_chain_weights()

                    # ---- Checkpoint state ----
                    self.checkpoint()

                except KeyboardInterrupt:
                    # --- User ended session ----
                    break

                except Exception as e:
                    # --- Unknown error ----
                    logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                    if self.config.miner.restart_on_failure == True:
                        logger.info('Restarting from last saved state.')
                        self.reload()
                    else:
                        break

    # --- Run Epoch ----
    def run_epoch( self ):
        r""" Runs a single training epoch pulled from the dataloader.
        """
        # --- Init Epoch ----
        total_epoch_loss = 0.0
        epoch_batches = self.dataset.dataloader( self.config.miner.epoch_length )
        progress_bar = qqdm(enumerate(epoch_batches), total=len(epoch_batches), desc=format_str('blue', f'Epoch Progress'))
        for iteration, (inputs) in progress_bar:

            # ---- Forward / Backward ----
            output = self.train ( batch = { 'inputs': inputs } )
            total_epoch_loss += output.local_target_loss.item()

            # ---- Logs ----
            self.logs (
                progress_bar,
                iteration = iteration,
                output = output,
            )
            self.global_step += 1

        self.epoch_loss = total_epoch_loss / self.config.miner.epoch_length
        self.epoch += 1

    # ---- Training call ----
    def train ( self, batch: dict ) -> SimpleNamespace:
        r""" Runs a single training batch through the nucleus and applies a gradient update.
            Args:
                batch ( dict, `required`):
                    training batch dictionary.
            Returns:
                output = SimpleNamespace (
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Representations produced by the nucleus's distillation-model prior to producing the hidden units.
                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer representations produced using the local_context.
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `required`):
                        GPT2 MLM target predictions produced using local_hidden.
                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        GPT2 MLM loss computed from the local_target.
                    remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Representations returned from the nucleus.remote function after querying the network.
                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer representations produced using the remote_context.
                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `required`):
                        GPT MLM Target predictions produced using remote_hidden.
                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        GPT2 MLM loss computed from the remote_target.
                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        Distillation loss between local_context and remote_context.
            )
        """
        # Zeros out gradients for next accummulation
        self.optimizer.zero_grad() 

        # ---- Forward pass ----
        inputs = batch['inputs']
        output = self.nucleus.remote_forward (
            inputs = inputs.to( self.device ),
            training = True,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the nucleus.
        clip_grad_norm_(self.nucleus.parameters(), self.config.miner.clip_gradients)
        self.optimizer.step() # Applies accumulated gradients.

        # ---- Update global loss ----
        return output

    # ---- Axon Forward call ----
    def forward ( self, pubkey:str, inputs_x: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: processes forward messages from the wire.
            The arguments reflect an RPC request from another miner in the network, the response tensor
            should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        inputs_x = inputs_x.to( self.device )
        output = self.nucleus.local_forward (
            inputs = inputs_x
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: Processes backward messages from the wire.
            Arguments reflect an RPC backward request from another miner in the network, the response tensor
            should be the gradients of the miner's nucleus w.r.t to the inputs_x and the passed output grads_dy.

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.

            Returns:
                outputs (:obj:`torch.FloatTensor`, `optional`):
                    The gradients w.r.t to the inputs [batch_size, sequence_len, -1]
        """
        if self.config.miner.compute_remote_gradients:
            with torch.enable_grad():

                # ---- Set up inputs for gradient computations.
                inputs_x.requires_grad = True
                inputs_x = inputs_x.to( self.device )
                grads_dy = grads_dy.to( self.device )
                outputs_y = self.nucleus.local_forward( inputs = inputs_x ).to( self.device )

                # ---- The backward call will accumulate gradients on our parameters.
                if self.config.miner.accumulate_remote_gradients:
                    torch.autograd.backward (
                        tensors = [outputs_y],
                        grad_tensors = [grads_dy]
                    )
                    return inputs_x.grad if inputs_x.grad != None else None

                # ---- The backward call will simply compute the gradients without accumulating them.
                else:
                    grads_dy = torch.autograd.grad (
                        outputs = outputs_y,
                        inputs = inputs_x,
                        grad_outputs = grads_dy,
                        only_inputs = True,
                        create_graph = False,
                        retain_graph = False
                    )[0]
                    return grads_dy

        # if ! compute_remote_gradients, NO-OP.
        else:
            return None

    def checkpoint( self ):
        r""" Optionally Saves, updates and then reloads the miner training state.
        """
        last_saved = self.get_saved_state()
        if last_saved == None or last_saved['epoch_loss'] >= self.epoch_loss:
            self.save()
        bittensor.neuron.metagraph.load()
        bittensor.neuron.metagraph.sync()
        bittensor.neuron.metagraph.save()
        self.reload()

    def get_saved_state( self ):
        r""" Returns a saved state dict or none.
        """
        try:
            return torch.load("{}/model.torch".format( self.config.miner.full_path ))
        except Exception as e:
            logger.exception('Failed to reload model with error: {}', e)
            return None

    def reload( self ):
        r""" Reloads/updates the training state from the disk.
        """
        state_dict = self.get_saved_state()

        # ---- Load training state.
        self.epoch = state_dict['epoch']
        self.epoch_loss = state_dict['epoch_loss']
        self.global_step = state_dict['global_step']
        for uid in bittensor.neuron.metagraph.uids.tolist():
            self.nucleus.chain_weights = torch.cat ((self.nucleus.chain_weights,torch.zeros([1], requires_grad=True)),0) #updates the shape of nucleus chain weights
        self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        self.nucleus.to( self.device ) # Load nucleus

        

        # --- Load optimizer.
        self.optimizer = torch.optim.SGD(
            [{"params": self.nucleus.parameters()}],
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            weight_decay = state_dict['optimizer_state']['param_groups'][0]['weight_decay'],
        )
        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.miner.full_path ))

    def save( self ):
        r""" Saves the training state to disk.
        """
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.epoch_loss,
                'global_step': self.global_step,
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.miner.full_path, self.epoch_loss ) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( self.config.miner.full_path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def set_chain_weights( self ):
        r""" Sets the chain weights.
        """
        # TODO(const): select topk.
        try:
            uids = bittensor.neuron.metagraph.uids
            did_set = bittensor.neuron.subtensor.set_weights(
                uids = uids,
                weights = self.nucleus.chain_weights,
                wait_for_inclusion = True,
                wallet = bittensor.neuron.wallet,
            )
            if did_set:
                bittensor.logging.success(prefix='Set weights:', sufix='{}'.format(self.nucleus.chain_weights.tolist()))
            else:
                logger.warning('Failed to set weights on chain.')
                bittensor.neuron.subtensor = bittensor.subtensor( config = self.config.subtensor )
                bittensor.neuron.subtensor.connect()

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)

    # ---- Training logs ----
    def logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called after every training step. Displays miner state to screen.
        """
        self_uid = bittensor.neuron.metagraph.hotkeys.index( bittensor.neuron.wallet.hotkey.public_key )
        stake = bittensor.neuron.metagraph.S[ self_uid ].item()
        rank = bittensor.neuron.metagraph.R[ self_uid ].item()
        incentive = bittensor.neuron.metagraph.I[ self_uid ].item()
        info = {
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Loss': colored('{:.4f}'.format(self.epoch_loss), 'yellow'),
            'Best': colored('{:.4f}'.format(self.best_epoch_loss), 'red'),
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(bittensor.neuron.metagraph.n.item(), 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(stake), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(rank), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(incentive), 'yellow'),
        }
        for uid in bittensor.neuron.metagraph.uids.tolist():
            if self.nucleus.chain_weights[uid] != 0:
                print(self.nucleus.chain_weights[uid])
                weight_dif = -self.nucleus.chain_weights[uid].grad
                if weight_dif > 0:
                    info[colored(str(uid), 'green')] = colored('{:.4f}'.format(self.nucleus.chain_weights[uid]), 'green')
                elif weight_dif == 0:
                    info[str(uid)] = colored('{:.4f}'.format(self.nucleus.chain_weights[uid]), 'white')
                else:
                    info[colored(str(uid), 'red')] = colored('{:.4f}'.format(self.nucleus.chain_weights[uid]), 'red')
        progress_bar.set_infos( info )

if __name__ == "__main__":
    Miner().run()
