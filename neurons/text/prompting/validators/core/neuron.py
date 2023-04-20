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

import os
import time
import json
import math
import copy
import queue
import torch
import random
import bittensor
import argparse
import bittensor as bt

from loguru import logger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from reward import RewardModel
from gating import GatingModel

from train import train
from forward import forward, setup_events_sink
from synapse import synapse_blacklist, synapse_priority, synapse_forward, synapse_backward

__default_question_prompt__ = '''
Ask me a random question about anything. Make the question very domain specific. Do not include the answer in the question.
'''

__default_base_prompt__ = '''
You are designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
'''

class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bt.logging.check_config( config )
        bt.wallet.check_config( config )
        bt.subtensor.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser( full_path )
        config.neuron.reward_path = os.path.expanduser( config.neuron.reward_path )
        if not os.path.exists( config.neuron.full_path ):
            os.makedirs( config.neuron.full_path, exist_ok = True)
        if not os.path.exists( config.neuron.reward_path + '/hf_ckpt.pt' ):
            os.makedirs( config.neuron.reward_path, exist_ok = True )
            os.system(
                f"wget -O { config.neuron.reward_path + '/hf_ckpt.pt'} \
                https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt"
            )
        if not config.neuron.dont_log_events_to_file:
            setup_events_sink( config )

    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument( '--netuid', type = int, help = 'Prompting network netuid', default = 1 )
        parser.add_argument( '--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument( '--neuron.base_prompt', type=str, help = 'Prompt injected before a question is completed by miners on the network', default = __default_base_prompt__ )
        parser.add_argument( '--neuron.question_prompt', type=str, help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = __default_question_prompt__ )
        parser.add_argument( '--neuron.reward_model_name', type = str, help = 'GPTRewardModel name', default = 'Dahoas/gpt2-rm-static')
        parser.add_argument( '--neuron.length_timeout_multiplier', type = int, help = 'Base timeout for all requests.', default = 0.01 )
        parser.add_argument( '--neuron.inference_topk', type = str, help = 'At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument( '--neuron.training_topk', type = str, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument( '--neuron.inference_timeout', type = float, help = 'At inference time, how long we wait before timeing out the query', default = 3 )
        parser.add_argument( '--neuron.training_timeout', type = float, help = 'During training time, how long we wait before timeing out the query', default = 3 )
        parser.add_argument( '--neuron.reward_path', type = str, help = 'Path to reward model.', default = '~/.bittensor/reward_models' )
        parser.add_argument( '--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 1000 )
        parser.add_argument( '--neuron.device', type = str, help = 'Device to run the validator on.', default = "cuda" if torch.cuda.is_available() else "cpu" )
        parser.add_argument( '--neuron.epoch_length_override', type = int, help = 'Override the default timeout', default = -1 )
        parser.add_argument( '--neuron.dont_train', action = 'store_true', help = 'If set, we dont train the validator from the self loop.', default = False )
        parser.add_argument( '--neuron.no_reward_model', action = 'store_true', help = 'If set, we dont load the reward model instead use just the scores.', default = False )
        parser.add_argument( '--neuron.dont_log_events_to_file', action = 'store_true', help = 'If set, we dont save events to a log file.', default = False )
        parser.add_argument( '--neuron.events_retention_size',  type = str,  help = 'Events retention size.', default = "500 MB" )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        bt.axon.add_args( parser )
        GatingModel.add_args( parser )
        cls.add_args( parser )
        return bt.config( parser )
    
    def __init__( self ):
        self.config = neuron.config()
        self.check_config( self.config )
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        print( self.config )
        
        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device( self.config.neuron.device )
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )

        # Reward model
        if not self.config.neuron.no_reward_model:
            bittensor.logging.info('Loading reward model')
            self.reward_model = RewardModel( model_path = 'EleutherAI/gpt-j-6b', device = self.config.neuron.device )
            for fpath in os.listdir( self.config.neuron.reward_path ):
                if fpath.endswith(".pt") or fpath.endswith(".bin"):
                    checkpoint = os.path.join( self.config.neuron.reward_path, fpath )
                    break
            ckpt_state = torch.load( checkpoint )
            self.reward_model.load_state_dict( ckpt_state )
            self.reward_model.eval()
            self.reward_model.half()
            self.reward_model.requires_grad_( False )
            self.reward_model.to( self.device )
            bittensor.logging.info('done loading reward model')

        # Init the gating model which learns which miners to select for each query.
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to( self.device )
        # Denddrite pool for querying the network.
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, keypair = self.wallet.hotkey )
        # History of forward events.
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )
        # Get a list of peers delegating to me
        self.my_nominators = { nomin[0]: nomin[1] for nomin in self.subtensor.get_delegated( self.wallet.coldkeypub.ss58_address )[0][0].nominators }

        # Build synapse entrypoint.
        class Synapse( bittensor.TextPromptingSynapse ):
            def priority( _, forward_call: "bittensor.TextPromptingForwardCall" ) -> float: 
                return synapse_priority( self, forward_call )
            def blacklist( _, forward_call: "bittensor.TextPromptingForwardCall" ) -> bool: 
                return synapse_blacklist( self, forward_call )
            def forward( _, messages: List[Dict[str, str]] ) -> str:  
                return synapse_forward( self, messages )
            def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: 
                pass

        # Serve axon.
        self.axon = bittensor.axon( 
            wallet = self.wallet,
            metagraph = self.metagraph,
            config = self.config,
        )
        self.synapse = Synapse( axon = self.axon )
        self.axon.start()
        #self.subtensor.serve_axon( self.config.netuid, self.axon )

    def train( self ): train( self )


if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().train()