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

import time
import bittensor
from forward import forward
from weights import compute_weights
from typing import List, Optional, Tuple, Dict

def train( neuron ):
    """ Training 
        The function uses an infinite loop to repeatedly generate a random question, 
        ask the network to complete the question, and train the gating network using 
        the question and the resulting completions.
    """
    # Store the current epoch block number for comparison later.
    last_epoch_block = neuron.subtensor.block
    
    # Start an infinite loop for training.
    while True:
    
        # Optionally turn off training here.
        if not neuron.config.neuron.dont_train:
            # Query the network for a random question.
            question_result = forward( 
                neuron = neuron,
                roles = ['system', 'user' ],
                messages = [ neuron.config.neuron.base_prompt, neuron.config.neuron.question_prompt ],
                timeout = neuron.config.neuron.training_timeout,
                num_to_query = neuron.config.neuron.training_topk,
                random_sample_uids = True,
                train_gating_model = True,
                apply_backward = True,
            )
            if not question_result.success: continue # not a successful forward pass.
            
            # Ask the network to complete the random question, training the gating network.
            forward( 
                neuron = neuron,
                roles = ['system', 'user' ],
                messages = [ neuron.config.neuron.base_prompt, question_result.completion ],
                timeout = neuron.config.neuron.training_timeout,
                num_to_query = neuron.config.neuron.training_topk,
                random_sample_uids = True,
                train_gating_model = True,
                apply_backward = True,
            )
        else:
            # Wait.
            time.sleep( 1 )

        # Resync metagraph before returning. (sync every 15 min or ~75 blocks)
        if last_epoch_block % 10 == 0:
            neuron.metagraph.sync()
            neuron.my_nominators = { nomin[0]: nomin[1] for nomin in neuron.subtensor.get_delegated( neuron.wallet.coldkeypub.ss58_address )[0][0].nominators }

        # Check if enough epoch blocks have elapsed since the last epoch.
        epoch_length = neuron.subtensor.validator_epoch_length(neuron.config.netuid) if neuron.config.neuron.epoch_length_override == -1 else neuron.config.neuron.epoch_length_override
        blocks_until_epoch = epoch_length - ( neuron.subtensor.block - last_epoch_block )
        bittensor.logging.debug( 'blocks_until_epoch', blocks_until_epoch )
        if blocks_until_epoch <= 0: 
            bittensor.logging.trace( 'epoch()' )
            bittensor.logging.info( 'block', neuron.subtensor.block )

            # Update the last epoch block to the current epoch block.
            last_epoch_block = neuron.subtensor.block
            
            # Computes the average reward for each uid across non-zero values 
            # using the rewards history stored in the neuron.history list.
            uids, weights = compute_weights( neuron )
            bittensor.logging.info( 'weights', weights )

            # Set the weights on chain via our subtensor connection.
            neuron.subtensor.set_weights(
                wallet = neuron.wallet,
                netuid = neuron.config.netuid,
                uids = uids,
                weights = weights,
                wait_for_finalization = True,
            )