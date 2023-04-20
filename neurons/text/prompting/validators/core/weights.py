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
import torch
import random
import bittensor

from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict

def compute_weights( neuron ) -> Tuple[ torch.LongTensor, torch.FloatTensor ]:
    """
        Computes the average reward for each uid across non-zero values 
        using the rewards history stored in the self.history list.

        Returns:
            uids ( torch.LongTensor, shape = (n) ): 
                Uid to set weights on.
            weights ( torch.FloatTensor, shape = (n) ): 
                The weights for each uid.
    """
    bittensor.logging.info( 'compute_weights()' )

    # Return zeros weights if there is no history.
    if neuron.history.qsize() == 0: 
        bittensor.logging.warning( 'No history to compute weights returning all ones.' )
        return torch.ones((neuron.metagraph.n)) / neuron.metagraph.n

    # Iterate over all events in the `history` and perform a moving average of the normalized rewards.
    alpha = 0.01
    last_hotkeys = None
    moving_averaged_scores = torch.zeros((neuron.metagraph.n)).to( neuron.device )
    for event in neuron.history.queue:    
        # First we normalize the rewards with a softmax.
        normalized_rewards = torch.nn.functional.softmax( event.rewards.to( neuron.device ), dim=0 )
        # We scatter the normalized onto the moving averaged scores (updating them but not changing the source)
        scattered_rewards = moving_averaged_scores.scatter(0, event.successful_uids.to( neuron.device ), normalized_rewards.to( neuron.device ) )
        # We now perform a moving average of the scattered rewards.
        moving_averaged_scores = alpha * moving_averaged_scores + ( 1 - alpha ) * scattered_rewards
        bittensor.logging.trace( 'normalized_rewards', normalized_rewards )
        bittensor.logging.trace( 'scattered_rewards', scattered_rewards )
        bittensor.logging.trace( 'moving_averaged_scores', moving_averaged_scores )

        # If the hotkeys have changed, reset the moving averaged scores for the new hotkeys.
        if last_hotkeys is not None:
            for uid, hotkey in enumerate( last_hotkeys ):
                if hotkey != event.hotkeys[ uid ]:
                    moving_averaged_scores[ uid ] = 0
        # Update the last hotkeys.
        last_hotkeys = event.hotkeys

    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize( moving_averaged_scores, p=1, dim=0 )
    bittensor.logging.trace( 'raw_weights', raw_weights )
    bittensor.logging.trace( 'top10 values', raw_weights.sort()[0] )
    bittensor.logging.trace( 'top10 uids', raw_weights.sort()[1] )
    
    # Process the raw weights to final_weights via subtensor limitations.
    processed_weight_uids, processed_weights = bittensor.utils.weight_utils.process_weights_for_netuid(
        uids = neuron.metagraph.uids.to( "cpu" ),
        weights = raw_weights.to( "cpu" ),
        netuid = neuron.config.netuid,
        subtensor = neuron.subtensor,
        metagraph = neuron.metagraph
    )
    bittensor.logging.trace( 'processed_weights', processed_weights )
    bittensor.logging.trace( 'processed_weight_uids', processed_weight_uids )
    return processed_weight_uids, processed_weights
