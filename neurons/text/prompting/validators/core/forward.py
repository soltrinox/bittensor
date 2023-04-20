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

import json
import time
import copy
import torch
import random
import bittensor
from loguru import logger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict

# Needed for the reward model and the scoring model which act on raw strings.
def flattened_messages_for_reward( roles: List[str], messages: List[str], completion: str ) -> str:
    """ Returns completion formatted as to maximize reward model scoring.
    """
    flattened_message = ''
    for role_i, message_i in list(zip(roles, messages)):
        if role_i != 'system': flattened_message += message_i.strip() + '\n\n'
    flattened_message += completion.strip()
    return flattened_message

def flattened_messages_for_gating( roles: List[str], messages: List[str] ) -> str:
    """ Returns completion formatted as to maximize reward model scoring.
    """
    flattened_message = ''
    for role_i, message_i in list(zip(roles, messages)):
        if role_i != 'system': flattened_message += message_i.strip() + '\n\n'
    return flattened_message

def forward( 
        neuron: object, 
        roles: List[str],
        messages: List[str],
        timeout: float,
        num_to_query: Optional[int] = -1,
        random_sample_uids: Optional[ bool ] = True,
        train_gating_model: Optional[ bool ] = True,
        apply_backward: Optional[ bool ] = True,
        record_event: bool = True,
        is_synapse: Optional[ bool ] = False,
    ) -> SimpleNamespace:
    """ Apply a forward pass through the neuron.
        Args:   
            roles (List[str]): 
                List of role items, i.e. ['system', 'user', 'assistant']
            messages (List[str]): 
                List of associated messages. i.e. ['You are a chat bot', 'what time is it?', '10:35']
            timeout ( float ):
                The length of time before the query is stopped and the available responses are returned.
            num_to_query ( Optional[int] ):
                The number of endpoints to query, either randomly sampled or via the gating model.
                If -1 or None we query all uids.
            random_sample_uids ( Optional[ bool ] ):
                If True, we randomly sample the uids to query randomly from the set of uids.
            train_gating_model ( Optional[ bool ] ):
                If True, we apply a backward pass on the gating model to train it via the reward model.
            apply_backward ( Optional[ bool ] ):
                If True, rewards from the reward model are passed backward on the wire at the end of each
                forward pass.
            record_event ( Optional[ bool ] ):
                If True, the neuron records the forward event to history with inputs and outputs.
            is_synapse ( Optional[ bool ] ):
                For recording the whether this query is from the synapse or not.
        Returns:
            fpass (SimpleNamespace):
                A namespace containing the forward pass state and completion.
    """
    # Create our forward pass state which we will fill for record keeping.
    fpass = SimpleNamespace()
    fpass.start_time = time.time()
    fpass.success = False
    fpass.roles = roles
    fpass.messages = messages
    fpass.timeout = timeout
    fpass.num_to_query = num_to_query
    fpass.random_sample_uids = random_sample_uids
    fpass.train_gating_model = train_gating_model
    fpass.apply_backward = apply_backward
    fpass.is_synapse = is_synapse
    fpass.hotkeys = copy.deepcopy( neuron.metagraph.hotkeys )
    fpass.block = neuron.metagraph.block.item()

    # Get flattened representation of message.
    fpass.flattened_inputs = flattened_messages_for_gating( roles, messages )

    # Set `num_to_query` to the number of items in `neuron.metagraph.n` if `num_to_query` is not provided or is -1.
    # Find the available `uids` that are currently serving.
    # If `num_to_query` is larger than the number of available `uids`, set `num_to_query` to the number of available `uids`.
    fpass.available_uids = torch.tensor( [ uid for uid, ax in enumerate( neuron.metagraph.axons ) if ax.is_serving ], dtype = torch.int64 ).to( neuron.device )
    if fpass.num_to_query is None or fpass.num_to_query == -1: fpass.num_to_query = neuron.metagraph.n.item()
    if fpass.num_to_query > len( fpass.available_uids ): fpass.num_to_query = len( fpass.available_uids )
    if len( fpass.available_uids ) == 0: 
        return fpass

    # We run the gating network here to get the best uids
    # Use the gating model to generate scores for each `uid`.
    fpass.scores = neuron.gating_model( fpass.flattened_inputs ).to( neuron.device )

    if fpass.random_sample_uids:
        # Randomly sample the `uids_to_query` from the available uids.
        sample = random.sample( fpass.available_uids.tolist(), fpass.num_to_query )
        fpass.uids_to_query = torch.tensor( sample, dtype = torch.int64 ).to( neuron.device )
    else:
        # Select the top `uids_to_query` based on the highest `scores`.
        fpass.uids_to_query = fpass.available_uids[ fpass.scores[ fpass.available_uids ].sort()[ 1 ][ -fpass.num_to_query: ]]

    # Use the selected `uids_to_query` to query the dendrite pool.
    forward_calls = neuron.dendrite_pool( 
        roles = fpass.roles, 
        messages = fpass.messages, 
        uids = fpass.uids_to_query, 
        timeout = fpass.timeout,
    )

    # Filter out None, non responsive, or empty completions.
    fpass.successful_uids = []
    fpass.successful_completions = []
    for uid, call in list( zip( fpass.uids_to_query, forward_calls ) ):
        if call is not None and call.completion is not None and len(call.completion) > 0:
            fpass.successful_uids.append( uid.item() )
            fpass.successful_completions.append( call.completion )
    fpass.successful_uids = torch.tensor( fpass.successful_uids, dtype = torch.int64 )   
    if len( fpass.successful_completions ) == 0: 
        return fpass

    # Calculate the rewards for the successful `completions` using the reward model.
    if not neuron.config.neuron.no_reward_model:
        # First format the completions.
        formatted_completions = [ 
            flattened_messages_for_reward( roles, messages, completion ) 
            for completion in fpass.successful_completions 
        ]
        # Apply the completions to the reward model.
        fpass.rewards = neuron.reward_model.reward( 
            formatted_completions
        ).to( neuron.device )
    else: 
        # Instead we use the scoring model instead of the reward model.
        fpass.rewards = fpass.scores[ fpass.successful_uids ]

    # Train the gating model using the scores and rewards of the successful `completions`.
    if train_gating_model and not neuron.config.neuron.no_reward_model:
        # Note we cannot train the gating if we didnt use the reward model.
        neuron.gating_model.backward( scores = fpass.scores[ fpass.successful_uids ], rewards = fpass.rewards )

    # Pass rewards backward for potential PPO.
    if apply_backward and not neuron.config.neuron.no_reward_model:
        # Note we cannot send backward if we are not using the reward model.
        neuron.dendrite_pool.backward( 
            forward_calls = forward_calls,
            rewards = fpass.rewards,
        )

    # Set best result as completion
    fpass.completion = fpass.successful_completions[ fpass.rewards.argmax( dim = 0 ) ]

    # Return fpass result.
    fpass.success = True
    fpass.elpased = time.time() - fpass.start_time

    # Record forward pass.
    if record_event:
        neuron.history.put( fpass )
    log_event( neuron, fpass )        
    
    return fpass

# Save event to logging file and to history.
def log_event( neuron, event: SimpleNamespace ):
    bittensor.logging.trace( 'fpass', event )
    if not neuron.config.neuron.dont_log_events_to_file: 
        logger.log(
            "EVENTS", 
            "events", 
            success = event.success,
            roles = event.roles,
            messages = event.messages,
            timeout = event.timeout,
            num_to_query = event.num_to_query,
            random_sample_uids = event.random_sample_uids,
            train_gating_model = event.train_gating_model,
            apply_backward = event.apply_backward,
            is_synapse = event.is_synapse,
            block = event.block,
            scores = event.scores.tolist(),
            uids_to_query = event.uids_to_query.tolist(),
            successful_uids = event.successful_uids.tolist(),
            successful_completions = event.successful_completions,
            rewards = event.rewards.tolist(),
            completion = event.completion,
            elpased = event.elpased
        )

# Setup the log sink for events.
def setup_events_sink( config ):
    if not config.neuron.dont_log_events_to_file:
        # Add custom event logger for the events.
        log_format = '''{time:YYYY-MM-DD at HH:mm:ss} |
        {level} | 
        {message} | 
        {extra[success]} 
        {extra[roles]} 
        {extra[messages]} 
        {extra[timeout]} 
        {extra[num_to_query]} 
        {extra[random_sample_uids]} 
        {extra[train_gating_model]} 
        {extra[apply_backward]}
        {extra[is_synapse]}
        {extra[block]}
        {extra[scores]}
        {extra[uids_to_query]}
        {extra[successful_uids]}
        {extra[successful_completions]}
        {extra[rewards]}
        {extra[elpased]}
        {extra[completion]}'''
        logger.level("EVENTS", no=38, icon="📝")
        logger.add( 
            config.neuron.full_path + "/" + "events.log", 
            rotation = config.neuron.events_retention_size, serialize=True, enqueue=True, backtrace=False, diagnose=False, level="EVENTS", 
            format = log_format
        )