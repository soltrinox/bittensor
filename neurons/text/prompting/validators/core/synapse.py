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
import torch
import bittensor
from forward import forward
from typing import List, Optional, Tuple, Dict

# Format the messages to a list of roles and messages.
def process_messages( messages: List[Dict[str, str]] ) -> Tuple[ List[str], List[str] ]:
    roles = []; contents = []; 
    for message_i in messages:
        message_dict = json.loads( message_i )
        roles.append( message_dict['role'] )
        contents.append( message_dict['content'] )
    return roles, contents

def synapse_priority( neuron, forward_call: "bittensor.TextPromptingForwardCall" ) -> float:
    if forward_call.src_hotkey == neuron.wallet.hotkey.ss58_address: return math.inf # myself.
    elif forward_call.src_hotkey in neuron.my_nominators: return neuron.my_nominators[ forward_call.src_hotkey ].tao # Delegates.
    else: return 0.0 # Everyone else.

def synapse_blacklist( neuron, forward_call: "bittensor.TextPromptingForwardCall" ) -> bool:
    if forward_call.src_hotkey == neuron.wallet.hotkey.ss58_address: return True
    elif forward_call.src_hotkey in neuron.my_nominators: return False # Delegates, dont blacklist.
    else: return False # Everyone else, dont blacklist.

def synapse_forward( neuron, messages: List[Dict[str, str]] ) -> str: 
    bittensor.logging.success( "inference forward()" )
    roles, messages = process_messages( messages )
    return forward( 
        neuron = neuron,
        roles = roles,
        messages = messages,
        timeout = neuron.config.neuron.inference_timeout,
        num_to_query = neuron.config.neuron.inference_topk,
        random_sample_uids = False,
        train_gating_model = True,
        apply_backward = False,
        record_event = True,
        is_synapse = True
    ).completion

def synapse_backward( neuron, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: 
    pass