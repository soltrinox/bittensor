import json
import pytest
import torch
import bittensor
from unittest.mock import patch
from bittensor._subtensor.subtensor_mock import mock_subtensor
from typing import List, Dict, Union, Tuple


format_messages = lambda roles, messages: [{'role': roles[i], 'content': messages[i]} for i in range(len(messages))]

class Synapse( bittensor.TextPromptingSynapse ):
    def priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
        return 0.0

    def blacklist(self, forward_call: "bittensor.TextPromptingForwardCall") -> Union[ Tuple[bool, str], bool ]:
        return False

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str:
        pass

    def forward(self, messages: List[Dict[str, str]]) -> str:
        return "hello im a chat bot."

    def multi_forward(self, messages: List[Dict[str, str]]) -> List[ str ]:
        return ["hello im a chat bot.", "my name is bob" ]


def test_forward():
    # Create a mock wallet
    wallet = bittensor.wallet().create_if_non_existent()
    axon = bittensor.axon( wallet = wallet, port = 9090, ip = "127.0.0.1", metagraph = None )
    axon.start()

    # Create the dendrite
    dendrite = bittensor.text_prompting( axon = axon.info(), keypair = wallet.hotkey )

    # Define call object
    messages = ['you are chat bot', 'what is the whether']
    roles = ['system', 'assistant']

    # Forward pass
    fcall = dendrite.forward( roles, messages )

    assert fcall.completed == True
    assert fcall.dest_hotkey == wallet.hotkey.ss58_address
    assert fcall.is_forward == True
    assert fcall.dendrite == dendrite
    assert fcall.messages == messages
    assert fcall.roles == roles
    assert fcall.completion == ''
    assert fcall.get_inputs_shape() == torch.Size([49, 55])
    assert fcall.get_outputs_shape() == torch.Size([0])

    formatted_messages = format_messages(roles, messages)

    request_proto = fcall._get_request_proto()
    assert request_proto.version == bittensor.__version_as_int__
    assert request_proto.hotkey == wallet.hotkey.ss58_address
    assert request_proto.timeout == fcall.timeout

    # messages
    parsed_dicts = [json.loads(x) for x in request_proto.messages]
    assert parsed_dicts == formatted_messages

    # Ensure this passes, no need to inspect
    _ = dendrite.apply( fcall )

    # Alternate fcall
    _ = fcall.get_callable()(
        request = fcall._get_request_proto(),
        timeout = fcall.timeout,
        metadata = (
            ('rpc-auth-header','Bittensor'),
            ('bittensor-signature', dendrite.sign() ),
            ('bittensor-version',str(bittensor.__version_as_int__)),
        )
    )
    axon.stop()


def test_backward():
    # Create a mock wallet
    wallet = bittensor.wallet().create_if_non_existent()
    axon = bittensor.axon( wallet = wallet, port = 9090, ip = "127.0.0.1", metagraph = None )
    axon.start()

    # Create the dendrite
    dendrite = bittensor.text_prompting( axon = axon.info(), keypair = wallet.hotkey )

    # Define call object
    messages = ['you are chat bot', 'what is the whether']
    roles = ['system', 'assistant']
    completion = 'hello im a chat bot.'
    rewards = torch.tensor([1.0, 0.0])

    # Backward pass
    bcall = dendrite.backward( roles, messages, completion, rewards )

    assert bcall.completed == True
    assert bcall.dest_hotkey == wallet.hotkey.ss58_address
    assert bcall.is_forward == False
    assert bcall.dendrite == dendrite
    assert bcall.messages == messages
    assert bcall.roles == roles
    assert bcall.completion == completion
    assert bcall.get_inputs_shape() == torch.Size([49, 55])
    assert bcall.get_outputs_shape() == torch.Size([0])

    formatted_messages = format_messages(roles, messages)

    request_proto = bcall._get_request_proto() # backward request proto
    assert request_proto.hotkey == wallet.hotkey.ss58_address
    assert request_proto.timeout == bcall.timeout
    assert request_proto.rewards == rewards.tolist()
    assert request_proto.response == completion

    # messages
    parsed_dicts = [json.loads(x) for x in request_proto.messages]
    assert parsed_dicts == formatted_messages

    # Ensure this passes, no need to inspect
    _ = dendrite.apply( bcall )

    # Alternate fcall
    _ = bcall.get_callable()(
        request = bcall._get_request_proto(),
        timeout = bcall.timeout,
        metadata = (
            ('rpc-auth-header','Bittensor'),
            ('bittensor-signature', dendrite.sign() ),
            ('bittensor-version',str(bittensor.__version_as_int__)),
        )
    )
    axon.stop()
