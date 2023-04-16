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
import grpc
import time
import json
import torch
import asyncio
import bittensor
from typing import List, Union

from dataclasses import dataclass

@dataclass
class TextPromptingDendriteForwardCall:

    _packed_messages: List[ str ] = None
    _response_proto: bittensor.proto.ForwardTextPromptingResponse = None
    _request_proto: bittensor.proto.ForwardTextPromptingRequest = None
    _return_code: bittensor.proto.ReturnCode = None
    _return_message: str = None
    _completion: str = None

    def __init__( self, receptor, roles, messages, timeout ):
        self.receptor = receptor
        self.roles = roles
        self.messages = messages 
        self.timeout = timeout
        self.completed = False

    @property
    def src_hotkey( self ) -> str:
        return self.receptor.wallet.hotkey.ss58_address

    @property
    def dest_hotkey( self ) -> str:
        return self.receptor.endpoint.hotkey
    
    @property
    def dest_uid( self ) -> str:
        return self.receptor.endpoint.uid

    @property
    def packed_messages( self ) -> List[ str ]:
        if self._packed_messages is None:
            self._packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(self.roles, self.messages)]
        return self._packed_messages
    
    @property
    def return_code( self ) -> bittensor.proto.ReturnCode:
        if self._return_code is None: self.call()
        return self._return_code
    
    @property
    def return_message( self ) -> bittensor.proto.ReturnCode:
        if self._return_message is None: self.call()
        return self._return_message
    
    @property
    def response_proto( self ) -> bittensor.proto.ForwardTextPromptingResponse:
        if self._response_proto is None: self.call()
        return self._response_proto
    
    @property
    def completion( self ) -> str:
        if self._completion is None: self.call()
        return self._completion

    @property
    def request_proto( self ) -> bittensor.proto.ForwardTextPromptingRequest:
        if self._request_proto is None:
            self._packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(self.roles, self.messages)]
            self._request_proto = bittensor.ForwardTextPromptingRequest( 
                messages = self.packed_messages, 
                timeout = self.timeout, 
                hotkey = self.src_hotkey,
                version = bittensor.__version_as_int__
            )
        return self._request_proto
    
    @property
    def asyncio_future( self ) -> asyncio.Future:
        if self._asyncio_future is None:
            self._asyncio_future = bittensor.grpc.TextPromptingStub( self.receptor.channel ).Forward(
                request = self.request_proto,
                timeout = self.timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.receptor.sign() ),
                    ('bittensor-version',str( bittensor.__version_as_int__ )),
                )
            )
        return self._asyncio_future
    

    async def async_call( self ):
        if self.completed: return
        else:
            self.start_time = time.time()
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = True, 
                is_response = False, 
                code = bittensor.proto.ReturnCode.Success, 
                call_time = 0, 
                pubkey = self.dest_hotkey, 
                uid = self.dest_uid, 
                inputs = torch.Size( [len(message) for message in self.packed_messages ] ),
                outputs = None,
                message = "Success",
                synapse = "text_prompting"
            )
            try:
                self._response_proto = await asyncio.wait_for( self.asyncio_future, timeout = self.timeout )
                self._return_code = bittensor.proto.ReturnCode.Success
                self._return_message = "Success"

            # Catch grpc errors.
            except grpc.RpcError as rpc_error_call:
                # Request failed with GRPC code.
                self._return_code = rpc_error_call.code()
                self._return_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
                self._completion = ""

            # Catch timeout errors
            except asyncio.TimeoutError:
                # Catch timeout errors.
                self._return_code = bittensor.proto.ReturnCode.Timeout
                self._return_message = 'GRPC request timeout after: {}s'.format( self.timeout )
                self._completion = ""

            # Catch unknown exceptions.
            except Exception as e:
                # Catch unknown errors.
                self._return_code = bittensor.proto.ReturnCode.UnknownException
                self._return_message = str( e )
                self._completion = ""
            
            finally:
                self.completed = True
                self.end_time = time.time()
                bittensor.logging.rpc_log(
                    axon = False, 
                    forward = True, 
                    is_response = True, 
                    code = self._return_code, 
                    call_time = self.end_time - self.start_time, 
                    pubkey = self.dest_hotkey, 
                    uid = self.dest_uid, 
                    inputs = torch.Size( [len(message) for message in self.packed_messages ] ), 
                    outputs = torch.Size([len( self._completion )]),
                    message = self._return_message,
                    synapse = "text_prompting",
                )

    def call( self ): 
        loop = asyncio.get_event_loop()
        loop.run_until_complete( self.async_call() ) 



@dataclass
class TextPromptingDendriteBackwardCall:

    _packed_messages: List[ str ] = None
    _response_proto: bittensor.proto.ForwardTextPromptingResponse = None
    _request_proto: bittensor.proto.ForwardTextPromptingRequest = None
    _return_code: bittensor.proto.ReturnCode = None
    _return_message: str = None

    def __init__( 
            self,
            dendrite: bittensor.text_prompting,
            roles: List[ str ],
            messages: List[ str ],
            completion: str,
            rewards: Union[ List[ float], torch.FloatTensor ],
            timeout: float = bittensor.__blocktime__
        ):
        self.dendrite = dendrite
        self.roles = roles
        self.messages = messages 
        self.rewards = rewards.tolist() if isinstance( rewards, torch.FloatTensor ) else rewards
        self.completion = completion
        self.timeout = timeout
        self.completed = False

    @property
    def src_hotkey( self ) -> str:
        return self.dendrite.wallet.hotkey.ss58_address

    @property
    def dest_hotkey( self ) -> str:
        return self.dendrite.endpoint.hotkey
    
    @property
    def dest_uid( self ) -> str:
        return self.dendrite.endpoint.uid

    @property
    def packed_messages( self ) -> List[ str ]:
        if self._packed_messages is None:
            self._packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(self.roles, self.messages)]
        return self._packed_messages
    
    @property
    def return_code( self ) -> bittensor.proto.ReturnCode:
        if self._return_code is None: self.call()
        return self._return_code
    
    @property
    def return_message( self ) -> bittensor.proto.ReturnCode:
        if self._return_message is None: self.call()
        return self._return_message
    
    @property
    def response_proto( self ) -> bittensor.proto.ForwardTextPromptingResponse:
        if self._response_proto is None: self.call()
        return self._response_proto
    
    @property
    def request_proto( self ) -> bittensor.proto.ForwardTextPromptingRequest:
        if self._request_proto is None:
            self._packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(self.roles, self.messages)]
            self._request_proto = bittensor.BackwardTextPromptingRequest( 
                messages = self._packed_messages, 
                response = self.completion,
                rewards = self.rewards,
                hotkey = self.src_hotkey,
                version = bittensor.__version_as_int__
            )
        return self._request_proto
    
    @property
    def asyncio_future( self ) -> asyncio.Future:
        if self._asyncio_future is None:
            self._asyncio_future = bittensor.grpc.TextPromptingStub( self.dendrite.receptor.channel ).Backward(
                request = self.request_proto,
                timeout = self.timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.dendrite.receptor.sign() ),
                    ('bittensor-version',str( bittensor.__version_as_int__ )),
                )
            )
        return self._asyncio_future
    

    async def async_call( self ):
        if self.completed: return
        else:
            self.start_time = time.time()
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = False, 
                is_response = False, 
                code = bittensor.proto.ReturnCode.Success, 
                call_time = 0, 
                pubkey = self.dest_hotkey, 
                uid = self.dest_uid, 
                inputs = torch.Size( [ len( self.rewards ) ] ),
                outputs = None,
                message = "Success",
                synapse = "text_prompting"
            )   
            try:
                self._response_proto = await asyncio.wait_for( self.asyncio_future, timeout = self.timeout )
                self._return_code = bittensor.proto.ReturnCode.Success
                self._return_message = "Success"

            # Catch grpc errors.
            except grpc.RpcError as rpc_error_call:
                # Request failed with GRPC code.
                self._return_code = rpc_error_call.code()
                self._return_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
                self._completion = ""

            # Catch timeout errors
            except asyncio.TimeoutError:
                # Catch timeout errors.
                self._return_code = bittensor.proto.ReturnCode.Timeout
                self._return_message = 'GRPC request timeout after: {}s'.format( self.timeout )
                self._completion = ""

            # Catch unknown exceptions.
            except Exception as e:
                # Catch unknown errors.
                self._return_code = bittensor.proto.ReturnCode.UnknownException
                self._return_message = str( e )
                self._completion = ""
            
            finally:
                self.completed = True
                self.end_time = time.time()
                bittensor.logging.rpc_log(
                    axon = False, 
                    forward = True, 
                    is_response = True, 
                    code = self._return_code, 
                    call_time = self.end_time - self.start_time, 
                    pubkey = self.dest_hotkey, 
                    uid = self.dest_uid, 
                    inputs = torch.Size( [len(message) for message in self.packed_messages ] ), 
                    outputs = torch.Size([len( self._completion )]),
                    message = self._return_message,
                    synapse = "text_prompting",
                )

    def call( self ): 
        loop = asyncio.get_event_loop()
        loop.run_until_complete( self.async_call() ) 