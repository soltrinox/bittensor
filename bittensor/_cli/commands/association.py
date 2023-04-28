# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

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

import argparse
import bittensor
from rich.prompt import Prompt
console = bittensor.__console__

class AssociateCommand:

    @staticmethod
    def run( cli ):
        r""" Associates a wallet's coldkey with its hotkey.
        """
        config = cli.config.copy()
        wallet = bittensor.wallet( config = config )
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )

        subtensor.associate(
            wallet = wallet,
            wait_for_inclusion = True,
            wait_for_finalization= True,
            prompt=not config.no_prompt
        )
    

    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
             

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        associate_parser = parser.add_parser(
            'associate', 
            help='''Associate your keys on the chain.'''
        )
        associate_parser.add_argument( 
            '--no_version_checking', 
            action='store_true', 
            help='''Set false to stop cli version checking''', 
            default = False 
        )
        associate_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( associate_parser )
        bittensor.subtensor.add_args( associate_parser )

class DisassociateCommand:

    @staticmethod
    def run( cli ):
        r""" Disassociates a wallet's coldkey from its hotkey.
        """
        config = cli.config.copy()
        wallet = bittensor.wallet( config = config )
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )

        subtensor.disassociate(
            wallet = wallet,
            associate_ss58=wallet.hotkey.ss58_address,
            wait_for_inclusion = True,
            wait_for_finalization= True,
            prompt=not config.no_prompt
        )
    

    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
             

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        disassociate_parser = parser.add_parser(
            'disassociate', 
            help='''Disassociate your keys on the chain.'''
        )
        disassociate_parser.add_argument( 
            '--no_version_checking', 
            action='store_true', 
            help='''Set false to stop cli version checking''', 
            default = False 
        )
        disassociate_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( disassociate_parser )
        bittensor.subtensor.add_args( disassociate_parser )