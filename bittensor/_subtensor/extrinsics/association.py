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
import bittensor

from rich.prompt import Confirm
from bittensor.utils import is_valid_bittensor_address_or_public_key

def associate_extrinsic(
        subtensor: 'bittensor.Subtensor',
        wallet: 'bittensor.Wallet',
        associate_ss58: str,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
    r""" Associates a new public key with the coldkey of this wallet.
    Requires: 
        - wallet.coldkey must be able to sign the transaction.
        - wallet.coldkey must have sufficient balance to pay the transaction fee.
        - wallet.hotkey must match associate_ss58.
        - wallet.hotkey must be able to sign the transaction
    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object to associate with.
        associate_ss58 (str):
            SS58 encoded public key address of the new public key to associate with this wallet.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning true, 
            or returns false if the extrinsic fails to enter the block within the timeout.   
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call prompts the user for confirmation before proceeding.
    Returns:
        success (bool):
            Flag is true if extrinsic was finalized or included in the block. 
            If we did not wait for finalization / inclusion, the response is true.
            If the keys are already associated, the response is true.
    Raises:
        ValueError:
            If the hotkey does not match the associate_ss58.
    """ 
    wallet.hotkey

    if wallet.hotkey.ss58_address != associate_ss58:
        bittensor.__console__.print(":cross_mark: [red]Hotkey does not match address[/red]:[bold white]\n  hotkey: {}\n  address: {}[/bold white]".format( wallet.hotkey.ss58_address, associate_ss58 ))
        raise ValueError("Hotkey does not match associate address")

    # Validate destination address.
    if not is_valid_bittensor_address_or_public_key( associate_ss58 ):
        bittensor.__console__.print(":cross_mark: [red]Invalid SS58 address[/red]:[bold white]\n  {}[/bold white]".format(associate_ss58))
        return False
    
    with bittensor.__console__.status(":satellite: Checking Association..."):
        is_associated = subtensor.is_hotkey_owner( wallet.coldkeypub.ss58_address, associate_ss58 )
        if is_associated:
            # Already associated, return true.
            bittensor.__console__.print(":white_heavy_check_mark: [green]Already Associated[/green]")
            return True
    
    # Unlock wallet coldkey.
    wallet.coldkey

    # Check balance.
    with bittensor.__console__.status(":satellite: Checking Balance..."):
        account_balance = subtensor.get_balance( wallet.coldkey.ss58_address )
        # check existential deposit.
        existential_deposit = subtensor.get_existential_deposit()

    # Sign payload of coldkey bytes with hotkey.
    signed_coldkey = wallet.hotkey.sign( wallet.coldkeypub.public_key )

    with bittensor.__console__.status(":satellite: Checking Fee..."):
        with subtensor.substrate as substrate:     
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='associate',
                call_params={
                    'hotkey': associate_ss58,
                    'signed_coldkey': signed_coldkey,
                }
            )

            try:
                payment_info = substrate.get_payment_info( call = call, keypair = wallet.coldkey )
            except Exception as e:
                bittensor.__console__.print(":cross_mark: [red]Failed to get payment info[/red]:[bold white]\n  {}[/bold white]".format(e))
                payment_info = {
                    'partialFee': 2e7, # assume  0.02 Tao 
                }

            fee = bittensor.Balance.from_rao( payment_info['partialFee'] )
        

    # Check if we have enough balance.
    if account_balance < (fee + existential_deposit):
        bittensor.__console__.print(":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance: {}\n for fee: {}[/bold white]".format( account_balance, fee ))
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask("Do you want to associate:[bold white]\n  associate_ss58: {}\n  to  {}:{}\n  for fee: {}[/bold white]".format( associate_ss58, wallet.name, wallet.coldkey.ss58_address, fee )):
            return False

    with bittensor.__console__.status(":satellite: Creating Association..."):
        with subtensor.substrate as substrate:
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                return True

            # Otherwise continue with finalization.
            response.process_events()
            if response.is_success:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
            else:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

    if response.is_success:
        with bittensor.__console__.status(":satellite: Checking Balance..."):
            new_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
            bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(account_balance, new_balance))

        with bittensor.__console__.status(":satellite: Checking Association..."):
            is_associated = subtensor.is_hotkey_owner( coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=associate_ss58 )
            if not is_associated:
                bittensor.__console__.print(":cross_mark: [red]Failed to associate[/red]:[bold white]\n  {}[/bold white]".format(associate_ss58))
                return False
            else:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Successfully Associated![/green]")
                return True
    return False


def disassociate_extrinsic(
        subtensor: 'bittensor.Subtensor',
        wallet: 'bittensor.wallet',
        associate_ss58: str, 
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        prompt: bool = False,
    ) -> bool:
    r""" Disassociates a public key from the coldkey of this wallet.
    Requires: 
        - wallet.coldkey must be able to sign the transaction.
        - wallet.coldkey must have sufficient balance to pay the transaction fee.
    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object to disassociate from.
        associate_ss58 (str):
            SS58 encoded public key address of the new public key to diassociate from this wallet.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning true, 
            or returns false if the extrinsic fails to enter the block within the timeout.   
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call prompts the user for confirmation before proceeding.
    Returns:
        success (bool):
            Flag is true if extrinsic was finalized or included in the block. 
            If we did not wait for finalization / inclusion, the response is true.
            If the keys are not associated, the response is true.
    """ 
    # Validate destination address.
    if not is_valid_bittensor_address_or_public_key( associate_ss58 ):
        bittensor.__console__.print(":cross_mark: [red]Invalid SS58 address[/red]:[bold white]\n  {}[/bold white]".format(associate_ss58))
        return False
    
    with bittensor.__console__.status(":satellite: Checking Association..."):
        is_associated = subtensor.is_hotkey_owner( wallet.coldkeypub.ss58_address, associate_ss58 )
        if not is_associated:
            # Already disassociated, return true.
            bittensor.__console__.print(":white_heavy_check_mark: [green]Already Disassociated[/green]")
            return True
    
    # Unlock wallet coldkey.
    wallet.coldkey

    # Check balance.
    with bittensor.__console__.status(":satellite: Checking Balance..."):
        account_balance = subtensor.get_balance( wallet.coldkey.ss58_address )
        # check existential deposit.
        existential_deposit = subtensor.get_existential_deposit()


    with bittensor.__console__.status(":satellite: Checking Fee..."):
        with subtensor.substrate as substrate:     
            call = substrate.compose_call(
                call_module='SubtensorModule',
                call_function='disassociate',
                call_params={
                    'hotkey': associate_ss58,
                }
            )

            try:
                payment_info = substrate.get_payment_info( call = call, keypair = wallet.coldkey )
            except Exception as e:
                bittensor.__console__.print(":cross_mark: [red]Failed to get payment info[/red]:[bold white]\n  {}[/bold white]".format(e))
                payment_info = {
                    'partialFee': 2e7, # assume  0.02 Tao 
                }

            fee = bittensor.Balance.from_rao( payment_info['partialFee'] )
        

    # Check if we have enough balance.
    if account_balance < (fee + existential_deposit):
        bittensor.__console__.print(":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance: {}\n for fee: {}[/bold white]".format( account_balance, fee ))
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask("Do you want to disassociate:[bold white]\n  associate_ss58: {}\n  from  {}:{}\n  for fee: {}[/bold white]".format( associate_ss58, wallet.name, wallet.coldkey.ss58_address, fee )):
            return False

    with bittensor.__console__.status(":satellite: Removing Association..."):
        with subtensor.substrate as substrate:
            extrinsic = substrate.create_signed_extrinsic( call = call, keypair = wallet.coldkey )
            response = substrate.submit_extrinsic( extrinsic, wait_for_inclusion = wait_for_inclusion, wait_for_finalization = wait_for_finalization )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                return True

            # Otherwise continue with finalization.
            response.process_events()
            if response.is_success:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Finalized[/green]")
            else:
                bittensor.__console__.print(":cross_mark: [red]Failed[/red]: error:{}".format(response.error_message))

    if response.is_success:
        with bittensor.__console__.status(":satellite: Checking Balance..."):
            new_balance = subtensor.get_balance( wallet.coldkey.ss58_address )
            bittensor.__console__.print("Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(account_balance, new_balance))

        with bittensor.__console__.status(":satellite: Checking Association..."):
            is_associated = subtensor.is_hotkey_owner( wallet.coldkey.ss58_address, associate_ss58 )
            if is_associated:
                bittensor.__console__.print(":cross_mark: [red]Failed to disassociate[/red]:[bold white]\n  {}[/bold white]".format(associate_ss58))
                return False
            else:
                bittensor.__console__.print(":white_heavy_check_mark: [green]Successfully Disassociated![/green]")
                return True
    return False
    
