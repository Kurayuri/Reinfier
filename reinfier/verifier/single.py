from ..common.classes import VerificationAnswer
from ..common.DRLP import DRLP
from ..common.NN import NN
from ..import nn
from ..import interface
from ..import drlp
from ..import CONST
from .import lib
from .import selector
from typing import Tuple
import numpy


def bmc(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> VerificationAnswer:
    lib.log_call("bmc", network, property)

    if verifier is None:
        verifier = selector.select_verifier(network, property)

    violation = None

    for k in range(k_min, k_max + 1):

        dnn = nn.expander.unroll_nn(network, k, branchable=nn.lib.is_branchable(verifier))
        dnnp = drlp.parser.parse_vpq(property, k)[0]

        runable, result, time, violation = interface.dnnv.boot(dnn, dnnp, verifier) \
            if verifier != CONST.MARABOU else interface.marabou.boot(dnn, property)

        lib.log_call("base_ans", k, runable, result, time)

        if result == False:
            return False, k, violation
    # return None, k_max, violation
    return True, k_max, violation


def k_induction(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> VerificationAnswer:
    lib.log_call("k_induction", network, property)

    if verifier is None:
        verifier = selector.select_verifier(network, property)

    violation = None

    for k in range(k_min, k_max + 1):

        dnn = nn.expander.unroll_nn(network, k, branchable=nn.lib.is_branchable(verifier))
        dnnp = drlp.parser.parse_vpq(property, k)[0]

        runable, result, time, violation = interface.dnnv.boot(dnn, dnnp, verifier) \
            if verifier != CONST.MARABOU else interface.marabou.boot(dnn, property)

        lib.log_call("base_ans", k, runable, result, time)

        if result == True:

            dnn = nn.expander.unroll_nn(network, k + 1, branchable=nn.lib.is_branchable(verifier))
            dnnp = drlp.parser.parse_vpq(property, k, {}, True)[0]

            runable, result, time, violation = interface.dnnv.boot(dnn, dnnp, verifier) \
                if verifier != CONST.MARABOU else interface.marabou.boot(dnn, property)
            lib.log_call("induction_ans", k, runable, result, time)

            if result == True:
                return True, k, violation
            else:
                continue

        elif result == False:
            return False, k, violation

    return None, k_max, violation


def reach(network: NN, property: DRLP, k_max: int = 10, k_min: int = 1):
    runable, result, time, violation = interface.verisig.boot(
        network, property)
    lib.log_call(1, runable, result, time, "reach")
    return result, 1, violation


def verify(network: NN, property: DRLP,
           verifier: str = None, k_max: int = 10, k_min: int = 1, to_induct=True, reachability=False) -> VerificationAnswer:
    """Verify a neural network against a given property using a specified verifier.

    Args:
        network (NN): The neural network to be verified.
        property (DRLP): The property to verify against the neural network.
        verifier (str, optional): The verifier to use for the verification process. Defaults to None.
        k_max (int, optional): The maximum value of k for the verification process. Defaults to 10.
        k_min (int, optional): The minimum value of k for the verification process. Defaults to 1.
        to_induct (bool, optional): Flag to indicate if induction should be used. Defaults to True.
        reachability (bool, optional): Flag to indicate if reachability analysis should be performed. Defaults to False.

    Returns:
        VerificationAnswer: The result of the verification process, including:
            - result (bool): True if the property is verified, False otherwise.
            - depth (int): The depth at which the verification process concluded.
            - violation (np.ndarray): The array representing the violation if the property is not verified.
    """
    if reachability:
        return reach(network, property, k_max=k_max, k_min=k_min)
    elif to_induct:
        return k_induction(network, property, verifier=verifier, k_max=k_max, k_min=k_min)
    else:
        return bmc(network, property, verifier=verifier, k_max=k_max, k_min=k_min)
