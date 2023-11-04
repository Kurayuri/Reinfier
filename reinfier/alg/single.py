from ..drlp.DRLP import DRLP
from ..nn.NN import NN
from ..import nn
from ..import interface
from ..import drlp
from .import lib
from .import selector
from typing import Tuple
import numpy


def bmc(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> Tuple[int, bool, numpy.ndarray]:
    lib.log_call(network, property, "bmc")

    if verifier is None:
        verifier = selector.select_verifier(network, property)

    violation = None

    for k in range(k_min, k_max + 1):

        dnn = nn.expander.unroll_nn(network, k, branchable=nn.lib.is_branchable(verifier))
        dnnp = drlp.parser.parse_vpq(property, k)[0]

        runable, result, time, violation = interface.dnnv.boot(dnn, dnnp, verifier)
        lib.log_call(k, runable, result, time, "base")

        if result == False:
            return False, k, violation
    # return None, k_max, violation
    return True, k_max, violation


def k_induction(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> Tuple[int, bool, numpy.ndarray]:
    lib.log_call(network, property, "k_induction")

    if verifier is None:
        verifier = selector.select_verifier(network, property)

    violation = None

    for k in range(k_min, k_max + 1):

        dnn = nn.expander.unroll_nn(network, k, branchable=nn.lib.is_branchable(verifier))
        dnnp = drlp.parser.parse_vpq(property, k)[0]

        runable, result, time, violation = interface.dnnv.boot(dnn, dnnp, verifier)
        lib.log_call(k, runable, result, time, "base")

        if result == True:

            dnn = nn.expander.unroll_nn(network, k + 1, branchable=nn.lib.is_branchable(verifier))
            dnnp = drlp.parser.parse_vpq(property, k, {}, True)[0]

            runable, result, time, violation = interface.dnnv.boot(dnn, dnnp, verifier)
            lib.log_call(k, runable, result, time, "induction")

            if result == True:
                return True, k, violation
            else:
                continue

        elif result == False:
            return False, k, violation

    return None, k_max, violation

def reach(network: NN, property: DRLP, k_max: int = 10, k_min: int = 1):
    runable, result, time, violation = interface.verisig.boot(network,property)
    lib.log_call(1, runable, result, time, "reach")
    return result, 1, violation


def verify(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1, to_induct=True, reachability=False) -> Tuple[bool, int, numpy.ndarray]:
    if reachability:
        return reach(network, property,  k_max=k_max, k_min=k_min)
    elif to_induct:
        return k_induction(network, property, verifier=verifier, k_max=k_max, k_min=k_min)
    else:
        return bmc(network, property, verifier=verifier, k_max=k_max, k_min=k_min)
