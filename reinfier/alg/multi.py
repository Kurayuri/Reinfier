from ..drlp.DRLP import DRLP
from ..nn.NN import NN
from ..import util
from ..import drlp
from ..import CONSTANT
from .single import *
from .lib import *
from typing import Tuple, List


class Variable:
    def __init__(self, id, values, type):
        self.id = id
        self.values = values
        self.type = type
        self.state = None
        self.exchange = 0

        if self.type == 'l':
            self.state = True
        elif self.type == 'b':
            self.state = False
        elif self.type == 'm':
            self.state = False
        elif self.type == 'e':
            self.state = True
        elif self.type == 'r':
            self.state = True

    def signal(self, sign):
        if self.state != sign:
            self.exchange += 1
        self.state = sign

        if self.type == "l":
            return self.exchange == 0
        elif self.type == "b":
            return self.exchange == 0
        elif self.type == "m":
            return self.exchange < 2
        elif self.type == "e":
            return self.exchange < 2
        elif self.type == "r":
            return True

        return None


def verify_linear(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> List[Tuple[DRLP, int, bool]]:
    property_pqs = drlp.parse_v(property)

    ans = []
    for property_pq in property_pqs:
        k, result = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        ans.append((property_pq, k, result))

        util.log((property_pq, property_pq.kwargs, k, result), level=CONSTANT.INFO)
        util.log_prompt(3)
    return ans


def search_boundary(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1, kwargs: dict = {}, accuracy: float = 1e-2) -> List[Tuple[DRLP, int, bool]]:
    variable, boundary = list(kwargs.items())[0]
    lower = boundary[0]
    upper = boundary[1]
    property = property.obj

    util.log_prompt(3)
    util.log("########## Init ##########\n", level=CONSTANT.WARNING)

    value = upper
    property_vpq = DRLP(property).overwrite(f"{variable}={value}")
    property_pq = drlp.parse_v(property_vpq)[0]
    k, result_upper = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

    value = lower
    property_vpq = DRLP(property).overwrite(f"{variable}={value}")
    property_pq = drlp.parse_v(property_vpq)[0]
    k, result_lower = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

    util.log(f"## Result:\nUpper@{upper}\t: {result_upper}\nLower @{lower}\t: {result_lower}\n", level=CONSTANT.WARNING)
    while upper - lower > accuracy:
        value = (upper + lower) / 2
        property_vpq = DRLP(property).overwrite(f"{variable}={value}")
        property_pq = drlp.parse_v(property_vpq)[0]
        k, result = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        if result == result_lower:
            lower = value
        elif result == result_upper:
            upper = value

        util.log_prompt(3)
        util.log("########## Step ##########\n", level=CONSTANT.WARNING)
        util.log(f"## Result:\nMid @ {value}\t: {result}\n", level=CONSTANT.WARNING)
    return value


def verify_cubic(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> List[Tuple[DRLP, int, bool]]:
    property_pqs = drlp.parse_v(property)
    variables = drlp.parser.get_variables(property)
    dims = get_dims(variables)

    ans = []
    i = 0
    property_pqs_len = len(property_pqs)

    while i < property_pqs_len:
        coordinate = get_coordinate(dims, i)
        property_pq = property_pqs[i]

        k, result = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        ans.append((property_pq, k, result))

        util.log((property_pq, property_pq.kwargs, k, result), level=CONSTANT.INFO)
        util.log_prompt(3)
        if continue_verify(result):
            i += 1
        else:
            if len(dims) == 0:
                break
            i += dims[-1] - coordinate[-1]

    return ans
