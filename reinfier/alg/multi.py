
from .. import util
from .. import drlp
from .. import CONSTANT
from ..nn.NN import NN
from ..drlp.DRLP import DRLP
from .single import *
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


def get_coordinate(dims, id):
    coordinate = []
    for dim in reversed(dims):
        coordinate.append(id % dim)
        id //= dim
    return list(reversed(coordinate))


def get_id(dims, coordinate):
    id = 0
    base = 1
    for i in range(len(dims) - 1, -1, -1):
        id += (base * coordinate[i])
        base *= dims[i]
    return id


def get_dims(variables):
    dims = []
    for k, v in variables.items():
        if drlp.parser.is_iterable_variable(k):
            dims.append(len(v))
    return dims


def verify_linear(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> List[Tuple[DRLP, int, bool]]:
    drlps = drlp.parse_drlp_v(property)

    ans = []
    for drlp_ in drlps:
        k, result = verify(network, drlp_, verifier=verifier, k_max=k_max, k_min=k_min)

        ans.append((drlp_, k, result))

        util.log((drlp_, drlp_.kwargs, k, result), level=CONSTANT.INFO)
        util.log_prompt(3)
    return ans


def continue_verify(result):  # TODO: Judge from Variable type
    return result == True


def verify_cubic(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> List[Tuple[DRLP, int, bool]]:
    drlps = drlp.parse_drlp_v(property)
    variables = drlp.parser.get_variables(property)
    dims = get_dims(variables)

    ans = []
    i = 0
    drlps_len = len(drlps)

    while i < drlps_len:
        coordinate = get_coordinate(dims, i)
        drlp_ = drlps[i]

        k, result = verify(network, drlp_, verifier=verifier, k_max=k_max, k_min=k_min)

        ans.append((drlp_, k, result))

        util.log((drlp_, drlp_.kwargs, k, result), level=CONSTANT.INFO)
        util.log_prompt(3)
        if continue_verify(result):
            i += 1
        else:
            i += dims[-1] - coordinate[-1]

    return ans
