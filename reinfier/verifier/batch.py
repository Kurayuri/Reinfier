from ..common.DRLP import DRLP
from ..common.NN import NN
from ..import util
from ..import drlp
from ..import CONST
from .single import *
from .lib import *
from typing import Tuple, List, Union
from copy import deepcopy


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
        result, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        ans.append((property_pq, k, result))

        util.log((property_pq, property_pq.kwargs, k, result), level=CONST.INFO)
        util.log_prompt(3)
    return ans


def verify_hypercubic(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> List[Tuple[DRLP, int, bool]]:
    property_pqs = drlp.parse_v(property)
    variables = drlp.parser.get_variables(property)
    dims = get_dims(variables)

    ans = []
    i = 0
    property_pqs_len = len(property_pqs)

    while i < property_pqs_len:
        coordinate = get_coordinate(dims, i)
        property_pq = property_pqs[i]

        result, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        ans.append((property_pq, k, result))

        util.log((property_pq, property_pq.kwargs, k, result), level=CONST.INFO)
        util.log_prompt(3)
        if continue_verify(result):
            i += 1
        else:
            if len(dims) == 0:
                break
            i += dims[-1] - coordinate[-1]

    return ans


def search_boundary_dichotomy(network: NN, property: DRLP, kwargs: dict, accuracy: float = 1e-2, verifier: str = None, k_max: int = 10, k_min: int = 1) -> float:
    variable, bounds = list(kwargs.items())[0]
    lower = bounds[0]
    upper = bounds[1]
    property = property.obj

    util.log_prompt(3)
    util.log("########## Init ##########\n", level=CONST.WARNING)

    value = upper
    property_vpq = DRLP(property).append(f"{variable}={value}")
    property_pq = drlp.parse_v(property_vpq)[0]
    result_upper, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

    value = lower
    property_vpq = DRLP(property).append(f"{variable}={value}")
    property_pq = drlp.parse_v(property_vpq)[0]
    result_lower, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

    util.log(f"## Result:\nUpper@{upper}\t: {result_upper}\nLower @{lower}\t: {result_lower}\n", level=CONST.WARNING)
    while upper - lower > accuracy:
        value = (upper + lower) / 2
        property_vpq = DRLP(property).append(f"{variable}={value}")
        property_pq = drlp.parse_v(property_vpq)[0]
        result, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        if result == result_lower:
            lower = value
        elif result == result_upper:
            upper = value

        util.log_prompt(3)
        util.log("########## Step ##########\n", level=CONST.WARNING)
        util.log(f"## Result:\nMid @ {value}\t: {result}\n", level=CONST.WARNING)
    return value


def search_boundary_iteration_dichotomy(network: NN, property: DRLP, kwargs: dict, step: float = 0.3,
                                        accuracy: float = 1e-2, verifier: str = None, k_max: int = 10, k_min: int = 1) -> float:
    variable, bounds = list(kwargs.items())[0]
    lower = bounds[0]
    upper = bounds[1]
    property = property.obj

    util.log_prompt(3)
    util.log("########## Init ##########\n", level=CONST.WARNING)

    value = lower
    property_vpq = DRLP(property).append(f"{variable}={value}")
    property_pq = drlp.parse_v(property_vpq)[0]
    result, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

    while result and value <= upper:
        value = value * (1 + step)
        property_vpq = DRLP(property).append(f"{variable}={value}")
        property_pq = drlp.parse_v(property_vpq)[0]
        result, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        util.log_prompt(3)
        util.log("########## Step ##########\n", level=CONST.WARNING)
        util.log(f"## Result:\nUpper @ {value}\t: {result}\n", level=CONST.WARNING)

    if value > upper:
        return value

    value = search_boundary_dichotomy(network, property, {variable: [lower, value]}, accuracy, verifier, k_max, k_min)
    return value


def search_boundary_hypercubic(network: NN, property: DRLP, kwargs: dict, accuracy: Union[float, dict] = 1e-2, verifier: str = None, k_max: int = 10, k_min: int = 1):
    variables = list(kwargs.keys())

    values = {}
    for variable in list(kwargs.keys())[:-1]:
        values[variable] = kwargs[variable][0]

    results = []
    property = property.obj

    def search(values, depth):
        variable = variables[depth]
        if depth == len(variables) - 1:
            _property = DRLP(property).set_values(values)
            value = search_boundary_dichotomy(
                network, _property, {variable: kwargs[variable]}, accuracy, verifier, k_max, k_min)
            vals = values.copy()
            vals[variable] = value
            results.append(DRLP(property, vals).set_values(vals))
        else:
            while values[variable] <= kwargs[variable][1]:
                search(values.copy(), depth + 1)
                if isinstance(accuracy, dict):
                    values[variable] += accuracy[variable]
                else:
                    values[variable] += accuracy

    search(values, 0)

    return results


# kwargs: dict
# {var: {lb, ub, dv, prec}}


def search_breakpoints(network: NN, property: DRLP, kwargs: dict, default_precise: float = 1e-2, verifier: str = None,
                        k_max: int = 10, k_min: int = 1, to_induct: bool = True) -> List[Tuple[DRLP, Tuple[int, bool, numpy.ndarray]]]:
    breakpoints = []

    def next(lb: float, ub: float, curr: float, prec: float, method: str, lb_ans=None, ub_ans=None, curr_ans=None):

        _next = None
        if method == "linear":
            if curr < ub:
                _next = curr + prec
            else:
                _next = None
        elif method == "binary":
            if ub - lb > prec:
                if lb_ans[0] != curr_ans[0]:
                    ub = curr
                    _next = (lb + ub) / 2
                elif ub_ans[0] != curr_ans[0]:
                    lb = curr
                    _next = (lb + ub) / 2
        elif method == "iterative":
            if lb_ans == curr_ans:
                _next = curr * prec
            else:
                ub = curr

        return lb, ub, _next

    def call_verify(_prop: DRLP):
        util.log_prompt(3)
        util.log("*** Single DRL Query Verifying...", level=CONST.ERROR)
        util.log("Kwargs:", _prop.kwargs, level=CONST.ERROR)
        util.log("\n\n", level=CONST.ERROR)

        return verify(network, _prop, verifier, k_max, k_min, to_induct)

    def concrete(_propert: DRLP, var: str, value: float, all: bool = False):
        if all:
            left_kwargs = {k: v["default_value"]
                           for k, v in kwargs.items() if k not in _propert.kwargs.keys() and k != var}
            return DRLP(_propert.obj, _propert.kwargs).set_kwarg(var, value).set_kwargs(left_kwargs)
        return DRLP(_propert.obj, _propert.kwargs).set_kwarg(var, value)

    def search(_property: DRLP, _kwargs: dict):
        if len(_kwargs) == 1:
            util.log_prompt(4)
            util.log("########## Search ##########", level=CONST.ERROR)
            util.log("Env  kwargs: ", _property.kwargs, level=CONST.ERROR)
            util.log("Curr kwarg:  ", _kwargs, level=CONST.ERROR)
            util.log("Known breakpoints:  ", breakpoints, level=CONST.ERROR)

        _kwargs = deepcopy(_kwargs)
        var = list(_kwargs.keys())[0]
        var_v = _kwargs[var]
        _kwargs.pop(var)

        lb = var_v["lower_bound"]
        ub = var_v.get("upper_bound", 0)
        dv = var_v.get("default_value", 0)
        prec = var_v.get("precise", default_precise)
        method = var_v.get("method", "linear")
        skip = var_v.get("skip", False)

        curr = lb

        ans = None
        prev_ans = None
        lb_ans = None
        ub_ans = None

        if method == "binary":
            _prop = concrete(_property, var, lb, True)
            lb_ans = call_verify(_prop)
            init_lb_ans_set = (_prop, lb_ans)
            _prop = concrete(_property, var, ub, True)
            ub_ans = call_verify(_prop)
            init_ub_ans_set = (_prop, ub_ans)

        while True:
            _prop = concrete(_property, var, curr)

            if len(_kwargs) == 0:

                if method == "binary":
                    if curr == lb:
                        ans = lb_ans
                    elif curr == ub:
                        ans = ub_ans
                    else:
                        ans = call_verify(_prop)
                else:
                    ans = call_verify(_prop)

                if method == "linear":
                    if prev_ans is None or ans[0] != prev_ans[0]:
                        prev_ans = ans
                        breakpoints.append((_prop, ans))
                        if skip:
                            break

            else:
                search(_prop, _kwargs)

            lb, ub, curr = next(lb, ub, curr, prec, method, lb_ans, ub_ans, ans)

            if curr is None:
                if method == "linear":
                    if prev_ans is not None and breakpoints is not None and \
                            prev_ans != breakpoints[-1]:
                        breakpoints.append((_prop, prev_ans))
                elif method == "binary":
                    breakpoints.append(init_lb_ans_set)

                    if init_lb_ans_set[1][0] != init_ub_ans_set[1][0]:
                        ans = (not init_lb_ans_set[1][0], ans[1], ans[2])
                        breakpoints.append((_prop, ans))
                    breakpoints.append(init_ub_ans_set)
                break

    search(property, kwargs)
    return breakpoints
