from ..common.classes import VerificationAnswer, Breakpoint, BatchConfig
from ..common.DRLP import DRLP
from ..common.NN import NN
from ..import util
from ..import drlp
from ..import CONST
from ..import Setting
from .single import *
from .lib import *
from typing import Tuple, List, Union, Dict, Any
from copy import deepcopy
import numpy as np

# class Variable:
#     def __init__(self, id, values, type):
#         self.id = id
#         self.values = values
#         self.type = type
#         self.state = None
#         self.exchange = 0

#         if self.type == 'l':
#             self.state = True
#         elif self.type == 'b':
#             self.state = False
#         elif self.type == 'm':
#             self.state = False
#         elif self.type == 'e':
#             self.state = True
#         elif self.type == 'r':
#             self.state = True

#     def signal(self, sign):
#         if self.state != sign:
#             self.exchange += 1
#         self.state = sign

#         if self.type == "l":
#             return self.exchange == 0
#         elif self.type == "b":
#             return self.exchange == 0
#         elif self.type == "m":
#             return self.exchange < 2
#         elif self.type == "e":
#             return self.exchange < 2
#         elif self.type == "r":
#             return True

#         return None


def verify_linear(network: NN, property: DRLP, verifier: str = None, k_max: int = 10, k_min: int = 1) -> List[Tuple[DRLP, int, bool]]:
    property_pqs = drlp.parse_v(property)

    ans = []
    for property_pq in property_pqs:
        result, k, __ = verify(network, property_pq, verifier=verifier, k_max=k_max, k_min=k_min)

        ans.append((property_pq, k, result))

        util.log((property_pq, property_pq.variables, k, result), level=CONST.INFO)
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

        util.log((property_pq, property_pq.variables, k, result), level=CONST.INFO)
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


# def search_breakpoints(network: NN, property: DRLP, kwargs: Dict[str, BatchConfig], verifier: str = None,
#                        k_max: int = 10, k_min: int = 1, to_induct: bool = True) -> List[Breakpoint]:
#     breakpoints = []

#     def step(lb: float, ub: float, curr: float, prec: float, step_method: str, lb_ans=None, ub_ans=None, curr_ans=None):
#         next = None
#         if step_method == BatchConfig.LINEAR:
#             next = curr + prec if curr < ub else None

#         elif step_method == BatchConfig.BINARY:
#             if ub - lb > prec:
#                 if lb_ans[0] != curr_ans[0]:
#                     ub = curr
#                     next = (lb + ub) / 2
#                 elif ub_ans[0] != curr_ans[0]:
#                     lb = curr
#                     next = (lb + ub) / 2

#         elif step_method == BatchConfig.ITERATIVE:
#             if lb_ans == curr_ans:
#                 next = curr * prec
#             else:
#                 ub = curr

#         next = util.data.round_to_precision(next, prec) if next is not None else None

#         return lb, ub, next

#     def call_verify(property: DRLP):
#         util.log_prompt(3)
#         util.log("*** Single DRL Query Verifying...", level=CONST.ERROR)
#         util.log("Kwargs:", property.variables, level=CONST.ERROR)
#         util.log("\n\n", level=CONST.ERROR)

#         return verify(network, property, verifier, k_max, k_min, to_induct)

#     def concrete(property: DRLP, var_name: str, var_value: float, all: bool = False):
#         left_kwargs = {k: v.default
#                        for k, v in kwargs.items()
#                        if k not in property.variables.keys() and k != var_name} if all else {}
#         return DRLP(property.obj, property.variables).set_variable(var_name, var_value).set_variables(left_kwargs)

#     def search(property: DRLP, kwargs:  Dict[str, BatchConfig]):
#         if len(kwargs) == 1:
#             util.log_prompt(4)
#             util.log("########## Search ##########", level=CONST.ERROR)
#             util.log("Env  kwargs: ", property.variables, level=CONST.ERROR)
#             util.log("Curr kwarg:  ", kwargs, level=CONST.ERROR)
#             util.log("Known breakpoints:  ", breakpoints, level=CONST.ERROR)

#         kwargs = deepcopy(kwargs)
#         var_config: BatchConfig
#         var_name = next(iter(kwargs))
#         var_config = kwargs[var_name]
#         kwargs.pop(var_name)

#         lb = var_config.lower
#         ub = var_config.upper

#         curr = lb

#         curr_ans = None
#         prev_ans = None
#         lb_ans = None
#         ub_ans = None

#         if var_config.step_method == BatchConfig.LINEAR:
#             _prop = concrete(property, var_name, lb, True)
#             lb_ans = call_verify(_prop)
#             init_lb_bp = Breakpoint(_prop, lb_ans)

#             _prop = concrete(property, var_name, ub, True)
#             ub_ans = call_verify(_prop)
#             init_ub_bp = Breakpoint(_prop, ub_ans)

#         while True:
#             _prop = concrete(property, var_name, curr)

#             if len(kwargs) == 0:
#                 if var_config.step_method == BatchConfig.BINARY:
#                     if curr == lb:
#                         curr_ans = lb_ans
#                     elif curr == ub:
#                         curr_ans = ub_ans
#                     else:
#                         curr_ans = call_verify(_prop)
#                 else:
#                     curr_ans = call_verify(_prop)

#                 if var_config.step_method == BatchConfig.LINEAR:
#                     if prev_ans is None or curr_ans[0] != prev_ans[0]:
#                         prev_ans = curr_ans
#                         breakpoints.append(Breakpoint(_prop, curr_ans))
#                         if var_config.skip:
#                             break

#             else:
#                 search(_prop, kwargs)

#             lb, ub, curr = step(lb, ub, curr, var_config.precise, var_config.step_method, lb_ans, ub_ans, curr_ans)

#             if curr is None:
#                 if var_config.step_method == BatchConfig.LINEAR:
#                     if prev_ans is not None and breakpoints is not None and prev_ans != breakpoints[-1]:
#                         breakpoints.append(Breakpoint(_prop, prev_ans))

#                 elif var_config.step_method == BatchConfig.BINARY:
#                     breakpoints.append(init_lb_bp)

#                     if init_lb_bp.ans.result != init_ub_bp.ans.result:
#                         curr_ans.result = not init_lb_bp.ans.result
#                         breakpoints.append(Breakpoint(_prop, curr_ans))

#                     breakpoints.append(init_ub_bp)
#                 break

#     search(property, kwargs)
#     return breakpoints


def search_breakpoints(network: NN, property: DRLP,
                       var_configs: Dict[str, BatchConfig],
                       grouped: bool = False,
                       verify_kwargs: Dict[str, Any] = {}) -> List[Breakpoint]:

    def step(lb: float, ub: float, curr: float, prec: float, step_method: str,
             lb_ans: VerificationAnswer = None, ub_ans: VerificationAnswer = None, curr_ans: VerificationAnswer = None):
        next = None
        if step_method == BatchConfig.LINEAR:
            next = curr + prec if curr < ub else None

        elif step_method == BatchConfig.BINARY:
            if not util.data.equal(ub, lb, prec):
                if lb_ans.result != curr_ans.result:
                    ub = curr
                    next = (lb + ub) / 2
                elif ub_ans.result != curr_ans.result:
                    lb = curr
                    next = (lb + ub) / 2

        elif step_method == BatchConfig.ITERATIVE:
            if lb_ans == curr_ans:
                next = curr * prec
            else:
                ub = curr

        next = util.data.round_to_precision(next, prec/10) if next is not None else None

        return lb, ub, next

    def call_verify(property: DRLP):
        util.log_prompt(3)
        util.log("*** Single DRL Query Verifying...", level=CONST.ERROR)
        util.log("Kwargs:", property.variables, level=CONST.ERROR)
        util.log("\n\n", level=CONST.ERROR)
        return VerificationAnswer(np.random.rand() > 0.1, 1, None)
        # return verify(network, property, **verify_kwargs)

    def concrete(property: DRLP, var_name: str, var_value: float, all: bool = False):
        left_kwargs = {k: v.default
                       for k, v in var_configs.items()
                       if k not in property.variables.keys() and k != var_name} if all else {}
        return DRLP(property.obj, property.variables).set_variable(var_name, var_value).set_variables(left_kwargs)

    def search(property: DRLP, var_configs:  Dict[str, BatchConfig]):
        var_configs = deepcopy(var_configs)
        var_config: BatchConfig
        var_name = next(iter(var_configs))
        var_config = var_configs[var_name]
        var_configs.pop(var_name)

        if len(var_configs) == 0:
            util.log_prompt(4)
            util.log("########## Search ##########", level=CONST.ERROR)
            util.log("Env  kwargs: ", property.variables, level=CONST.ERROR)
            util.log("Curr kwarg:  ", var_configs, level=CONST.ERROR)
            # util.log("Known breakpoints:  ", breakpoints, level=CONST.ERROR)

        lb = var_config.lower
        ub = var_config.upper

        curr = lb

        curr_ans = None
        prev_ans = None
        lb_ans = None
        ub_ans = None

        if var_config.step_method == BatchConfig.BINARY:
            _prop = concrete(property, var_name, lb, True)
            lb_ans = call_verify(_prop)
            init_lb_bp = Breakpoint(_prop, lb_ans, lb_ans)

            if not util.data.equal(ub, lb, var_config.precise):

                _prop = concrete(property, var_name, ub, True)
                ub_ans = call_verify(_prop)
                init_ub_bp = Breakpoint(_prop, ub_ans, ub_ans)

        breakpoints = []

        while True:
            _prop = concrete(property, var_name, curr)

            if len(var_configs) == 0:
                if var_config.step_method == BatchConfig.BINARY:
                    if curr == lb:
                        curr_ans = lb_ans
                    elif curr == ub:
                        curr_ans = ub_ans
                    else:
                        curr_ans = call_verify(_prop)
                else:
                    curr_ans = call_verify(_prop)

                if var_config.step_method == BatchConfig.LINEAR:
                    if prev_ans is None:
                        if grouped:
                            breakpoints.append(Breakpoint(_prop, curr_ans, curr_ans))

                    elif curr_ans.result != prev_ans.result:
                        breakpoints.append(Breakpoint(_prop, curr_ans, prev_ans))
                        prev_ans = curr_ans
                        if var_config.skip:
                            break
                    prev_ans = curr_ans

            else:
                breakpoints.append(search(_prop, var_configs))

            lb, ub, curr = step(lb, ub, curr, var_config.precise, var_config.step_method, lb_ans, ub_ans, curr_ans)

            if curr is None:
                if len(var_configs) == 0:
                    match var_config.step_method:
                        case BatchConfig.BINARY:
                            if grouped:
                                breakpoints.append(init_lb_bp)
                            if init_lb_bp.ans.result != init_ub_bp.ans.result:
                                btype = Breakpoint.calc_btype(init_lb_bp.ans, init_ub_bp.ans)
                                breakpoints.append(Breakpoint(_prop, curr_ans, btype))
                            if grouped:
                                breakpoints.append(init_ub_bp)
                        case BatchConfig.LINEAR:
                            # Add Last if not added
                            if grouped:
                                if breakpoints and breakpoints[-1].ans is not curr_ans:
                                    breakpoints.append(Breakpoint(_prop, curr_ans, prev_ans))

                break
        return breakpoints
        # if len(var_configs) == 0:
        #     breakpoints.append(_breakpoints)

    ans = search(property, var_configs)

    return ans if grouped else util.data.flatten_list(ans)
