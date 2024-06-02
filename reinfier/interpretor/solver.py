from ..common.DRLP import DRLP
from ..common.NN import NN
from ..common.aliases import *
from ..import nn
from ..import drlp
from ..import util
from typing import Tuple, List, Callable, Union, Dict
from scipy.spatial import distance
import torch as th


def answer_intuitiveness_examination(inline_break_points: List[List]) -> Tuple[bool, List[bool]]:
    ans = []
    total_ans = True
    for line in inline_break_points:
        if len(line) <= 1:
            ans.append(True)
        else:
            ans.append(False)
            total_ans = False
    return total_ans, ans


def answer_counterfactual_explanation(inline_break_points: List[List], dist: Union[str, Callable] = "euclidean") -> Tuple[List[float], float, Tuple[DRLP, Tuple]]:
    min_val = float('inf')
    min_obj = (None, None)
    for line in inline_break_points:
        for break_point in line:
            vector = list(break_point[2].values())
            d = 0
            if isinstance(dist, Callable):
                d = dist(vector, [0] * len(vector))
            elif isinstance(dist, str):
                matrix = [vector, [0] * len(vector)]
                d = distance.pdist(matrix, dist)
            if d < min_val:
                min_val = d
                min_obj = (vector, break_point)

    return min_obj[0], min_val, min_obj[1]


def answer_sensitivity_analysis(inline_break_points: List[List], original_output=None, dist: Union[str, Callable] = "euclidean") -> Tuple[List[float], float, Tuple[DRLP, Tuple]]:
    min_val = float('inf')
    min_obj = (None, None)

    for line in inline_break_points:
        for break_point in line:
            vector = list(break_point[2].values())
            if original_output is None:
                original_output = [0 for i in vector]
            d = 0
            if isinstance(dist, Callable):
                d = dist(vector, original_output)
            elif isinstance(dist, str):
                matrix = [vector, original_output]
                d = distance.pdist(matrix, dist)
            if d < min_val:
                min_val = d
                min_obj = (vector, break_point)

    return min_obj[0], min_val, min_obj[1]


def answer_importance_analysis(inline_break_points: List[List], dist: Union[str, Callable] = "euclidean") -> Tuple[List[float], float, Tuple[DRLP, Tuple]]:
    min_val = float('inf')
    min_obj = (None, None)

    for line in inline_break_points:
        for break_point in line:
            vector = list(break_point[2].values())
            d = 0
            if isinstance(dist, Callable):
                d = dist(vector, [0] * len(vector))
            elif isinstance(dist, str):
                matrix = [vector, [0] * len(vector)]
                d = distance.pdist(matrix, dist)
            if d < min_val:
                min_val = d
                min_obj = (vector, break_point)

    return min_obj[0], min_val, min_obj[1]


def measure_grads(network: NN, input, index, lower, upper, precision):
    model = network.to_torch()
    model.eval()

    grads = []

    if lower < upper:
        vals = util.prange(lower, upper, precision)
    else:
        vals = [lower]
    input = th.tensor(input, dtype=th.float32)
    for val in vals:
        _input_ = input.clone().detach()
        _input_[index] = val
        _input_.requires_grad = True

        output = model(_input_)

        model.zero_grad()
        output.backward()
        grads.append(_input_.grad[index].item())
    integral = sum(grads) * precision if len(grads) > 1 else grads[0]
    return grads, integral


def measure_sensitivity(network: NN, input, index, lower, upper, precision):
    grads, integral = measure_grads(network, input, index, lower, upper, precision)
    curr_sum = 0.0
    max_sum = 0.0
    max_idx = 0
    min_sum = 0.0
    min_idx = 0

    for idx in range(len(grads)):
        curr_sum += grads[idx]
        max_sum, max_idx = (curr_sum, idx) if curr_sum > max_sum else (max_sum, max_idx)
        min_sum, min_idx = (curr_sum, idx) if curr_sum < min_sum else (min_sum, min_idx)
    max_val = lower + precision*max_idx
    min_val = lower + precision*min_idx
    max_sum, min_sum = (max_sum*precision, min_sum*precision) if len(grads) > 1 else (max_sum, min_sum)
    return (max_sum, max_val), (min_sum, min_val)
