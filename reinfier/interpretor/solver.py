from ..drlp.DRLP import DRLP
from ..nn.NN import NN
from ..import nn
from ..import dnnv
from ..import drlp
from .import lib
from typing import Tuple, List, Callable, Union
from scipy.spatial import distance


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
            vector = list(break_point[0].kwargs.values())
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


def answer_sensitivity_analysis(inline_break_points: List[List], original_output, dist: Union[str, Callable] = "euclidean") -> Tuple[List[float], float, Tuple[DRLP, Tuple]]:
    min_val = float('inf')
    min_obj = (None, None)

    for line in inline_break_points:
        for break_point in line:
            vector = list(break_point[0].kwargs.values())
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
            vector = list(break_point[0].kwargs.values())
            d = 0
            if isinstance(dist, Callable):
                d = dist(vector, [0] * len(vector))
            elif isinstance(dist, str):
                matrix = [vector, [0] * len(vector)]
                d = distance.pdist(matrix, dist)
            if d < min_val:
                min_val = d
                min_obj = vector, break_point

    return min_obj[0], min_val, min_obj[1]
