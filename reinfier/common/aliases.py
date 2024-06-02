from typing import NamedTuple, Dict, IO
from .Feature import Feature

import os
import onnx

PathLike = IO | str | os.PathLike
ONNXModel = onnx.onnx_ml_pb2.ModelProto


class SearchConfig:
    BINARY = "binary"
    LINEAR = "linear"
    ITERATIVE = "iterative"

    def __init__(self, lower: float, upper: float, precise: float = 1e-2, method: str = "binary"):
        self.lower = lower
        self.upper = upper
        self.precise = precise
        self.method = method


class PropertyFeatures(NamedTuple):
    input: Dict[int, Feature]
    output: Dict[int, Feature]


class WhichFtr(NamedTuple):
    io: str
    index: int
