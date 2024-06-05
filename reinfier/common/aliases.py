from typing import NamedTuple, Dict, IO
import numpy as np
import os
import onnx

PathLike = IO | str | os.PathLike
ONNXModel = onnx.onnx_ml_pb2.ModelProto
Array = np.ndarray


