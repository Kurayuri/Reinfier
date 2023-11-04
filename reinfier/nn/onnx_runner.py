from ..import CONSTANT
from ..import util
from .NN import NN
from typing import Union
import onnxruntime
import numpy as np
import onnx


# onnxruntime.set_default_logger_severity(0)


def run_onnx(network: Union[str, NN], input: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(input, str):
        input_array = np.load(input)
    elif isinstance(input, np.ndarray):
        input_array = input
    else:
        raise TypeError("Unable to run on type '{0}'".format(type(input)))
    input_array = input_array.astype(np.float32).reshape(1,-1)
    if isinstance(network, str):
        pass
    elif isinstance(network, NN):
        network = network.path
    else:
        raise TypeError("Unable to run on type '{0}'".format(type(network)))
    session = onnxruntime.InferenceSession(network,)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input_array})
    util.log("With Input:\n", input_array, "\nOutput:\n", output, level=CONSTANT.INFO)
    return output
