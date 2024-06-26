import onnx
import onnxruntime
import numpy as np
from typing import Any
from .base import BaseObject
from .aliases import ONNXModel, PathLike


class NN(BaseObject):

    def __init__(self, arg, filename="tmp.onnx"):
        super().__init__(arg, filename)
        self.input_size = None
        self.output_size = None
        if isinstance(arg, str):
            self.path = arg
            self.obj = onnx.load(arg)
        elif isinstance(arg, ONNXModel):
            self.obj
            self.path = filename
        elif isinstance(arg, NN):
            self.path = arg.path
            self.obj = onnx.load(self.path)
        else:
            raise Exception("Invalid type to initialize NN object")

    def save_obj(self, path: str):
        onnx.save(self.obj, path)

    def size(self):
        if self.input_size is None:
            self.input_size = int(self.obj.graph.input[0].type.tensor_type.
                                  shape.dim[1].dim_value)
            self.output_size = int(self.obj.graph.output[0].type.tensor_type.
                                   shape.dim[1].dim_value)

        return self.input_size, self.output_size

    def run(self, input: PathLike | np.ndarray | Any) -> np.ndarray:
        input_value = np.load(input) if isinstance(input, PathLike) else input
        input_value = np.array(input_value).astype(np.float32).reshape(1, -1)

        session = onnxruntime.InferenceSession(self.obj.SerializeToString())
        input_name = session.get_inputs()[0].name
        output = session.run([], {input_name: input_value})[0]
        return output

    def to_torch(self):
        import onnx2torch
        return onnx2torch.convert(self.obj)

    def to_yaml(self):
        import yaml
        weights = self.obj.graph.initializer

        nn_dict = {'weights': {}, 'offsets': {}, 'activations': {}}
        activation_functions = set([
            'Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Softmax',
            'Elu', 'Selu', 'Softsign', 'Softplus', 'Prelu'
        ])
        activations = []
        for node in self.obj.graph.node:
            if node.op_type in activation_functions:
                activations.append(str(node.op_type))
        for layer_idx in range(len(weights) // 2):
            nn_dict['weights'][layer_idx + 1] = []
            for row in onnx.numpy_helper.to_array(weights[layer_idx * 2]):
                nn_dict['weights'][layer_idx + 1].append(list(map(float, row)))

            nn_dict['offsets'][layer_idx + 1] = onnx.numpy_helper.to_array(weights[layer_idx * 2 + 1]).tolist()

            activation = activations.pop(0) if activations else "Linear"
            nn_dict['activations'][layer_idx + 1] = activation

        return yaml.dump(nn_dict)

    def __str__(self):
        return self.path
