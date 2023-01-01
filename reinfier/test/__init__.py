"""Reinfier.Test Sample"""
from ..nn.NN import NN
from ..drlp.DRLP import DRLP


def get_sample_path():
    path = str(__file__)
    path = path.rsplit("/", 1)[0] + "/" + "test01/"
    network = path + "nn/test01.onnx"
    property = path + "drlp/test01_p1.drlp"
    return NN(network), DRLP(property)
