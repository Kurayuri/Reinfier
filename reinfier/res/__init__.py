"""Reinfier.Test Sample"""
from ..common.NN import NN
from ..common.DRLP import DRLP
import os


def get_example():
    path = str(__file__)
    path = os.path.join(os.path.dirname(path), "test01")
    network = os.path.join(path, "nn", "test01.onnx")
    property = os.path.join(path, "drlp", "test01_p1.drlp")
    return NN(network), DRLP(property)
