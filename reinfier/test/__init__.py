"""Reinfier.Test Sample"""
import os

def get_sample_path():
    path=str(__file__)
    path=path.rsplit("/",1)[0]+"/"+"test01/"
    dnn=path+"nn/test01.onnx"
    drlp=path+"drlp/test01_p1.drlp"
    return dnn,drlp