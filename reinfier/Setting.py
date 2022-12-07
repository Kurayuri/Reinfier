from . import CONSTANT
LogLevel = CONSTANT.INFO
TmpPath = "tmp"

def set_LogLevel(level):
    global LogLevel
    LogLevel=level

def set_TmpPath(path):
    global TmpPath
    TmpPath=path