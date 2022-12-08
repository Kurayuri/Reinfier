from . import CONSTANT
LogLevel = CONSTANT.INFO
TmpPath = "tmp"
BranchableVerifier = [CONSTANT.MARABOU]


def set_LogLevel(level):
    global LogLevel
    LogLevel = level


def set_TmpPath(path):
    global TmpPath
    TmpPath = path
