from . import CONSTANT
LogLevel = CONSTANT.INFO
TmpPath = "tmp"
BranchableVerifier = [CONSTANT.MARABOU]
SelectionFull = False


def set_LogLevel(level):
    global LogLevel
    LogLevel = level


def set_TmpPath(path):
    global TmpPath
    TmpPath = path


def set_SelectionFull(full_test):
    global SelectionFull
    SelectionFull = full_test
