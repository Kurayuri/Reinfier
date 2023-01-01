from .import CONSTANT
LogLevel = CONSTANT.INFO
TmpPath = "tmp"
BranchableVerifiers = [CONSTANT.MARABOU]
ToTestAllVerifier = False


def set_LogLevel(level):
    global LogLevel
    LogLevel = level


def set_TmpPath(path):
    global TmpPath
    TmpPath = path


def set_ToTestAllVerifier(to_test_all_verifier):
    global ToTestAllVerifier
    ToTestAllVerifier = to_test_all_verifier
