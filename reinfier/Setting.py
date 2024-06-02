from .import CONST
LogLevel = CONST.INFO
TmpPath = "tmp"
ContainerTmpPath = "/tmp"
BranchableVerifiers = [CONST.MARABOU]
ToTestAllVerifier = False
ContainerNames = {CONST.DNNV: CONST.DNNV, CONST.VERISIG: CONST.VERISIG, CONST.MARABOU: CONST.MARABOU}


def set_LogLevel(level):
    global LogLevel
    prev_level = LogLevel
    LogLevel = level
    return prev_level


def set_TmpPath(path):
    global TmpPath
    TmpPath = path


def set_ContainerTmpPath(path):
    global ContainerTmpPath
    ContainerTmpPath = path


def set_ContainerName(framework_name, container_name):
    global ContainerNames
    ContainerNames[framework_name] = container_name


def set_ToTestAllVerifier(to_test_all_verifier):
    global ToTestAllVerifier
    ToTestAllVerifier = to_test_all_verifier
