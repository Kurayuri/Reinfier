from .import CONSTANT
LogLevel = CONSTANT.INFO
TmpPath = "tmp"
ContainerTmpPath = "/tmp"
BranchableVerifiers = [CONSTANT.MARABOU]
ToTestAllVerifier = False
ContainerNames = {CONSTANT.DNNV:CONSTANT.DNNV,CONSTANT.VERISIG:CONSTANT.VERISIG}

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
