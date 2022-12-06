from . import CONSTANT
LogLevel = CONSTANT.INFO

def set_LogLevel(level):
    global LogLevel
    LogLevel=level
