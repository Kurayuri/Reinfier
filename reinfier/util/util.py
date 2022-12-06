from .. import CONSTANT
from .. import Setting

def get_filename_from_path(path:str):
    path=path.rsplit("/",1)
    if len(path)==1:
        return path[0]
    else:
        return path[1]

def log(*args,level=CONSTANT.DEBUG):
    if level>=Setting.LogLevel:
        print(" ".join(map(str,args)))