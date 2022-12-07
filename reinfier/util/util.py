from .. import CONSTANT
from .. import Setting
import os

def get_filename_from_path(path:str):
    path=path.rsplit("/",1)
    if len(path)==1:
        return path[0]
    else:
        return path[1]

def get_savepath(filename,step:int,type:str):
    try:
        os.mkdir(Setting.TmpPath)
    except:
        pass
    # filename=get_filename_from_path(filename)
    if isinstance(filename,str):
        filename = filename.rsplit(".")[0]
        return "%s/%s#%d.%s"% (Setting.TmpPath,filename,step,type)
    else:
        filename = [get_filename_from_path(x).rsplit(".")[0] for x in filename]
        return "%s/%s.%s"%(Setting.TmpPath,"@".join(filename),type)

def log(*args,level=CONSTANT.DEBUG):
    if level>=Setting.LogLevel:
        print(" ".join(map(str,args)))