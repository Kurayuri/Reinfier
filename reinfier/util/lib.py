from .. import CONSTANT
from .. import Setting
import os
import time


def get_filename_from_path(path: str):
    path = path.rsplit("/", 1)
    if len(path) == 1:
        return path[0]
    else:
        return path[1]


def get_savepath(filename, step: int, type: str):
    try:
        os.mkdir(Setting.TmpPath)
    except BaseException:
        pass
    # filename=get_filename_from_path(filename)
    if isinstance(filename, str):
        filename = filename.rsplit(".")[0]
        return "%s/%s#%d_%s.%s" % (Setting.TmpPath, filename, step, str(time.time()).replace(".", ""), type)
        # return "%s/%s#%d.%s" % (Setting.TmpPath, filename, step,type)
    else:
        filename = [get_filename_from_path(x).rsplit(".")[0] for x in filename]
        return "%s/%s.%s" % (Setting.TmpPath, "@".join(filename), type)


def log(*args, level=CONSTANT.DEBUG):
    if len(args) == 1 and (isinstance(args[0], tuple) or isinstance(args[0], list)):
        args = args[0]
    if level >= Setting.LogLevel:
        print(" ".join(map(str, args)))
    
def log_prompt(prompt_level,level=CONSTANT.WARNING):
    if prompt_level==1:
        log(("*" * 80), level=CONSTANT.INFO)
    elif prompt_level==2:
        log(("\n\n" + "#" * 80 + "\n" + "#" * 80), level=level)
    elif prompt_level==3:
        log(("\n" + ("-" * 120 + "\n")*3), level=level)


