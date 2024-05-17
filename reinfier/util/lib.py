from .. import CONST
from .. import Setting
import os
import time


def get_filename_from_path(path: str):
    return os.path.basename(path)


def get_savepath(filename, step: int, filename_extension: str):
    try:
        os.mkdir(Setting.TmpPath)
    except BaseException:
        pass
    if isinstance(filename, str):
        filename = filename.rsplit(".")[0]
        filename = "%s#%d_%s.%s"%(filename, step, str(time.time()).replace(".", ""), filename_extension)
        return os.path.join(Setting.TmpPath, filename)
    else:
        filename = [get_filename_from_path(x).rsplit(".")[0] for x in filename]
        filename= "%s.%s"%("@".join(filename),filename_extension)
        return os.path.join(Setting.TmpPath, filename)

def get_savepath_container(basename=None, extension=None):
    filename=""
    if basename:
        filename+=f'{basename}#0_{str(time.time()).replace(".", "")}'
    if extension:
        filename+=f".{extension}"
    return os.path.join(Setting.ContainerTmpPath,filename)


def log(*args, level=CONST.DEBUG, style=CONST.STYLE_RESET,end:str="\n"):
    if len(args) == 1 and (isinstance(args[0], tuple) or isinstance(args[0], list)):
        args = args[0]
    if level >= Setting.LogLevel:
        print(style+" ".join(map(str, args))+CONST.STYLE_RESET,end=end)


def log_prompt(prompt_level: int, text="",level=CONST.WARNING,style=CONST.STYLE_RESET):
    prompt_lens=[60,60,80,80,120]
    prompts=[".","*","#","-","="]


    if prompt_level == 1:
        log(("*" * prompt_lens[prompt_level]), level=CONST.INFO,style=style)
    elif prompt_level == 2:
        log(("\n\n" + "#" * prompt_lens[prompt_level] + "\n" + "#" * prompt_lens[prompt_level]), level=level,style=style)
    elif prompt_level == 3:
        log(("\n" + ("-" * prompt_lens[prompt_level] + "\n") * 3), level=level,style=style,end="")
    elif prompt_level == 4:
        log("\n\n\n\n"+((prompts[prompt_level] * prompt_lens[prompt_level] + "\n") * 4), level=level,style=style,end="")

    if text:
        textlen=len(text)
        bounds = prompt_lens[prompt_level]-2-textlen
        left = bounds//2
        right = bounds-left
        log(f"{prompts[prompt_level] * left} {text} {prompts[prompt_level] * right}",level=level,style=style)


def confirm_input(text,itype):
    if itype == CONST.INTERACTIVE_ITYPE_y_or_N:
        log(text,level=CONST.CRITICAL)
        log(f"Proceed ({CONST.INTERACTIVE_ITYPE_y_or_N})?",level=CONST.CRITICAL,end=" ")
        response=input().strip().lower()
        if response == "y" or response == "yes":
            return True
        else:
            return False


