import re


def get_filename_from_path(path:str):
    path=path.rsplit("/",1)
    if len(path)==1:
        return path[0]
    else:
        return path[1]