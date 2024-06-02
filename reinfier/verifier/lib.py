from .. import drlp

import os
import inspect


def log_call(*args):
    with open("log.txt", 'a+') as f:
        args = []
        for arg in args:
            arg = str(arg)
            if "\n" in arg:
                arg = "\n" + arg
            args.append(arg)
        f.write(" ".join(args) + "\n")


def get_coordinate(dims, id):
    coordinate = []
    for dim in reversed(dims):
        coordinate.append(id % dim)
        id //= dim
    return list(reversed(coordinate))


def get_id(dims, coordinate):
    id = 0
    base = 1
    for i in range(len(dims) - 1, -1, -1):
        id += (base * coordinate[i])
        base *= dims[i]
    return id


def get_dims(variables):
    dims = []
    for k, v in variables.items():
        if drlp.parser.is_iterable_variable(k):
            dims.append(len(v))
    return dims


def continue_verify(result):  # TODO: Check from Variable type
    return result == True
