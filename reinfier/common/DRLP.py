from __future__ import annotations
import copy
from typing import Dict, Any
from ..import Protocal
from .base import BaseObject


class DRLP(BaseObject):
    def __init__(self, arg, variables: dict = {}, filename="tmp.drlp"):
        super().__init__(arg, filename)

        self.variables = copy.deepcopy(variables)

        if isinstance(arg, str):
            try:
                with open(arg) as f:
                    self.obj = f.read()
                self.path = arg
            except Exception:
                self.path = filename
                self.obj = arg

            # if DRLPTransformer.PRECONDITION_DELIMITER not in self.obj:
            #     raise Exception('Invalid type to initialize DRLP object, DRLP cannot be splitted by EXPECTATION_DELIMITER "@Exp"')
        elif isinstance(arg, DRLP):
            self.path = arg.path
            self.obj = arg.obj
        else:
            raise Exception("Invalid type to initialize DRLP object")

    def save_obj(self, path: str):
        with open(path, 'w') as file:
            file.write(self.obj)

    def edit(self, code: str, to_overwrite: bool = False) -> str:
        if to_overwrite:
            return self.overwrite(code)
        else:
            return self.append(code)

    def append(self, code: str) -> str:
        from ..drlp import lib

        drlp_v, drlp_pq = lib.split_drlp_vpq(self.obj)
        drlp_v += code
        self.obj = "\n".join((drlp_v, Protocal.DRLP.Delimiter.Precondition, drlp_pq))
        return self

    def overwrite(self, code: str) -> str:
        from ..drlp import lib

        drlp_v, drlp_pq = lib.split_drlp_vpq(self.obj)
        drlp_v = code
        self.obj = "\n".join((drlp_v,  Protocal.DRLP.Delimiter.Precondition, drlp_pq))
        return self

    def set_variable(self, name: str, value) -> DRLP:
        self = self.append(f"{name}={value}")
        self.variables[name] = value
        return self

    def set_variables(self, variables: Dict[str, Any]) -> DRLP:
        code = ""
        for k, v in variables.items():
            code += f"{k}={v}\n"
        self = self.append(code)
        self.variables = {**self.variables, **variables}
        return self

    def set_value(self, variable: str, value) -> DRLP:
        self = self.append(f"{variable}={value}")
        return self

    def set_values(self, kwargs: dict) -> DRLP:
        code = ""
        for k, v in kwargs.items():
            code += f"{k}={v}\n"
        self = self.append(code)
        return self

    def __str__(self):
        return f"{self.obj}"

    def __repr__(self):
        return f"{self.path}#{self.variables}"
