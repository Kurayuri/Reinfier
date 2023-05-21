from .DRLPTransformer import DRLPTransformer
from copy import deepcopy


class DRLP:
    def __init__(self, arg, kwargs: dict = {}, filename="tmp.drlp"):
        self.path = None
        self.obj = None
        self.kwargs = deepcopy(kwargs)

        if isinstance(arg, str):
            try:
                with open(arg) as f:
                    self.obj = f.read()
                self.path = arg
            except Exception:
                self.path = filename
                self.obj = arg
        elif isinstance(arg, DRLP):
            self.path = arg.path
            self.obj = arg.obj
        else:
            raise Exception("Invalid type to initialize DRLP object")

    def save(self, path: str = None):
        try:
            if path is None:
                path = self.path
            open(path, "w").write(self.obj)
        except BaseException:
            raise BaseException

    def edit(self, code: str, to_overwrite: bool = False) -> str:
        if to_overwrite:
            return self.overwrite(code)
        else:
            return self.append(code)

    def append(self, code: str) -> str:
        from . import lib

        drlp_v, drlp_pq = lib.split_drlp_vpq(self.obj)
        drlp_v += code
        self.obj = "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))
        return self

    def overwrite(self, code: str) -> str:
        from . import lib

        drlp_v, drlp_pq = lib.split_drlp_vpq(self.obj)
        drlp_v = code
        self.obj = "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))
        return self

    def set_kwarg(self, variable: str, value):
        self = self.append(f"{variable}={value}")
        self.kwargs[variable] = value
        return self

    def set_kwargs(self, kwargs: dict):
        code = ""
        for k, v in kwargs.items():
            code += f"{k}={v}\n"
        self = self.append(code)
        self.kwargs = {**self.kwargs, **kwargs}
        return self

    def __str__(self):
        return f"{self.obj}"

    def __repr__(self):
        return f"{self.path}#{self.kwargs}"
