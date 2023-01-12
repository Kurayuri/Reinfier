from .DRLPTransformer import DRLPTransformer

class DRLP:
    def __init__(self, arg, kwargs=None):
        self.path = None
        self.obj = None
        self.kwargs = kwargs

        if isinstance(arg, str):
            try:
                with open(arg) as f:
                    self.obj = f.read()
                self.path = arg
            except Exception:
                self.path = "tmp.drlp"
                self.obj = arg

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

    def __str__(self):
        return self.path

    def __repr__(self):
        return self.path + str(self.kwargs)
