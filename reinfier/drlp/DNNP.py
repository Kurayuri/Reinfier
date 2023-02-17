class DNNP:
    def __init__(self, arg, filename="tmp.dnnp"):
        self.path = None
        self.obj = None

        if isinstance(arg, str):
            try:
                with open(arg) as f:
                    self.obj = f.read()
                self.path = arg
            except Exception:
                self.path = filename
                self.obj = arg

        else:
            raise Exception("Invalid type to initialize DNNP object")

    def save(self, path: str = None):
        try:
            if path is None:
                path = self.path
            open(path, "w").write(self.obj)
        except BaseException:
            raise BaseException

    def __str__(self):
        return f"{self.path}\n{self.obj}"
    
    def __repr__(self):
        return self.path
