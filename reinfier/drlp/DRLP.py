class DRLP:
    def __init__(self, arg):
        self.path = None
        self.obj = None

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

    def __str__(self):
        return self.path
