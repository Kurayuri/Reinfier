from .base import BaseObject

class DNNP(BaseObject):
    def __init__(self, arg, filename="tmp.dnnp"):

        if isinstance(arg, str):
            try:
                with open(arg) as f:
                    self.obj = f.read()
                self.path = arg
            except Exception:
                self.path = filename
                self.obj = arg
        elif isinstance(arg, DNNP):
            self.path = arg.path
            self.obj = arg.obj
        else:
            raise Exception("Invalid type to initialize DNNP object")

    def save_obj(self, path: str):
        with open(path, 'w') as file:
            file.write(self.obj)

    def __str__(self):
        return f"{self.obj}"
    
    def __repr__(self):
        return self.path
