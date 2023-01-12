import onnx


class NN:
    def __init__(self, arg):
        self.path = None
        self.obj = None
        if isinstance(arg, str):
            self.path = arg
            self.obj = onnx.load(arg)
        elif isinstance(arg, onnx.onnx_ml_pb2.ModelProto):
            self.obj
            self.path = "tmp.onnx"
        elif isinstance(arg, NN):
            self.path = arg.path
            self.obj = onnx.load(self.path)
        else:
            raise Exception("Invalid type to initialize NN object")

    def save(self, path: str = None):
        try:
            if path is None:
                path = self.path
            open(path, "w").write(self.obj)
        except BaseException:
            raise BaseException

    def __str__(self):
        return self.path
