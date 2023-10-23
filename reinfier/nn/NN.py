import onnx


class NN:
    def __init__(self, arg, filename="tmp.onnx"):
        self.path = None
        self.obj = None
        self.input_size = None
        self.output_size = None
        if isinstance(arg, str):
            self.path = arg
            self.obj = onnx.load(arg)
        elif isinstance(arg, onnx.onnx_ml_pb2.ModelProto):
            self.obj
            self.path = filename
        elif isinstance(arg, NN):
            self.path = arg.path
            self.obj = onnx.load(self.path)
        # elif arg is None:
        #     pass
        else:
            raise Exception("Invalid type to initialize NN object")

    def save(self, path: str = None):
        try:
            if path is None:
                path = self.path
            open(path, "w").write(self.obj)
        except BaseException:
            raise BaseException
    
    def size(self):
        if self.input_size is None:
            self.input_size =  int(self.obj.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
            self.output_size = int(self.obj.graph.output[0].type.tensor_type.shape.dim[1].dim_value)
        
        return self.input_size,self.output_size

    def to_torch(self):
        import onnx2torch
        return onnx2torch.convert(self.obj)

    def __str__(self):
        return self.path
