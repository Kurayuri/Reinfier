from .  import onnx_runner
from .. import utils
import numpy as np
import onnx
import copy
import onnx.numpy_helper

def convert_name(name: str, step: int):
    return name+"@"+str(step)

def diag_copy(matrix:np.ndarray,k:int):
    if matrix.ndim==1:
        matrix_new=np.zeros((matrix.shape[0]*k),matrix.dtype)
        for i in range(k):
            row=i*matrix.shape[0]
            for j in range(matrix.shape[0]):
                matrix_new[row+j]=matrix[j]
    elif matrix.ndim ==2:
        matrix_new=np.zeros((matrix.shape[0]*k,matrix.shape[1]*k),matrix.dtype)
        for i in range(k):
            row=i*matrix.shape[0]
            col=i*matrix.shape[1]
            for j in range(matrix.shape[0]):
                for l in range(matrix.shape[1]):
                    matrix_new[row+j][col+l]=matrix[j][l]
    else:
        raise Exception
    return matrix_new




def unwind_network(network, k: int):
    if isinstance(network,str):
        model = onnx.load(network)
    else:
        model=network
        network="tmp.onnx"
    origin_filename=network
    network=utils.utils.get_filename_from_path(network)
        
    # Check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')
    graph = model.graph
    graph_node = graph.node
    graph_input = graph.input[0]
    graph_input_length = int(
        graph_input.type.tensor_type.shape.dim[1].dim_value)
    graph_output = graph.output[0]
    graph_output_length = int(
        graph_output.type.tensor_type.shape.dim[1].dim_value)

    if k > 1:
        graph_input.name = "Input"
        graph_output.name = "Output"
        initializer = graph.initializer
        initializer_dict= {x.name:(idx,onnx.numpy_helper.to_array(x)) for idx,x in enumerate(initializer)}

        graph_input.type.tensor_type.shape.dim[0].dim_value = 1
        graph_input.type.tensor_type.shape.dim[1].dim_value = graph_input_length*k
        graph_output.type.tensor_type.shape.dim[0].dim_value = 1
        graph_output.type.tensor_type.shape.dim[1].dim_value = graph_output_length*k

        
        for node in graph_node:
            for name in node.input:
                if name in initializer_dict.keys():
                    input_name=name
                    idx,matrix=initializer_dict[input_name]
                    matrix_new=diag_copy(matrix,k)
                    tensor=onnx.numpy_helper.from_array(matrix_new,input_name)
                    model.graph.initializer[idx].CopyFrom(tensor)

    model.opset_import[0].version = 9
    print(onnx.helper.printable_graph(model.graph))
    # print(graph.input)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    filename = network.rsplit(".")
    unwinded_network_filename = filename[0]+"_step_%d" % (k)+".onnx"
    print(unwinded_network_filename)
    onnx.save(model, unwinded_network_filename)

    onnx_runner.run_onnx(origin_filename, np.array(
        [[1.0]*graph_input_length], dtype=np.float32))
    onnx_runner.run_onnx(unwinded_network_filename, np.array(
        [[1.0]*graph_input_length*k], dtype=np.float32))

    return unwinded_network_filename

if __name__ == "__main__":
    unwind_network("test01.onnx",3)
