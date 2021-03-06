from .  import onnx_runner
from .. import utils
import numpy as np
import onnx
import copy


def convert_name(name: str, step: int):
    return name+"@"+str(step)


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
    graph_input_origin = graph_input.name
    graph_input_length = int(
        graph_input.type.tensor_type.shape.dim[1].dim_value)
    graph_output = graph.output[0]
    graph_output_origin = graph_output.name
    graph_output_length = int(
        graph_output.type.tensor_type.shape.dim[1].dim_value)

    if k > 1:
        graph_input.name = "Input"
        graph_output.name = "Output"
        initializer = graph.initializer
        initializer_name = [x.name for x in initializer]
        # print(onnx.helper.printable_graph(model.graph))

        # input_name=graph.input[0]
        # input=onnx.helper.make_tensor_value_info(graph_input.name,
        #     elem_type=graph_input.type.tensor_type.elem_type,
        #     shape=[30])
        # model.graph.input.append(input)

        node_origin = copy.deepcopy(graph_node)
        step = 0

        graph_input.type.tensor_type.shape.dim[0].dim_value = 1
        graph_input.type.tensor_type.shape.dim[1].dim_value = graph_input_length*k
        graph_output.type.tensor_type.shape.dim[0].dim_value = 1
        graph_output.type.tensor_type.shape.dim[1].dim_value = graph_output_length*k

        for x in node_origin:
            graph_node.remove(x)

        # Add Concat Zero Matrix
        mat = np.zeros((graph_output_length, k*graph_output_length))
        concat_zero_matrix_name = "concat_zero_matrix"
        tensor = onnx.helper.make_tensor(
            name=concat_zero_matrix_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(graph_output_length, k*graph_output_length),
            vals=mat.flatten()
        )
        initializer.append(tensor)

        # concat_name="Concat"
        # concat_names=[]
        for step in range(k):
            # Add Split node
            start_index = step*graph_input_length
            indices_name = convert_name("SplitIndeices", step)
            mat = np.zeros((k*graph_input_length, graph_input_length))
            for i in range(graph_input_length):
                mat[start_index+i, i] = 1

            tensor = onnx.helper.make_tensor(
                name=indices_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=(k*graph_input_length, graph_input_length),
                vals=mat.flatten()
            )
            initializer.append(tensor)

            split_name = convert_name("Split", step)
            split_node = onnx.helper.make_node(
                name=split_name,
                op_type="MatMul",
                inputs=[graph_input.name, indices_name],
                outputs=[split_name]
            )
            graph_node.append(split_node)

            # Input copy

            # input=copy.deepcopy(graph_input)
            # input.name=nameconverter(graph_input.name,step)
            # model.graph.input.append(input)
            # input=onnx.helper.make_tensor_value_info(nameconverter(graph_input.name,step),
            #     elem_type=graph_input.type.tensor_type.elem_type,
            #     shape=[1,30])
            # model.graph.input.append(input)

            # #Output copy
            # output=copy.deepcopy(graph_output)
            # output.name=nameconverter(graph_output.name,step)
            # model.graph.output.append(output)

            # Node copy
            end_name = convert_name("", step)
            for x in node_origin:
                # print(x)
                input_names = []
                output_names = []
                for name in x.input:
                    if name in initializer_name:
                        input_name = name
                    elif name == graph_input_origin:
                        input_name = graph_input.name  # TODO
                        input_name = split_name
                    else:
                        input_name = convert_name(name, step)
                    input_names.append(input_name)

                for name in x.output:
                    if name == graph_output_origin:
                        # output_name=output.name
                        output_name = convert_name(name, step)
                        end_name = output_name
                        # concat_names.append(nameconverter(name,step))
                    else:
                        output_name = convert_name(name, step)
                    output_names.append(output_name)

                node = onnx.helper.make_node(
                    name=convert_name(name, step),
                    op_type=x.op_type,
                    inputs=input_names,
                    outputs=output_names,
                    doc_string=x.doc_string,
                    domain=x.domain
                )
                node.attribute.extend(x.attribute)
                graph_node.append(node)

            # Add Merge node
            start_index = step*graph_output_length
            indices_name = convert_name("MergeIndeices", step)
            mat = np.zeros((graph_output_length, k*graph_output_length))
            for i in range(graph_output_length):
                mat[i, start_index+i] = 1

            tensor = onnx.helper.make_tensor(
                name=indices_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=(graph_output_length, k*graph_output_length),
                vals=mat.flatten()
            )
            initializer.append(tensor)

            merge_name = convert_name("Merge", step)
            merge_node = onnx.helper.make_node(
                name=merge_name,
                op_type="MatMul",
                inputs=[end_name, indices_name],
                outputs=[merge_name]
            )
            graph_node.append(merge_node)

            # Add Concat Node
            concat_name = convert_name("Concat", step)
            prev_concat_name = convert_name("Concat", step-1)
            if step == 0:
                prev_concat_name = concat_zero_matrix_name
            if step == k-1:
                concat_name = graph_output.name
            print(prev_concat_name, concat_name, merge_name)
            concat_node = onnx.helper.make_node(
                name=concat_name,
                op_type="Add",
                inputs=[merge_name, prev_concat_name],
                outputs=[concat_name]
            )
            graph_node.append(concat_node)

        # concat_node=onnx.helper.make_node(
        #     name=concat_name,
        #     op_type="Concat",
        #     inputs=concat_names,
        #     axis=1,
        #     outputs=[graph_output.name]
        # )
        # graph_node.append(concat_node)

        # concat_node=onnx.helper.make_node(
        #     name=concat_name,
        #     op_type="Concat",
        #     inputs=concat_names,
        #     axis=1,
        #     outputs=[graph_output.name]
        # )
        # graph_node.append(concat_node)

        # graph.output.remove(graph_output)
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
