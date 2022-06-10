import onnxruntime
import numpy as np

# onnxruntime.set_default_logger_severity(0)

def run_onnx(network: str,input):
    if isinstance(input,str):
        input_array = np.load(input)
    elif isinstance(input,np.ndarray):
        input_array=input

    session = onnxruntime.InferenceSession(network,)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input_array})
    with open("sss.log",'a') as f:
        f.write(str(input_array)+"\n"+str(output)+"\n")
    print("With Input:\n",input_array,"\nOutput:\n",output)
    return output

if __name__=="__main__":
    run_onnx("test01.onnx",np.array([[-1,0.5]],dtype=np.float32))