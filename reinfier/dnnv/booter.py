import contextlib
import os
import numpy as np
import subprocess
from .. import nn
from .. import utils
import re
import sys
from dnnv.__main__ import _main as dnnv_main



def boot_dnnv(network: str, property: str, verifier: str = "eran",
              network_alias: str = "N", violation: str = None):
    if violation == None:
        violation = "_".join([network, property, "violation.npy"])

    verifier = "--"+verifier
    dnnv = "dnnv"
    cmd = [dnnv,
           property,
           "--network", network_alias, network,
           verifier,
           "--save-violation", violation
           ]

    if os.path.exists(violation):
        os.remove(violation)

    # Call DNNV
    sys.argv = cmd
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])

    dnnv_out=open("dnnv_out","w") 
    with utils.io.output_wrapper(dnnv_out):
        dnnv_main()

    dnnv_out.close()
    with open("dnnv_out","r") as f:
        x=f.read()
    os.remove("dnnv_out")
    x = x.split("\n")

    # Check DNNV output
    time = float('inf')
    result = False
    runable = False

    try:
        for i in range(len(x)-1, -1, -1):
            if "time: " in x[i]:
                line = x[i]
                line = line.split(": ")
                time = float(line[1])
            if "result: " in x[i]:
                line = x[i]
                line = line.split(": ")
                if line[1] == "unsat" or line[1] == "unknown":  # TODO
                    # if line[1] == "unsat":
                    result = True
                    runable = True
                elif line[1] == "sat":
                    result = False
                    runable = True
                else:
                    runable = False
                break
    except Exception as e:
        print(e)
        runable = False

    print(runable, result, time)

    if runable == True:
        if result == False:
            ans = np.load(violation)
            print("SAT")
            nn.onnx_runner.run_onnx(network=network, input=ans)
        else:
            print("UNSAT")
    else:
        print("Error")

    return runable, result, time


if __name__ == "__main__":
    verifier = "marabou"
    network = "test01.onnx"
    property = "test01_p1.dnnp"
    boot_dnnv(network=network, property=property, verifier=verifier)
