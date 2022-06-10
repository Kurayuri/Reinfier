import os
import numpy as np
import subprocess
from .. import nn


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

    myenv = os.environ.copy()
    if 'VIRTUAL_ENV' in os.environ:
        myenv['PATH'] = ':'.join(
            [x for x in os.environ['PATH'].split(':')
                if x != os.path.join(os.environ['VIRTUAL_ENV'], 'bin')])

    print(" ".join(cmd))

    try:
        x = subprocess.check_output(cmd)
    except:
        x = b""
        pass

    x = str(x, 'utf-8')
    print(x)
    x = x.split("\n")

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
