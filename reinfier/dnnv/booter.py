import contextlib
import os
import numpy as np
import subprocess
from .. import nn
from .. import util
from .. import CONSTANT
import re
import sys
# from dnnv.__main__ import _main as dnnv_main


def is_retry(txt: str):
    if "Assertion `fixval != -kHighsInf' failed" in txt:
        return True
    return False


def boot_dnnv(network: str, property: str, verifier: str,
              network_alias: str = "N", violation: str = None):
    if violation is None:
        violation = util.util.get_savepath([network, property], None, "npy")

    assert verifier in CONSTANT.VERIFIER, "Unsupported verifier: %s" % verifier

    verifier = "--" + verifier
    dnnv = "dnnv"
    cmd = [dnnv,
           property,
           "--network", network_alias, network,
           verifier,
           "--save-violation", violation
           ]

    if os.path.exists(violation):
        os.remove(violation)

    while True:
        print("*" * 80 + "\n" + "Verifying...")
        print(" ".join(cmd))

        # %% Call DNNV from fucntion

        # # Call DNNV
        # sys.argv = cmd
        # sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])

        # dnnv_out_filename="dnnv.out"
        # dnnv_out=open(dnnv_out_filename,"w")
        # with util.io.output_wrapper(dnnv_out):
        #     dnnv_main()

        # dnnv_out.close()
        # with open(dnnv_out_filename,"r") as f:
        #     x=f.read()
        # os.remove(dnnv_out_filename)
        # x = x.split("\n")

        # %% Call DNNV from cmd

        myenv = os.environ.copy()
        if 'VIRTUAL_ENV' in os.environ:
            myenv['PATH'] = ':'.join(
                [x for x in os.environ['PATH'].split(':')
                    if x != os.path.join(os.environ['VIRTUAL_ENV'], 'bin')])

        try:
            # x = subprocess.check_output(cmd,stderr=subprocess.STDOUT)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dnnv_stdout = proc.stdout
            dnnv_stderr = proc.stderr
            # print(dnnv_stdout)
            # print(dnnv_stderr)
            dnnv_stdout = dnnv_stdout.split("\n")
            dnnv_stderr = dnnv_stderr.split("\n")
        except Exception as e:
            print(e)

        # %% Check DNNV output

        time = float('inf')
        result = False
        runable = False

        retry = False

        # Check dnnv_stderr
        try:
            for i in range(len(dnnv_stderr) - 1, -1, -1):
                if is_retry(dnnv_stderr[i]):
                    retry = True
                    break
        except BaseException:
            pass

        if retry:
            print("Retrying...")
            continue

        # %% Check dnnv_stdout
        try:
            got_info = False
            for i in range(len(dnnv_stdout) - 1, -1, -1):
                if "time: " in dnnv_stdout[i]:
                    line = dnnv_stdout[i]
                    line = line.split(": ")
                    time = float(line[1])
                    got_info = True
                elif "result: " in dnnv_stdout[i]:
                    got_info = True
                    line = dnnv_stdout[i]
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
            if got_info:
                print("## Info:")
                print("\n".join(dnnv_stdout[-4:]))
            else:
                print("## Error:")
                print("\n".join(dnnv_stderr[-5:]))
        except Exception as e:
            print(e)
            runable = False

        print("## Ans:")
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
        break
    print("\n\n" + "#" * 80 + "\n" + "#" * 80)
    return runable, result, time


if __name__ == "__main__":
    verifier = "marabou"
    network = "test01.onnx"
    property = "test01_p1.dnnp"
    boot_dnnv(network=network, property=property, verifier=verifier)
