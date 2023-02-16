from ..drlp.DNNP import DNNP
from ..nn.NN import NN
from ..import CONSTANT
from ..import Setting
from ..import util
from ..import nn
from typing import Tuple
import numpy as np
import subprocess
import os

# from dnnv.__main__ import _main as dnnv_main


def is_to_retry(txt: str):
    if "Assertion `fixval != -kHighsInf' failed" in txt:
        return True
    return False


def log_dnnv_output(stdout, stderr, ans_gotten):
    if Setting.LogLevel == CONSTANT.DEBUG:
        util.log(("## Error:"), level=CONSTANT.INFO)
        util.log(("\n".join(stderr)), level=CONSTANT.DEBUG)
        util.log(("## Info:"), level=CONSTANT.INFO)
        util.log(("\n".join(stdout)), level=CONSTANT.DEBUG)

    else:
        if ans_gotten:
            util.log(("## Info:"), level=CONSTANT.INFO)
            util.log(("\n".join(stdout[:-4])), level=CONSTANT.DEBUG)
            util.log(("\n".join(stdout[-4:])), level=CONSTANT.INFO)
        else:
            util.log(("## Error:"), level=CONSTANT.INFO)
            util.log(("\n".join(stderr[:-5])), level=CONSTANT.DEBUG)
            util.log(("\n".join(stderr[-5:])), level=CONSTANT.INFO)


def extract_stdout_ans(stdout):
    time = float('inf')
    result = False
    runable = False
    try:
        ans_gotten = False
        for i in range(len(stdout) - 1, -1, -1):
            if "time: " in stdout[i]:
                line = stdout[i]
                line = line.split(": ")
                time = float(line[1])
                ans_gotten = True
            elif "result: " in stdout[i]:
                ans_gotten = True
                line = stdout[i]
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
        util.log((e), level=CONSTANT.INFO)
        runable = False
    return ans_gotten, runable, result, time


def boot_dnnv(network: NN, property: DNNP, verifier: str,
              network_alias: str = "N", violation: str = None) -> Tuple[bool, bool, float, np.ndarray]:
    network_path = network.path
    property_path = property.path
    if violation is None:
        violation_path = util.lib.get_savepath([network_path, property_path], None, "npy")

    assert verifier in CONSTANT.VERIFIERS, "Unsupported verifier: %s" % verifier

    verifier = "--" + verifier
    dnnv = "dnnv"
    cmd = [dnnv,
           property_path,
           "--network", network_alias, network_path,
           verifier,
           "--save-violation", violation_path
           ]

    if os.path.exists(violation_path):
        os.remove(violation_path)

    while True:
        util.log_prompt(1)
        util.log("Verifying...", level=CONSTANT.INFO)
        util.log((" ".join(cmd)), level=CONSTANT.INFO)

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
            # util.log((dnnv_stdout),level=CONSTANT.INFO)
            # util.log((dnnv_stderr),level=CONSTANT.INFO)
            dnnv_stdout = dnnv_stdout.split("\n")
            dnnv_stderr = dnnv_stderr.split("\n")
        except Exception as e:
            util.log((e), level=CONSTANT.INFO)

        # %% Check DNNV output

        time = float('inf')
        result = False
        runable = False

        to_retry = False

        # Check dnnv_stderr
        try:
            for i in range(len(dnnv_stderr) - 1, -1, -1):
                if is_to_retry(dnnv_stderr[i]):
                    to_retry = True
                    break
        except BaseException:
            pass

        if to_retry:
            util.log(("Retrying..."), level=CONSTANT.INFO)
            continue

        # %% Check dnnv_stdout
        ans_gotten, runable, result, time = extract_stdout_ans(dnnv_stdout)
        log_dnnv_output(dnnv_stdout, dnnv_stderr, ans_gotten)

        util.log(("## Ans:"), level=CONSTANT.WARNING)
        util.log((runable, result, time), level=CONSTANT.WARNING)

        violation = None
        if runable == True:
            if result == False:
                violation = np.load(violation_path)
                util.log(("SAT"), level=CONSTANT.WARNING)
                nn.onnx_runner.run_onnx(network=network_path, input=violation)
            else:
                util.log(("UNSAT"), level=CONSTANT.WARNING)
        else:
            util.log(("Error"), level=CONSTANT.WARNING)
        break

    util.log_prompt(2)
    return runable, result, time, violation
