from ...drlp.DNNP import DNNP
from ...nn.NN import NN
from ...import CONSTANT
from ...import Setting
from ...import util
from ...import nn
from ..import dk
from typing import Tuple
import numpy as np
import subprocess
import os
import re

# from dnnv.__main__ import _main as dnnv_main


def is_to_retry(txt: str):
    if "Assertion `fixval != -kHighsInf' failed" in txt:
        return True
    return False


def log_output(stdout, stderr, ans_gotten):
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
    pattern = re.compile(r'\x1b\[\d+m')
    try:
        ans_gotten = False
        for line in reversed(stdout):
            if "Total time cost" in line:
                ans_gotten = True
                line = pattern.sub('', line).split()
                time = float(line[3])

            elif "Result of the safety verification on the computed flowpipes" in line:
                ans_gotten = True
                line = pattern.sub('', line).split()

                if line[-1] == "SAFE" or line[-1] == "UNKNOWN":  # TODO
                    result = True
                    runable = True
                elif line[-1] == "UNSAFE":
                    result = False
                    runable = True
                else:
                    runable = False
                break
    except Exception as e:
        print(e)
        runable = False
    return ans_gotten, runable, result, time

def exec_docker(property_path,network_path):
    cmd = [
        "docker",
        "exec",
        Setting.ContainerName,
        "mkdir",
        Setting.TmpPath
        ]
    output_bytes = subprocess.run(cmd,capture_output=True, text=True)

    cmd = [
        "docker",
        "cp",
        property_path,
        f"{Setting.ContainerName}:/home/dnnv/{property_path}"
        ]
    output_bytes = subprocess.run(cmd,capture_output=True, text=True)

    cmd = [
        "docker",
        "cp",
        network_path,
        f"{Setting.ContainerName}:/home/dnnv/{network_path}"
        ]
    output_bytes = subprocess.run(cmd,capture_output=True, text=True)

def boot(network: NN, property: DNNP, violation: str = None) -> Tuple[bool, bool, float, np.ndarray]:
    network_path = network.path
    property_path = property.path

    containor_name = Setting.ContainerNames[CONSTANT.VERISIG]

    if violation is None:
        violation_path = util.lib.get_savepath([network_path, property_path], None, "npy")

    if (network.obj is None and network.path is None) or \
            (property.obj is None and property.path is None):
        return (False, None, float('inf'), None)
    
    network.to_yaml()
    network_path=util.lib.get_savepath_container("network", "yml")

    dk.write_in(containor_name,network.to_yaml(), network_path)

    executable = [           
            "/home/tmp/flowstar",
            "-t","32"
    ]

    property_path = "/home/verisig_models/ex1_tanh/ex1_tanh_p5.model"
    property_path = "/home/verisig_models/ex2_tanh/ex2_tanh_p6.model"
    property_path = "/home/verisig_models/tora_tanh/tora_tanh_p8.model"

    cmd = executable + [
           network_path,
           "<",
           property_path
           ]

    cmd_readable = executable + [
           f"'{network_path}'",
           "<",
           f"'{property_path}'"
           ]

    while True:
        util.log_prompt(1)
        util.log("Single DNN Query Verifying...", level=CONSTANT.INFO)
        util.log((" ".join(cmd_readable)), level=CONSTANT.INFO)

        try:
            exit_code, proc = dk.exec(containor_name, cmd)
            stdout = []
            stderr = []
            for chunk in proc:
                stdout, stderr = chunk
                # print(stdout)
                util.log(stdout)
        except Exception as e:
            util.log((e), level=CONSTANT.INFO)
        # %% Check output

        time = float('inf')
        result = False
        runable = False

        to_retry = False

        # Check stderr
        try:
            for i in range(len(stderr) - 1, -1, -1):
                if is_to_retry(stderr[i]):
                    to_retry = True
                    break
        except BaseException:
            pass

        if to_retry:
            util.log(("Retrying..."), level=CONSTANT.INFO)
            continue

        # %% Check stdout
        ans_gotten, runable, result, time = extract_stdout_ans(stdout)
        log_output(stdout, stderr, ans_gotten)

        util.log(("## Ans:"), level=CONSTANT.WARNING)
        util.log(("Runable:", runable, "   Result:", result, "   Time:", time), level=CONSTANT.WARNING)

        violation = None
        if runable == True:
            if result == False:
                # violation = np.load(violation_path)
                violation = None
                util.log(("False"), level=CONSTANT.WARNING)
                # nn.onnx_runner.run_onnx(network=network_path, input=violation)
            else:
                util.log(("True"), level=CONSTANT.WARNING)
        else:
            util.log(("Error"), level=CONSTANT.WARNING)
        break

    util.log_prompt(2)
    return runable, result, time, violation
