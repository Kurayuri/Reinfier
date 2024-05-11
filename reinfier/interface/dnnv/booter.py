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

def boot(network: NN, property: DNNP, verifier: str,
              network_alias: str = "N", violation: str = None) -> Tuple[bool, bool, float, np.ndarray]:
    
    containor_name = Setting.ContainerNames[CONSTANT.DNNV]
    run_dk=True

    if violation is None:
        violation_path = util.lib.get_savepath([network.path, property.path], None, "npy")

    if run_dk:
        save_dirpath=util.lib.get_savepath_container()
        network_path=os.path.join(save_dirpath,os.path.basename(network.path))
        property_path=os.path.join(save_dirpath,os.path.basename(property.path))
        _violation_path=violation_path
        violation_path = os.path.join(save_dirpath,os.path.basename(violation_path))
        dk.copy_in(containor_name,[network.path, property.path], save_dirpath)
    else:
        network_path = network.path
        property_path = property.path



    if (network.obj is None and network.path is None) or \
            (property.obj is None and property.path is None):
        return (False, None, float('inf'), None)

    verifier = verifier.lower()
    if verifier not in CONSTANT.VERIFIERS:
        raise AssertionError(f"Unsupported verifier: {verifier}")

    executable = [". /home/dnnv/.venv/bin/activate && /home/dnnv/.venv/bin/dnnv"]
    cmd = executable + [
           property_path,
           "--network", network_alias, network_path,
           f"--{verifier}",
           "--save-violation", violation_path
           ]
    cmd_readable = executable + [
           f"'{property_path}'",
           "--network", network_alias, f"'{network_path}'",
           f"--{verifier}",
           "--save-violation", f"'{violation_path}'"
           ]



    if os.path.exists(violation_path):
        os.remove(violation_path)

    while True:
        util.log_prompt(1)
        util.log("Single DNN Query Verifying...", level=CONSTANT.INFO)
        util.log((" ".join(cmd_readable)), level=CONSTANT.INFO)

        myenv = os.environ.copy()

        if run_dk:
            try:
                exit_code, proc = dk.exec(containor_name, cmd)
                stdout = []
                stderr = []
                for chunk in proc:
                    stdout, stderr = chunk
                    # print(stdout)
                    util.log(stdout)
                    util.log(stderr)
            except Exception as e:
                util.log((e), level=CONSTANT.INFO)
        else:
            if 'VIRTUAL_ENV' in os.environ:
                myenv['PATH'] = ':'.join(
                    [x for x in os.environ['PATH'].split(':')
                        if x != os.path.join(os.environ['VIRTUAL_ENV'], 'bin')])

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                stdout = proc.stdout
                stderr = proc.stderr
                util.log(stdout)
                util.log(stderr)
                stdout = stdout.split("\n")
                stderr = stderr.split("\n")
            except Exception as e:
                util.log((e), level=CONSTANT.INFO)

        # %% Check DNNV output

        time = float('inf')
        result = False
        runable = False

        to_retry = False

        # Check dnnv_stderr
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

        # %% Check dnnv_stdout
        ans_gotten, runable, result, time = extract_stdout_ans(stdout)
        log_dnnv_output(stdout, stderr, ans_gotten)

        util.log(("## Ans:"), level=CONSTANT.WARNING)
        util.log(("Runable:", runable, "   Result:", result, "   Time:", time), level=CONSTANT.WARNING)

        violation = None
        if runable == True:
            if result == False:
                if run_dk:
                    dk.copy_out(containor_name,violation_path,Setting.TmpPath)
                    violation_path=_violation_path
                    network_path=network.path
                violation = np.load(violation_path)
                util.log(("False"), level=CONSTANT.WARNING)
                nn.onnx_runner.run_onnx(network=network_path, input=violation)
            else:
                util.log(("True"), level=CONSTANT.WARNING)
        else:
            util.log(("Error"), level=CONSTANT.WARNING)
        break

    util.log_prompt(2)
    return runable, result, time, violation