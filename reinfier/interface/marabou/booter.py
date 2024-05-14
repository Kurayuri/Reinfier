from ...drlp.DNNP import DNNP
from ...drlp.DRLP import DRLP
from ...drlp.Feature import *
from ...nn.NN import NN
from ...import CONSTANT
from ...import Setting
from ...import util
from ...import drlp
from ...import nn
from ..import dk
from typing import Tuple, Dict
import numpy as np
import subprocess
import time
import os

# from dnnv.__main__ import _main as dnnv_main


def is_to_retry(txt: str):
    if "Assertion `fixval != -kHighsInf' failed" in txt:
        return True
    return False


# def log_marabou_output(stdout):
#     if Setting.LogLevel == CONSTANT.DEBUG:
#         util.log(("## Info:"), level=CONSTANT.INFO)
#         util.log(("\n".join(stdout)), level=CONSTANT.DEBUG)

#     else:
#         if ans_gotten:
#             util.log(("## Info:"), level=CONSTANT.INFO)
#             util.log(("\n".join(stdout[:-4])), level=CONSTANT.DEBUG)
#             util.log(("\n".join(stdout[-4:])), level=CONSTANT.INFO)
#         else:
#             util.log(("## Error:"), level=CONSTANT.INFO)
#             util.log(("\n".join(stderr[:-5])), level=CONSTANT.DEBUG)
#             util.log(("\n".join(stderr[-5:])), level=CONSTANT.INFO)

def exec_docker(property_path, network_path):
    cmd = [
        "docker",
        "exec",
        Setting.ContainerName,
        "mkdir",
        Setting.TmpPath
    ]
    output_bytes = subprocess.run(cmd, capture_output=True, text=True)

    cmd = [
        "docker",
        "cp",
        property_path,
        f"{Setting.ContainerName}:/home/dnnv/{property_path}"
    ]
    output_bytes = subprocess.run(cmd, capture_output=True, text=True)

    cmd = [
        "docker",
        "cp",
        network_path,
        f"{Setting.ContainerName}:/home/dnnv/{network_path}"
    ]
    output_bytes = subprocess.run(cmd, capture_output=True, text=True)


def boot(network: NN,
         property: DRLP,
         verifier: None = None,
         network_alias: None = None,
         violation: str = None) -> Tuple[bool, bool, float, np.ndarray]:

    containor_name = Setting.ContainerNames[CONSTANT.MARABOU]
    run_dk = False

    if violation is None:
        violation_path = util.lib.get_savepath([network.path, property.path],
                                               None, "npy")

    if run_dk:
        save_dirpath = util.lib.get_savepath_container()
        network_path = os.path.join(save_dirpath,
                                    os.path.basename(network.path))
        property_path = os.path.join(save_dirpath,
                                     os.path.basename(property.path))
        _violation_path = violation_path
        violation_path = os.path.join(save_dirpath,
                                      os.path.basename(violation_path))
        dk.copy_in(containor_name, [network.path, property.path], save_dirpath)
    else:
        network_path = network.path
        property_path = property.path

    if (network.obj is None and network.path is None) or \
            (property.obj is None and property.path is None):
        return (False, None, float('inf'), None)

    class MarabouExitcode:
        UNSAT = "UNSAT"
        SAT = "SAT"
        UNKNOWN = "UNKNOWN"
        TIMEOUT = "TIMEOUT"
        ERROR = "ERROR"
        QUIT_REQUESTED = "QUIT_REQUESTED"

    util.log_prompt(1)
    util.log("Single DNN Query Verifying...", level=CONSTANT.INFO)
    util.log("## Info:", level=CONSTANT.INFO)

    # Pre
    inputFtrs: Dict[int, Feature]
    outputFtrs: Dict[int, Feature]
    _, (inputDynamic, outputDynamic), (inputStatic, outputStatic) \
        = drlp.parse_drlp_get_constraint(property)

    inputFtrs = {**inputDynamic, **inputStatic}
    outputFtrs = {**outputDynamic, **outputStatic}

    from maraboupy import Marabou
    verbosity = 2 if Setting.LogLevel == CONSTANT.DEBUG else 0
    verbose = True if Setting.LogLevel <= CONSTANT.INFO else 0
    options = Marabou.createOptions(verbosity=verbosity)

    def run(inputFtrs, outputFtrs):
        net = Marabou.read_onnx(network.path)
        inputVars = net.inputVars[0].flatten()
        outputVars = net.outputVars[0].flatten()
        util.log(outputFtrs, level=CONSTANT.INFO)
        for idx, var in enumerate(inputVars):
            net.setLowerBound(var, inputFtrs[idx].lower)
            net.setUpperBound(var, inputFtrs[idx].upper)

        for idx, var in enumerate(outputVars):
            if outputFtrs[idx].lower is not None:
                net.setLowerBound(var, outputFtrs[idx].lower)
            if outputFtrs[idx].upper is not None:
                net.setUpperBound(var, outputFtrs[idx].upper)

        for idx, var in enumerate(outputVars):
            outputFtr = outputFtrs[idx]
            if outputFtr.lower is not None:
                Feature(None, outputFtr.lower)
            if outputFtr.upper is not None:
                Feature(outputFtr.upper, None)

            if outputFtrs[idx].lower is not None:
                net.setLowerBound(var, outputFtrs[idx].lower)
            if outputFtrs[idx].upper is not None:
                net.setUpperBound(var, outputFtrs[idx].upper)

        stime = time.time()
        exitCode, vals, stats = net.solve(verbose=verbose, options=options)
        dtime = time.time() - stime
        violation = None
        match exitCode.upper():
            case MarabouExitcode.UNSAT:
                result, runable = True, True
            case MarabouExitcode.SAT:
                result, runable = False, True
                violation = np.zeros_like(net.inputVars[0], dtype=np.float64)
                for idx, arr_idx in enumerate(np.ndindex(violation.shape)):
                    violation[arr_idx] = vals[idx]
                np.save(violation_path, violation)
            case MarabouExitcode.UNKNOWN:
                result, runable = None, True
            case MarabouExitcode.TIMEOUT:
                result, runable = None, True
            case _:
                result, runable = None, False

        util.log(f"Runable: {runable}   Result: {result}   Time: {dtime}\n",
                 level=CONSTANT.INFO)
        return runable, result, dtime, violation

    complement_spaces = calc_complement_spaces(outputFtrs.values())
    runable, result, sum_time, violation = False, True, 0, None

    for complement_space in complement_spaces:
        complement_outputIntvs = {
            idx: interval
            for idx, interval in enumerate(complement_space)
        }
        _runable, _result, _dtime, violation = run(inputFtrs,
                                                   complement_outputIntvs)
        runable = _runable or runable
        result = _result and result
        sum_time += _dtime
        if result == False:
            break

    util.log("\n## Ans:", level=CONSTANT.WARNING)
    util.log(f"Runable: {runable}   Result: {result}   Time: {sum_time}",
             level=CONSTANT.WARNING)
    if result == False:
        nn.run_onnx(network, violation)
    util.log_prompt(2)
    return runable, result, sum_time, violation
