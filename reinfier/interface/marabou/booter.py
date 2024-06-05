from ...common.DNNP import DNNP
from ...common.DRLP import DRLP
from ...common.Feature import *
from ...common.NN import NN
from ...import CONST
from ...import Setting
from ...import util
from ...import drlp
from ...import nn
from ..docker import docker
from typing import Tuple, Dict
import numpy as np
import subprocess
import copy
import time
import os


class MarabouExitcode:
    UNSAT = "UNSAT"
    SAT = "SAT"
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    QUIT_REQUESTED = "QUIT_REQUESTED"


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
         property: DRLP | None,
         verifier: None = None,
         network_alias: None = None,
         violation: str | None = None,
         inputFtrs: Dict[int, Feature] | None = None,
         outputFtrs: Dict[int, Feature] | None = None,
         enabled_outputFtrs_complement: bool = True
         ) -> Tuple[bool, bool, float, np.ndarray]:

    containor_name = Setting.ContainerNames[CONST.MARABOU]
    run_dk = False

    if violation is None:
        violation_path = util.io.get_savepath([network.path, property.path],
                                               None, "npy")

    if run_dk:
        save_dirpath = util.io.get_savepath_container()
        network_path = os.path.join(save_dirpath,
                                    os.path.basename(network.path))
        property_path = os.path.join(save_dirpath,
                                     os.path.basename(property.path))
        _violation_path = violation_path
        violation_path = os.path.join(save_dirpath,
                                      os.path.basename(violation_path))
        docker.copy_in(containor_name, [network.path, property.path], save_dirpath)
    else:
        network_path = network.path
        property_path = property.path

    if (not network.isValid()) or (not property.isValid()):
        return (False, None, float('inf'), None)

    util.log_prompt(1)
    util.log("Single DNN Query Verifying...", level=CONST.INFO)
    util.log("## Info:", level=CONST.INFO)

    # Pre
    if inputFtrs is None and outputFtrs is None:
        _, (inputDynamic, outputDynamic), (inputStatic, outputStatic) \
            = drlp.parse_drlp_get_constraint(property)

        inputFtrs = {**inputDynamic, **inputStatic} if inputFtrs is None else inputFtrs
        outputFtrs = {**outputDynamic, **outputStatic} if outputFtrs is None else outputFtrs

    from maraboupy import Marabou
    verbosity = 2 if Setting.LogLevel == CONST.DEBUG else 0
    verbose = True if Setting.LogLevel <= CONST.INFO else 0
    options = Marabou.createOptions(verbosity=verbosity)

    net = Marabou.read_onnx(network.path)
    inputVars = net.inputVars[0].flatten()
    outputVars = net.outputVars[0].flatten()

    def run(net, inputFtrs, outputFtrs):
        util.log(outputFtrs, level=CONST.INFO)

        for idx, var in enumerate(inputVars):
            net.setLowerBound(var, inputFtrs[idx].lower)
            net.setUpperBound(var, inputFtrs[idx].upper)

        for idx, var in enumerate(outputVars):
            outputFtr = outputFtrs[idx]

            if outputFtr.lower is not None:
                net.setLowerBound(var, outputFtr.lower)
            if outputFtr.upper is not None:
                net.setUpperBound(var, outputFtr.upper)

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
                 level=CONST.INFO)
        return runable, result, dtime, violation

    outputConditions = [outputFtrs]
    if enabled_outputFtrs_complement:
        outputConditions = []
        outputSpaces = calc_complement_spaces(outputFtrs.values())
        for outputSpace in outputSpaces:
            outputIntvs = {
                idx: interval for idx, interval in zip(outputFtrs.keys(), outputSpace)
            }
            outputConditions.append(outputIntvs)

    runable, result, sum_time, violation = False, True, 0, None

    for outputCondition in outputConditions:
        _runable, _result, _dtime, violation = run(copy.deepcopy(net),
                                                   inputFtrs,
                                                   outputCondition)
        runable = _runable or runable
        result = _result and result
        sum_time += _dtime
        if result == False:
            break

    util.log("\n## Ans:", level=CONST.WARNING)
    util.log(f"Runable: {runable}   Result: {result}   Time: {sum_time}",
             level=CONST.WARNING)
    if result == False:
        nn.run_onnx(network, violation)
    util.log_prompt(2)
    return runable, result, sum_time, violation
