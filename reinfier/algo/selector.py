from ..common.DRLP import DRLP
from ..common.NN import NN
from ..import CONST
from ..import Setting
from ..import interface
from ..import drlp
from ..import util
from ..import nn
from typing import Tuple, Union, List
import json


def select_verifier(networks: Union[NN, List[NN]], properties: Union[DRLP, List[DRLP]], verifiers: list = None, network_alias: str = "N"):
    util.log("## Selecting verifier...", level=CONST.INFO)
    if verifiers is None:
        if Setting.ToTestAllVerifier:
            verifiers = CONST.VERIFIERS
        else:
            verifiers = CONST.RUNABLE_VERIFIERS
    util.log("Runable verifiers:", level=CONST.INFO)
    util.log(verifiers, level=CONST.INFO)

    log_level = Setting.LogLevel
    if Setting.LogLevel == CONST.DEBUG:
        Setting.set_LogLevel(CONST.DEBUG)
    else:
        Setting.set_LogLevel(CONST.ERROR)

    if isinstance(networks, NN) and isinstance(properties, DRLP):
        network = networks
        property = properties
        networks = []
        properties = []
        for i in range(1, 3):
            # TODO
            networks.append({
                True: nn.expander.unroll_nn(network, i, branchable=True),
                False: nn.expander.unroll_nn(network, i, branchable=False),
            })
            # nn.expander.unroll_nn(network, i))

            dnnp = drlp.parser.parse_vpq(property, i)[0]
            properties.append(dnnp)

    status = {}
    num = len(networks)
    for verifier in verifiers:
        util.log(("Testing...", verifier), level=CONST.ERROR)
        status[verifier] = {
            "runable": True,
            "time_sum": 0.0,
            "log": []
        }
        for j in range(0, num):
            network = networks[j][nn.lib.is_branchable(verifier)]
            property = properties[j]
            runable, result, time, __ = interface.dnnv.boot(
                network=network, property=property, verifier=verifier)
            util.log("    ", runable, result, time, level=CONST.ERROR)

            status[verifier]["log"].append({
                "network": network.path,
                "property": property.path,
                "runable": runable,
                "result": result,
                "time": time
            })
            status[verifier]["time_sum"] += time
            if runable == False:
                status[verifier]["runable"] = False

    Setting.set_LogLevel(log_level)

    time_sum_min = float("inf")
    time_sum_min_verifier = "None"
    for key in status.keys():
        if status[key]["runable"] == True and status[key]["time_sum"] < time_sum_min:
            time_sum_min_verifier = key
            time_sum_min = status[key]["time_sum"]
    util.log(json.dumps(status, indent=4), level=CONST.DEBUG)
    util.log("## Chosen verifier: \n" + time_sum_min_verifier + "\n", level=CONST.INFO)

    return time_sum_min_verifier
