import json
from .. import dnnv
from .. import nn
from .. import drlp
from .. import CONSTANT
from .. import Setting
from .. import util


def select_verifier(networks, properties, verifiers: list = None, network_alias: str = "N"):
    util.log("## Selecting verifier...", level=CONSTANT.INFO)
    if verifiers is None:
        if Setting.SelectionFull:
            verifiers = CONSTANT.VERIFIER
        else:
            verifiers = CONSTANT.RUNABLE_VERIFIER
    util.log("Runable verifiers:", level=CONSTANT.INFO)
    util.log(verifiers, level=CONSTANT.INFO)

    log_level = Setting.LogLevel
    if Setting.LogLevel == CONSTANT.DEBUG:
        Setting.set_LogLevel(CONSTANT.DEBUG)
    else:
        Setting.set_LogLevel(CONSTANT.ERROR)

    if isinstance(networks, str) and isinstance(properties, str):
        network = networks
        property = properties
        networks = []
        properties = []
        for i in range(1, 3):
            # TODO
            networks.append({
                True: nn.expander.unwind_network(network, i, branchable=True),
                False: nn.expander.unwind_network(network, i, branchable=False),
            })
            # nn.expander.unwind_network(network, i))

            code, dnnp = drlp.parser.parse_drlp(property, i)
            properties.append(dnnp)

    status = {}
    num = len(networks)
    for verifier in verifiers:
        util.log(("Testing...", verifier), level=CONSTANT.CRITICAL)
        status[verifier] = {
            "runable": True,
            "time_sum": 0.0,
            "log": []
        }
        for j in range(0, num):
            network = networks[j][nn.util.is_branchable(verifier)]
            property = properties[j]
            runable, result, time = dnnv.booter.boot_dnnv(
                network=network, property=property, verifier=verifier)
            status[verifier]["log"].append({
                "network": network,
                "property": property,
                "runable": runable,
                "result": result,
                "time": time
            })
            status[verifier]["time_sum"] += time
            if runable == False:
                status[verifier]["runable"] = False

    Setting.set_LogLevel(log_level)

    time_sum_min = float("inf")
    time_sum_min_verifier = None
    for key in status.keys():
        if status[key]["runable"] == True and status[key]["time_sum"] < time_sum_min:
            time_sum_min_verifier = key
            time_sum_min = status[key]["time_sum"]
    util.log(json.dumps(status, indent=4), level=CONSTANT.DEBUG)
    util.log("## Chosen verifier: \n" + time_sum_min_verifier + "\n", level=CONSTANT.INFO)

    return time_sum_min_verifier


if __name__ == "__main__":
    verifier = select_verifier("test01.onnx", "test01_p1.drlp")
    print(verifier)
