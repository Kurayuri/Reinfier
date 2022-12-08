import json
from .. import dnnv
from .. import nn
from .. import drlp
from .. import CONSTANT
from .. import Setting
from .. import util


def select_verifier(networks, properties, verifiers: list = None, network_alias: str = "N"):
    if verifiers is None:
        verifiers = CONSTANT.VERIFIER

    log_level = Setting.LogLevel
    Setting.set_LogLevel(CONSTANT.CRITICAL)

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

    print(json.dumps(status, indent=4))

    time_sum_min = float("inf")
    time_sum_min_verifier = None
    for key in status.keys():
        if status[key]["runable"] == True and status[key]["time_sum"] < time_sum_min:
            time_sum_min_verifier = key
            time_sum_min = status[key]["time_sum"]

    return time_sum_min_verifier


if __name__ == "__main__":
    verifier = select_verifier("test01.onnx", "test01_p1.drlp")
    print(verifier)
