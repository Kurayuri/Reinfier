import json
from .. import dnnv
from .. import nn
from .. import drlp


def select_verifier(networks, properties, verifiers: list = None, network_alias: str = "N"):
    if verifiers == None:
        verifiers = ['bab', 'eran', 'marabou', 'mipverify',
                     'neurify', 'nnenum', 'planet', 'reluplex', 'verinet']
    
    
    if isinstance(networks,str) and isinstance(properties,str):
        network=networks
        property=properties
        networks=[]
        properties=[]
        for i in range(1,3):
            networks.append(nn.expander.unwind_network(network,i))
            code,dnnp=drlp.parse.parse_drlp(property,i)
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
            network = networks[j]
            property = properties[j]
            runable, result, time = dnnv.booter.boot_dnnv(
                network=network, property=property, verifier=verifier)
            status[verifier]["log"].append({
                "network":network,
                "property":property,
                "runable":runable,
                "result":result,
                "time": time
            })
            status[verifier]["time_sum"]+=time
            if runable==False:
                status[verifier]["runable"]=False

    print(json.dumps(status,indent=4))

    time_sum_min=float("inf")
    time_sum_min_verifier=None
    for key in status.keys():
        if status[key]["runable"]==True and status[key]["time_sum"]<time_sum_min:
            time_sum_min_verifier=key
            time_sum_min=status[key]["time_sum"]
    
    return time_sum_min_verifier


if __name__=="__main__":
    verifier= select_verifier("test01.onnx","test01_p1.drlp")
    print(verifier)