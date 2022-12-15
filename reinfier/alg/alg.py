from . import selector
from .. import dnnv
from .. import nn
from .. import drlp
from .. import CONSTANT


def log_call(*args):
    with open("log.txt", 'a+') as f:
        args = [str(arg) for arg in args]
        f.write(" ".join(args) + "\n")


def bmc(network, property, verifier=None, k_max=10, k_min=1):
    log_call(network, property, "bmc")

    if verifier is None:
        verifier = selector.select_verifier(network, property)

    for k in range(k_min, k_max + 1):

        dnn = nn.expander.unroll_nn(network, k, branchable=nn.lib.is_branchable(verifier))
        dnnp = drlp.parser.parse_drlp(property, k)

        runable, result, time = dnnv.booter.boot_dnnv(dnn, dnnp, verifier)
        log_call(k, runable, result, time, "base")

        if result == False:
            return k, False
    return k_max, None


def k_induction(network, property, verifier=None, k_max=10, k_min=1):
    log_call(network, property, "k_induction")

    if verifier is None:
        verifier = selector.select_verifier(network, property)

    for k in range(k_min, k_max + 1):

        dnn = nn.expander.unroll_nn(network, k, branchable=nn.lib.is_branchable(verifier))
        dnnp = drlp.parser.parse_drlp(property, k)

        runable, result, time = dnnv.booter.boot_dnnv(dnn, dnnp, verifier)
        log_call(k, runable, result, time, "base")

        if result == True:

            dnn = nn.expander.unroll_nn(network, k + 1, branchable=nn.lib.is_branchable(verifier))
            dnnp = drlp.parser.parse_drlp_induction(property, k)

            runable, result, time = dnnv.booter.boot_dnnv(dnn, dnnp, verifier)
            log_call(k, runable, result, time, "induction")

            if result == True:
                return k, True
            else:
                continue

        elif result == False:
            return k, False

    return k_max, None


def verify(network, drlp_txt, verifier=None, k_max=10, k_min=1):
    return k_induction(network, drlp_txt, verifier=verifier, k_max=k_max, k_min=k_min)


if __name__ == "__main__":
    final_k, ans = k_induction("test01.onnx", "test01_p1.drlp", 5, "marabou")
    print('''%s%s Property is %s with k=%d %s''' % ('\n' * 5, "*" * 10, ans, final_k, "*" * 10))
