from .  import selector
from .. import dnnv
from .. import nn
from .. import drlp
from .. import CONSTANT

def logging(*args):
    with open("log.txt",'a') as f:
        args=[str(arg) for arg in args]
        f.write(" ".join(args)+"\n")

def is_branchable(verifier):
    return verifier==CONSTANT.MARABOU

def bmc(network,drlp_txt,k_max=10,verifier=None,k_min=1):
    if verifier is None:
        verifier=selector.select_verifier(network, drlp_txt)

    for k in range(k_min,k_max+1):
        dnn=nn.expander.unwind_network(network,k,branchable=is_branchable(verifier))
        code,dnnp=drlp.parser.parse_drlp(drlp_txt,k)
        runable, result, time=dnnv.booter.boot_dnnv(dnn,dnnp,verifier)
        if result == False:
            return k,False
    return k_max,None
        
def  k_induction(network,drlp_txt,k_max=10,verifier=None,k_min=1):
    logging(network,drlp_txt,"k_induction")
    if verifier is None:
        verifier=selector.select_verifier(network, drlp_txt)
        
    for k in range(k_min,k_max+1):
        dnn=nn.expander.unwind_network(network,k,branchable=is_branchable(verifier))
        code,dnnp=drlp.parser.parse_drlp(drlp_txt,k)
        runable, result, time=dnnv.booter.boot_dnnv(dnn,dnnp,verifier)
        logging(k,runable, result, time,"base")
        if result == True:
            dnn=nn.expander.unwind_network(network,k+1,branchable=is_branchable(verifier))
            code,dnnp=drlp.parser.parse_drlp_induction(drlp_txt,k)
            runable, result, time=dnnv.booter.boot_dnnv(dnn,dnnp,verifier)
            logging(k,runable, result, time,"induction")
            if result == True:
                return k,True
            else:
                continue
        elif result == False:
            return k,False
    return k_max,None



if __name__=="__main__":
    final_k,ans=k_induction("test01.onnx","test01_p1.drlp",5,"marabou")
    print('''%s%s Property is %s with k=%d %s'''%('\n'*5,"*"*10,ans,final_k,"*"*10))