from .. import util
from .. import CONSTANT
from .DRLP import DRLP
from .DNNP import DNNP
from .lib import *
from .DRLPTransformer import *
import ast
import astpretty
import yapf
import astor


src = '''
a=[1,2]
@Pre
y_size=1

[[-1]*2]*k <= x <= [[a]*2]*k

[0]*2 <= x[0] <= [0]*2

for i in range(0,k):
    Implies(y[i] > [1],  x[i]+0.5 >= x[i+1] >= x[i])
    Implies(y[i] <= [1], x[i]-0.5 <= x[i+1] <= x[i])

@Exp
y >= [[0]]*k
'''






# %% Parse DRLP PQ
def parse_drlp(drlp: DRLP, depth: int, kwgs: dict = {}) -> DNNP:
    if isinstance(drlp, DNNP):
        return drlp

    filename, drlp = read_drlp(drlp)
    drlp_v, drlp = split_drlp_vpq(drlp)
    drlp_p, drlp_q = split_drlp_pq(drlp)

    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    # util.log(astpretty.pformat(ast_root_p, show_offsets=False))

    (ast_root_p, ast_root_q), input_size, output_size = transform_pipeline((ast_root_p, ast_root_q), depth, kwgs)

    # Make and save
    dnnp_root = make_dnnp(ast_root_p, ast_root_q)
    dnnp = save_dnnp(dnnp_root, filename, depth)

    return dnnp


# %% Parse DRLP PQ for k-induction
def parse_drlp_induction(drlp: DRLP, depth: int, kwargs: dict = {}) -> DNNP:
    if isinstance(drlp, DNNP):
        return drlp

    filename, drlp = read_drlp(drlp)
    drlp_v, drlp = split_drlp_vpq(drlp)
    drlp_p, drlp_q = split_drlp_pq(drlp)

    # k+1 Precondition
    depth += 1
    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    # util.log(astpretty.pformat(ast_root_p,show_offsets=False))

    (ast_root_p, ast_root_q), input_size, output_size = transform_pipeline((ast_root_p, ast_root_q), depth, kwargs)

    transformer_indction = DRLPTransformer_Induction(depth, input_size, output_size, to_fix_subscript=False)
    ast_root_p = transformer_indction.visit(ast_root_p)

    # Assume i~i+k holds
    depth -= 1

    ast_root_q_ = ast.parse(drlp_q)
    (ast_root_q_,), ins, outs = transform_pipeline((ast_root_q_,), depth, kwargs)

    transformer_indction = DRLPTransformer_Induction(depth, input_size, output_size, to_fix_subscript=True)
    ast_root_q_ = transformer_indction.visit(ast_root_q_)

    # Make and save
    ast_root_p.body += ast_root_q_.body

    dnnp_root = make_dnnp(ast_root_p, ast_root_q)
    dnnp = save_dnnp(dnnp_root, filename.rsplit(".")[0] + "!ind", depth)

    return dnnp

# %% Parse DRLP VPQ


def parse_drlps(drlp: DRLP, depth: int, to_induct: bool = False, to_filter_unused_variables: bool = True):
    filename, drlp = read_drlp(drlp)
    drlp_v, drlp_pq = split_drlp_vpq(drlp)

    varibles_v = exec_drlp_v(drlp_v)
    variables_pq = get_variables_pq(drlp_pq)

    if to_filter_unused_variables:
        varibles_v = filter_unused_variables(varibles_v, variables_pq)

    kwargss = get_product(varibles_v)

    dnnps = []

    for kwargs in kwargss:
        if not to_induct:
            dnnp = parse_drlp(DRLP(drlp_pq), depth, kwargs)
        else:
            dnnp = parse_drlp_induction(DRLP(drlp_pq), depth, kwargs)

        dnnps.append(dnnp)

    return dnnps

# %% Parse DRLP VPQ for k-induction


def parse_drlps_induction(drlp: str, depth: int, to_filter_unused_variables: bool = True):
    return parse_drlps(drlp, depth, True, to_filter_unused_variables)


# %% Parse DRLP V
def parse_drlp_v(drlp: DRLP, to_filter_unused_variables: bool = True):
    kwargss = get_product(get_variables(drlp))
    filename, drlp = read_drlp(drlp)
    drlp_v, drlp_pq = split_drlp_vpq(drlp)
    drlp_p, drlp_q = split_drlp_pq(drlp_pq)

    drlps = []

    for kwargs in kwargss:
        ast_root_p = ast.parse(drlp_p)
        ast_root_q = ast.parse(drlp_q)
        __, (ast_root_p, ast_root_q) = transform(DRLPTransformer_VR(kwargs=kwargs), (ast_root_p, ast_root_q))

        drlp_pqi = make_drlp(ast_root_p, ast_root_q)
        util.log("## DRLP\n", drlp_pqi)
        drlps.append(DRLP(drlp_pqi, kwargs))

    return drlps


def parse_drlp_get_constraint(drlp: DRLP, kwgs: dict = {}) -> DNNP:
    if isinstance(drlp, DNNP):
        return drlp
    filename, drlp = read_drlp(drlp)
    drlp_v, drlp = split_drlp_vpq(drlp)
    drlp_p, drlp_q = split_drlp_pq(drlp)

    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)

    transformer, (ast_root_p, ast_root_q) = transform(DRLPTransformer_Init(1), (ast_root_p, ast_root_q))
    input_size = transformer.input_size
    output_size = transformer.output_size

    __, (ast_root_p, ast_root_q) = transform(DRLPTransformer_RIC(input_size, output_size), (ast_root_p, ast_root_q))
    __, (ast_root_p, ast_root_q) = transform(DRLPTransformer_REC(), (ast_root_p, ast_root_q))

    drlp_pqi = make_drlp(ast_root_p, ast_root_q)
    return DRLP(drlp_pqi)


if __name__ == "__main__":
    parse_drlp(src, 3)
    # parse_drlp_induction(src, 3)
