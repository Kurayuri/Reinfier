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
VIOLATED_ID = "violated"
IS_VIOLATED_ID = "is_violated"


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


def parse_drlp(drlp: DRLP, depth: int, kwgs: dict = {}) -> DNNP:
    '''Parse DRLP PQ'''
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


def parse_drlp_induction(drlp: DRLP, depth: int, kwargs: dict = {}) -> DNNP:
    '''Parse DRLP PQ for k-induction'''
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
    (ast_root_q_,), __, __ = transform_pipeline((ast_root_q_,), depth, kwargs, input_size, output_size)

    transformer_indction = DRLPTransformer_Induction(depth, input_size, output_size, to_fix_subscript=True)
    ast_root_q_ = transformer_indction.visit(ast_root_q_)

    # Make and save
    ast_root_p.body += ast_root_q_.body

    dnnp_root = make_dnnp(ast_root_p, ast_root_q)
    dnnp = save_dnnp(dnnp_root, filename.rsplit(".")[0] + "!ind", depth)

    return dnnp


def parse_drlps(drlp: DRLP, depth: int, to_induct: bool = False, to_filter_unused_variables: bool = True):
    ''' Parse DRLP VPQ'''
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


def parse_drlps_induction(drlp: str, depth: int, to_filter_unused_variables: bool = True):
    ''' Parse DRLP VPQ for k-induction'''
    return parse_drlps(drlp, depth, True, to_filter_unused_variables)


def parse_drlps_v(drlp: DRLP, to_filter_unused_variables: bool = True):
    ''' Parse DRLP V'''
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
        util.log("## DRLP:\n", drlp_pqi)
        drlps.append(DRLP(drlp_pqi, kwargs))

    return drlps


def parse_drlp_get_constraint(drlp: DRLP) -> DRLP:
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
    __, (ast_root_p, ast_root_q) = transform(DRLPTransformer_RSC(), (ast_root_p, ast_root_q))

    drlp_pqi = make_drlp(ast_root_p, ast_root_q)
    return DRLP(drlp_pqi)


def parse_constaint_to_code(drlp: DRLP) -> str:
    filename, drlp = read_drlp(drlp)
    drlp_v, drlp = split_drlp_vpq(drlp)
    drlp_p, drlp_q = split_drlp_pq(drlp)

    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    transformer, (ast_root_p, ast_root_q) = transform(DRLPTransformer_Init(1), (ast_root_p, ast_root_q))

    # Expectation if test
    values = []
    for exp in ast_root_q.body:
        ops = exp
        if isinstance(exp, ast.Expr):
            ops = exp.value
            values.append(ops)
    node_q_test = ast.UnaryOp(
        op=ast.Not(),
        operand=ast.BoolOp(
            op=ast.And(),
            values=values
        )
    )
    # Expectation if
    node_q_if = ast.If(
        test=node_q_test,
        body=[
            ast.Assign(
                targets=[ast.Name(id=VIOLATED_ID, ctx=ast.Store())],
                value=ast.Constant(value=True, kind=None),
                type_comment=None,
            ),
        ],
        orelse=[]
    )

    # Precondition if test
    values = []
    for exp in ast_root_p.body:
        ops = exp
        if isinstance(exp, ast.Expr):
            ops = exp.value
            values.append(ops)
    node_p_test = ast.BoolOp(
        op=ast.And(),
        values=values
    )
    # Precondition if
    node_p_if = ast.If(
        test=node_p_test,
        body=[node_q_if],
        orelse=[]
    )

    py_node = ast.parse("")
    py_node.body = [
        ast.FunctionDef(
            name=IS_VIOLATED_ID,
            args=ast.arguments(
                args=[
                    ast.arg(arg=DRLPTransformer.INPUT_ID,annotation=None),
                    ast.arg(arg=DRLPTransformer.OUTPUT_ID,annotation=None),
                ],
                defaults=[],vararg=None,kwarg=None
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=VIOLATED_ID, ctx=ast.Store())],
                    value=ast.Constant(value=False, kind=None),
                    type_comment=None,
                ),
                node_p_if,
                ast.Return(
                    value=ast.Name(id=VIOLATED_ID, ctx=ast.Load())
                )
            ],
            decorator_list=[]
        )
    ]
    py_code = astor.to_source(py_node)
    return py_code


def parse_pq(drlp: DRLP, depth: int, kwargs: dict = {}, to_induct: bool = False) -> DNNP:
    '''API to parse DRLP_PQ part'''
    if to_induct:
        return parse_drlp_induction(drlp, depth, kwargs)
    else:
        return parse_drlp(drlp, depth, kwargs)


def parse_vpq(drlp: DRLP, depth: int, kwargs: dict = {}, to_induct: bool = False, to_filter_unused_variables: bool = True) -> DNNP:
    '''API to parse DRLP_VPQ part'''
    return parse_drlps(drlp, depth, to_induct, to_filter_unused_variables)


def parse_v(drlp: DRLP, to_filter_unused_variables: bool = True):
    '''API to parse DRLP_V part'''
    return parse_drlps_v(drlp, to_filter_unused_variables)


if __name__ == "__main__":
    parse_drlp(src, 3)
    # parse_drlp_induction(src, 3)
