from ..import util
from ..common.Feature import Dynamic, Static
from .DRLPTransformer import *
from ..common.DRLP import DRLP
from ..common.DNNP import DNNP
from ..common.aliases import PropertyFeatures

from .lib import *
from typing import List, Tuple, Dict
import astpretty
import astor
import ast
VIOLATED_ID = "violated"
OCCURRED_ID = "occurred"
IS_VIOLATED_ID = "is_violated"


def parse_drlp(property: DRLP, depth: int, kwgs: dict = {}) -> DNNP:
    '''Parse DRLP PQ'''
    if isinstance(property, DNNP):
        return property

    filename, drlp_vpq = read_drlp(property)
    drlp_v, drlp_pq = split_drlp_vpq(drlp_vpq)
    drlp_p, drlp_q = split_drlp_pq(drlp_pq)

    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    # util.log(astpretty.pformat(ast_root_p, show_offsets=False))
    # pprint(ast_root_p)

    (ast_root_p, ast_root_q), input_size, output_size = transform_pipeline(
        (ast_root_p, ast_root_q), depth, kwgs)

    # Make and save
    dnnp_root = make_dnnp(ast_root_p, ast_root_q)
    dnnp = save_dnnp(dnnp_root, filename, depth)

    return dnnp


def parse_drlp_induction(property: DRLP, depth: int, kwargs: dict = {}) -> DNNP:
    '''Parse DRLP PQ for k-induction'''
    if isinstance(property, DNNP):
        return property

    filename, drlp_vpq = read_drlp(property)
    drlp_v, drlp_pq = split_drlp_vpq(drlp_vpq)
    drlp_p, drlp_q = split_drlp_pq(drlp_pq)

    # k+1 Precondition
    depth += 1
    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    # util.log(astpretty.pformat(ast_root_p,show_offsets=False))

    (ast_root_p, ast_root_q), input_size, output_size = transform_pipeline(
        (ast_root_p, ast_root_q), depth, kwargs)

    transformer_indction = DRLPTransformer_Induction(
        depth, input_size, output_size, to_fix_subscript=False)
    ast_root_p = transformer_indction.visit(ast_root_p)

    # Assume i~i+k holds
    depth -= 1

    ast_root_q_ = ast.parse(drlp_q)
    (ast_root_q_,), __, __ = transform_pipeline(
        (ast_root_q_,), depth, kwargs, input_size, output_size)

    transformer_indction = DRLPTransformer_Induction(
        depth, input_size, output_size, to_fix_subscript=True)
    ast_root_q_ = transformer_indction.visit(ast_root_q_)

    # Make and save
    ast_root_p.body += ast_root_q_.body

    dnnp_root = make_dnnp(ast_root_p, ast_root_q)
    dnnp = save_dnnp(dnnp_root, filename.rsplit(".")[0] + "!ind", depth)

    return dnnp


def parse_drlps(property: DRLP, depth: int, to_induct: bool = False, to_filter_unused_variables: bool = True) -> List[DNNP]:
    '''Parse DRLP VPQ'''
    filename, drlp_vpq = read_drlp(property)
    drlp_v, drlp_pq = split_drlp_vpq(drlp_vpq)

    varibles_v = exec_drlp_v(drlp_v)
    variables_pq = get_variables_pq(drlp_pq)

    if to_filter_unused_variables:
        varibles_v = filter_unused_variables(varibles_v, variables_pq)

    kwargss = get_product(varibles_v)

    dnnps = []

    for kwargs in kwargss:
        if not to_induct:
            dnnp = parse_drlp(DRLP(drlp_pq, filename=filename), depth, kwargs)
        else:
            dnnp = parse_drlp_induction(
                DRLP(drlp_pq, filename=filename), depth, kwargs)

        dnnps.append(dnnp)

    return dnnps


def parse_drlps_v(property: DRLP, to_filter_unused_variables: bool = True) -> List[DRLP]:
    '''Parse DRLP V'''
    kwargss = get_product(get_variables(property))
    filename, drlp_vpq = read_drlp(property)
    drlp_v, drlp_pq = split_drlp_vpq(drlp_vpq)
    drlp_p, drlp_q = split_drlp_pq(drlp_pq)

    property_pqs = []

    for kwargs in kwargss:
        ast_root_p = ast.parse(drlp_p)
        ast_root_q = ast.parse(drlp_q)
        __, (ast_root_p, ast_root_q) = transform(
            DRLPTransformer_Concretize(kwargs=kwargs), (ast_root_p, ast_root_q))

        drlp_pqi = make_pq(ast_root_p, ast_root_q)
        util.log("## DRLP:\n", drlp_pqi)
        property_pqs.append(DRLP(drlp_pqi, kwargs, filename=filename))

    return property_pqs


def parse_drlp_get_constraint(property: DRLP) -> Tuple[DRLP, PropertyFeatures, PropertyFeatures]:
    if isinstance(property, DNNP):
        return property
    filename, drlp_vpq = read_drlp(property)
    drlp_v, drlp_pq = split_drlp_vpq(drlp_vpq)
    drlp_p, drlp_q = split_drlp_pq(drlp_pq)

    varibles_v = exec_drlp_v(drlp_v)
    kwargs = get_product(varibles_v)[0]

    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)

    transformer, (ast_root_p, ast_root_q) = transform(
        DRLPTransformer_Init(1, kwargs=kwargs), (ast_root_p, ast_root_q))
    input_size = transformer.input_size
    output_size = transformer.output_size

    # TODO
    # __, (ast_root_p, ast_root_q) = transform(DRLPTransformer_RIC(input_size, output_size), (ast_root_p, ast_root_q))
    transformer, (ast_root_p, ast_root_q) = transform(
        DRLPTransformer_RSC(), (ast_root_p, ast_root_q))
    transformer, (ast_root_p, ast_root_q) = transform(
        DRLPTransformer_Boundary(input_size, output_size), (ast_root_p, ast_root_q))

    drlp_pqi = make_pq(ast_root_p, ast_root_q)
    return DRLP(drlp_pqi, filename=filename), PropertyFeatures(transformer.input_dynamics, transformer.output_dynamics), PropertyFeatures(transformer.input_statics, transformer.output_statics)


def parse_constaint_to_code(property: DRLP, dynamics, statics) -> str:
    def ast_method():
        filename, drlp_vpq = read_drlp(property)
        drlp_v, drlp_pq = split_drlp_vpq(drlp_vpq)
        drlp_p, drlp_q = split_drlp_pq(drlp_pq)

        ast_root_p = ast.parse(drlp_p)
        ast_root_q = ast.parse(drlp_q)
        transformer, (ast_root_p, ast_root_q) = transform(
            DRLPTransformer_Init(1), (ast_root_p, ast_root_q))
        transformer, (ast_root_p, ast_root_q) = transform(
            DRLPTransformer_SplitCompare(), (ast_root_p, ast_root_q))

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
            body=[
                ast.Assign(
                    targets=[ast.Name(id=OCCURRED_ID, ctx=ast.Store())],
                    value=ast.Constant(value=True, kind=None),
                    type_comment=None,
                ),
                node_q_if
            ],
            orelse=[]
        )

        py_node = ast.parse("")
        py_node.body = [
            ast.Import(
                names=[ast.alias(name='numpy', asname='np')],
                # names=[ast.alias(name='array')],
                level=0
            ),
            ast.FunctionDef(
                name=IS_VIOLATED_ID,
                args=ast.arguments(
                    args=[
                        ast.arg(arg=DRLPTransformer.INPUT_ID, annotation=None),
                        ast.arg(arg=DRLPTransformer.OUTPUT_ID,
                                annotation=None),
                    ],
                    defaults=[], vararg=None, kwarg=None
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=VIOLATED_ID, ctx=ast.Store())],
                        value=ast.Constant(value=False, kind=None),
                        type_comment=None,
                    ),
                    ast.Assign(
                        targets=[ast.Name(id=OCCURRED_ID, ctx=ast.Store())],
                        value=ast.Constant(value=False, kind=None),
                        type_comment=None,
                    ),
                    node_p_if,
                    ast.Return(
                        value=ast.Tuple(
                            elts=[
                                ast.Name(id=OCCURRED_ID, ctx=ast.Load()),
                                ast.Name(id=VIOLATED_ID, ctx=ast.Load())
                            ],
                            ctx=ast.Load()
                        ))
                ],
                decorator_list=[]
            )
        ]
        code = astor.to_source(py_node)
        return code

    def str_method():
        input_srcs = []
        for idx, feature in {**dynamics[0], **statics[0]}.items():
            op = ">=" if feature.lower_closed else ">"
            input_srcs.append(
                f'''{DRLPTransformer.INPUT_ID}[0][{idx}] {op} {feature.lower}''')
            op = "<=" if feature.upper_closed else "<"
            input_srcs.append(
                f'''{DRLPTransformer.INPUT_ID}[0][{idx}] {op} {feature.upper}''')
        output_srcs = []
        for idx, feature in {**dynamics[1], **statics[1]}.items():
            op = ">=" if feature.lower_closed else ">"
            output_srcs.append(
                f'''{DRLPTransformer.OUTPUT_ID}[0][{idx}] {op} {feature.lower}''')
            op = "<=" if feature.upper_closed else "<"
            output_srcs.append(
                f'''{DRLPTransformer.OUTPUT_ID}[0][{idx}] {op} {feature.upper}''')

        code = f'''
def {IS_VIOLATED_ID}({DRLPTransformer.INPUT_ID}, {DRLPTransformer.OUTPUT_ID}):
    {VIOLATED_ID} = False
    {OCCURRED_ID} = False
    if ''' + " and ".join(input_srcs) + f''' :
        {OCCURRED_ID} = True
        if not (''' + " and ".join(output_srcs) + f'''):
            {VIOLATED_ID} = True
    return {OCCURRED_ID}, {VIOLATED_ID}
    '''
        return code

    # return str_method()
    return ast_method()


def parse_pq(property: DRLP, depth: int, kwargs: dict = {}, to_induct: bool = False) -> DNNP:
    '''API to parse delp_pq part
       Note that any statements in drlp_v are ignored, which may cause unconcretized values'''

    if to_induct:
        return parse_drlp_induction(property, depth, kwargs)
    else:
        return parse_drlp(property, depth, kwargs)


def parse_vpq(property: DRLP, depth: int, kwargs: dict = {}, to_induct: bool = False, to_filter_unused_variables: bool = True) -> List[DNNP]:
    '''API to parse drlp_vpq part'''
    return parse_drlps(property, depth, to_induct, to_filter_unused_variables)


def parse_v(property: DRLP, to_filter_unused_variables: bool = True) -> List[DRLP]:
    '''API to parse drlp_v part'''
    return parse_drlps_v(property, to_filter_unused_variables)
