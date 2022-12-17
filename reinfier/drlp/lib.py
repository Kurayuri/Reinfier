from .. import util
from .. import CONSTANT
from .DRLPTransformer import *
from .DRLP import DRLP
from .DNNP import DNNP
from . import lib
import re
import ast
import astpretty
import itertools
import yapf
import os
import astor

yamf_style = '''
[style]
based_on_style = pep8
column_limit=90
dedent_closing_brackets = true
'''


def split_drlp_pq(drlp):
    try:
        drlp = re.split("%s[^\n]*\n" % (DRLPTransformer.expectation_delimiter), drlp)
        if len(drlp) != 2:
            raise Exception
    except Exception:
        raise DRLPParsingError("Invalid DRLP format")
    return drlp[0], drlp[1]


def split_drlp_vpq(drlp):
    try:
        drlp = re.split("%s[^\n]*\n" % (DRLPTransformer.precondition_delimiter), drlp)
        if len(drlp) != 2:
            drlp.insert(0, "")
    except Exception as e:
        raise DRLPParsingError("Invalid DRLP format")
    return drlp[0], drlp[1]


def format_dnnp(code: str):
    with open("./style.style_config", 'w') as f:
        f.write(yamf_style)
    code, changed = yapf.yapflib.yapf_api.FormatCode(
        code, style_config="./style.style_config")
    os.remove("./style.style_config")
    util.log("\n## DNNP:")
    util.log(code)
    return code


def make_dnnp(ast_root_p, ast_root_q):
    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    # Add And
    if len(ast_root_p.body) < 2:
        node_p = ast_root_p.body[0]
    else:
        node_p = ast.Call(
            func=ast.Name(id=DRLPTransformer.dnnp_and_id, ctx=ast.Load()),
            args=ast_root_p.body,
            keywords=[]
        )

    if len(ast_root_q.body) < 2:
        node_q = ast_root_q.body[0]
    else:
        node_q = ast.Call(
            func=ast.Name(id=DRLPTransformer.dnnp_and_id, ctx=ast.Load()),
            args=ast_root_q.body,
            keywords=[]
        )

    # Add Impies
    impiles_node = ast.Call(
        func=ast.Name(id=DRLPTransformer.dnnp_impiles_id, ctx=ast.Load()),
        args=[node_p, node_q],
        keywords=[]
    )

    forall_node = ast.Expr(ast.Call(
        func=ast.Name(id=DRLPTransformer.dnnp_forall_id, ctx=ast.Load()),
        args=[
            ast.Name(id=DRLPTransformer.dnnp_input_id, ctx=ast.Load()),
            impiles_node],
        keywords=[]
    ))

    # Add additional code
    dnnp_node = ast.parse("")
    dnnp_node.body = [
        ast.ImportFrom(
            module="numpy",
            names=[ast.alias(name='array', asname='array')],
            # names=[ast.alias(name='array')],
            level=0
        ),
        ast.ImportFrom(
            module="dnnv.properties",
            names=[ast.alias(name='*', asname=None)],
            # names=[ast.alias(name='array')],
            level=0
        ),
        ast.Assign(
            targets=[
                ast.Name(id=DRLPTransformer.dnnp_network_alias, ctx=ast.Store)],
            value=ast.Name(id='''Network("N")''', ctx=ast.Load())
        ),
        forall_node]
    return dnnp_node


def save_dnnp(dnnp_root, filename, depth):
    code = astor.to_source(dnnp_root)

    code = format_dnnp(code)

    path = util.lib.get_savepath(filename, depth, "dnnp")
    with open(path, "w") as f:
        f.write(code)
    util.log("\n## DNNP Filename:", level=CONSTANT.INFO)
    util.log(path, level=CONSTANT.INFO)
    return DNNP(path)


def make_drlp(ast_root_p, ast_root_q):
    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    drlp_pi = astor.to_source(ast_root_p)
    drlp_qi = astor.to_source(ast_root_q)
    drlp_pqi = "\n".join((drlp_pi, DRLPTransformer.expectation_delimiter, drlp_qi))

    return drlp_pqi


def read_drlp(drlp: DRLP):
    path = drlp.path
    drlp = drlp.obj
    filename = util.lib.get_filename_from_path(path)
    return filename, drlp


def transform(transformer: DRLPTransformer, ast_roots):
    for ast_root in ast_roots:
        ast_root = transformer.visit(ast_root)
    return transformer, ast_roots


def transform_pipeline(ast_roots, depth: int, kwargs: dict):
    transformer_init = DRLPTransformer_Init(depth, kwargs)
    for ast_root in ast_roots:
        ast_root = transformer_init.visit(ast_root)
    input_size = transformer_init.input_size
    output_size = transformer_init.output_size
    variables = transformer_init.variables

    transformers = [
        DRLPTransformer_1(depth, input_size, output_size),
        DRLPTransformer_2(depth),
    ]
    for transformer in transformers:
        for ast_root in ast_roots:
            ast_root = transformer.visit(ast_root)

    util.log("Input size:", input_size)
    util.log("Output size:", output_size)

    return ast_roots, input_size, output_size


def exec_drlp_v(drlp):
    exec(drlp)
    if isinstance(drlp, str):
        del drlp
    return locals()


def is_iterable_variable(id: str):
    return id[0] == "_"


def convert_iterable_variable_to_normal_variable(id: str):
    return id[1:]


def get_product(dicts):
    iterable_variables = {}
    normal_variables = {}
    for k, v in dicts.items():
        if k[0] == "_":
            iterable_variables[convert_iterable_variable_to_normal_variable(k)] = v
        else:
            normal_variables[k] = v

    product = list(
        dict(zip(iterable_variables.keys(), values))
        for values in itertools.product(*iterable_variables.values())
    )
    for x in product:
        x.update(normal_variables)
    return product


def get_variables_pq(drlp_pq):
    drlp_blocks = lib.split_drlp_pq(drlp_pq)
    transformer = DRLPTransformer(1)
    for drlp_block in drlp_blocks:
        ast_root = ast.parse(drlp_block)
        ast_root = transformer.visit(ast_root)
    variables = transformer.variables
    return variables


def get_variables(drlp: DRLP, to_filter_unused_variables: bool = True):
    drlp = drlp.obj
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp)
    drlp_p, drlp_q = lib.split_drlp_pq(drlp_pq)

    varibles_v = exec_drlp_v(drlp_v)
    variables_pq = get_variables_pq(drlp_pq)

    if to_filter_unused_variables:
        varibles_v = filter_unused_variables(varibles_v, variables_pq)

    return varibles_v


def filter_unused_variables(variables_v, variables_pq):
    for k in list(variables_v.keys()):
        ki = k
        if is_iterable_variable(k):
            ki = convert_iterable_variable_to_normal_variable(k)
        if ki not in variables_pq:
            variables_v.pop(k)
    return variables_v
