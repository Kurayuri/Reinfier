from ..import CONSTANT
from ..import Setting
from ..import util
from .DRLPTransformer import *
from .error import *
from .DRLP import DRLP
from .DNNP import DNNP
from .import lib
import astpretty
import itertools
import astor
import yapf
import ast
import re
import os

yamf_style = '''
[style]
based_on_style = pep8
column_limit=90
dedent_closing_brackets = true
'''


def split_drlp_pq(drlp_pq):
    try:
        drlp_pq = re.split("%s[^\n]*\n" % (DRLPTransformer.EXPECTATION_DELIMITER), drlp_pq)
        if len(drlp_pq) != 2:
            raise Exception
    except Exception:
        raise DRLPParsingError(f'Invalid DRLP format, DRLP cannot be splitted by EXPECTATION_DELIMITER "@Exp"\n{drlp_pq}')
    return drlp_pq[0], drlp_pq[1]


def split_drlp_vpq(drlp_vpq):
    try:
        drlp_vpq = re.split("%s[^\n]*\n" % (DRLPTransformer.PRECONDITION_DELIMITER), drlp_vpq)
        if len(drlp_vpq) != 2:
            drlp_vpq.insert(0, "")
    except Exception as e:
        raise DRLPParsingError(f'Invalid DRLP format, DRLP cannot be splitted by PRECONDITION_DELIMITER "@Pre"\n{drlp_vpq}')
    return drlp_vpq[0], drlp_vpq[1]


def format_dnnp(code: str):
    path = f"./{Setting.TmpPath}/style.style_config"
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(yamf_style)
    code, changed = yapf.yapflib.yapf_api.FormatCode(
        code, style_config=path)
    # os.remove("./style.style_config")
    util.log("\n## DNNP:")
    util.log(code)
    return code


def make_dnnp(ast_root_p, ast_root_q):
    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    node_pq = []
    # Add And
    for ast_root in [ast_root_p, ast_root_q]:
        if len(ast_root.body) == 0:
            node = ast.Constant(value=True, kind=None)
        elif len(ast_root.body) == 1:
            node = ast_root.body[0]
        else:
            node = ast.Call(
                func=ast.Name(id=DRLPTransformer.DNNP_AND_ID, ctx=ast.Load()),
                args=ast_root.body,
                keywords=[]
            )
        node_pq.append(node)

    # Add Impies
    impiles_node = ast.Call(
        func=ast.Name(id=DRLPTransformer.DNNP_IMPILES_ID, ctx=ast.Load()),
        args=node_pq,
        keywords=[]
    )

    forall_node = ast.Expr(ast.Call(
        func=ast.Name(id=DRLPTransformer.DNNP_FORALL_ID, ctx=ast.Load()),
        args=[
            ast.Name(id=DRLPTransformer.DNNP_INPUT_ID, ctx=ast.Load()),
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
        ast.Import(
            names=[ast.alias(name='numpy', asname='np')],
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
                ast.Name(id=DRLPTransformer.DNNP_NETWORK_ALIAS, ctx=ast.Store)],
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


def make_pq(ast_root_p, ast_root_q):
    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    expr_true = ast.Expr(value=ast.Constant(value=True, kind=None))

    for ast_root in [ast_root_p, ast_root_q]:
        if len(ast_root.body) == 0:
            ast_root.body.append(expr_true)

    drlp_pi = astor.to_source(ast_root_p)
    drlp_qi = astor.to_source(ast_root_q)

    drlp_pqi = "\n".join((drlp_pi, DRLPTransformer.EXPECTATION_DELIMITER, drlp_qi))

    return drlp_pqi


def read_drlp(property: DRLP):
    path = property.path
    drlp_vpq = property.obj
    filename = util.lib.get_filename_from_path(path)
    return filename, drlp_vpq


def transform(transformer: DRLPTransformer, ast_roots):
    for ast_root in ast_roots:
        ast_root = transformer.visit(ast_root)
    return transformer, ast_roots


def transform_pipeline(ast_roots, depth: int, kwargs: dict, input_size: int = None, output_size: int = None):
    transformer_init = DRLPTransformer_Init(depth, kwargs)
    for ast_root in ast_roots:
        ast_root = transformer_init.visit(ast_root)

    if input_size is None:
        input_size = transformer_init.input_size
    if output_size is None:
        output_size = transformer_init.output_size
    variables = transformer_init.variables

    transformers = [
        DRLPTransformer_Concretize(kwargs),
        DRLPTransformer_1(depth, input_size, output_size),
        DRLPTransformer_2(depth),
        DRLPTransformer_RSC()
    ]
    for transformer in transformers:
        for ast_root in ast_roots:
            ast_root = transformer.visit(ast_root)

    util.log("Input size:", input_size)
    util.log("Output size:", output_size)

    return ast_roots, input_size, output_size


def exec_drlp_v(drlp_v):
    exec(drlp_v)
    if isinstance(drlp_v, str):
        del drlp_v
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
    transformer = DRLPTransformer()
    for drlp_block in drlp_blocks:
        ast_root = ast.parse(drlp_block)
        ast_root = transformer.visit(ast_root)
    variables = transformer.variables
    return variables


def get_variables(property: DRLP, to_filter_unused_variables: bool = True):
    drlp_vpq = property.obj
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp_vpq)
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


