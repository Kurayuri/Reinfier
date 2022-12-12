from .. import util
from .. import CONSTANT
import ast

import astpretty
import contextlib
import copy
import itertools
import numpy
import yapf
import re
import os
import astor
import copy
import sys
from .DRLPTransformer import DRLPParsingError, DRLPTransformer, DRLPTransformer_Init, DRLPTransformer_1, DRLPTransformer_2, DRLPTransformer_Induction


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

yamf_style = '''
[style]
based_on_style = pep8
column_limit=90
dedent_closing_brackets = true
'''


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

    unwinded_dnnp_filename = util.util.get_savepath(filename, depth, "dnnp")
    with open(unwinded_dnnp_filename, "w") as f:
        f.write(code)
    util.log("\n## DNNP Filename:", level=CONSTANT.INFO)
    util.log(unwinded_dnnp_filename, level=CONSTANT.INFO)
    return code, unwinded_dnnp_filename


def read_drlp(drlp):
    filename = drlp
    try:
        with open(filename) as f:
            drlp = f.read()
    except Exception:
        filename = "tmp.drlp"
    filename = util.util.get_filename_from_path(filename)
    return filename, drlp


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


def transform(ast_roots, depth: int, kwgs: dict):
    transformer_init = DRLPTransformer_Init(depth, kwgs)
    for ast_root in ast_roots:
        ast_root = transformer_init.visit(ast_root)
    input_size = transformer_init.input_size
    output_size = transformer_init.output_size

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


def get_product(dicts):
    iterable_variables = {}
    normal_variables = {}
    for k, v in dicts.items():
        if k[0] == "_":
            iterable_variables[k[1:]] = v
        else:
            normal_variables[k] = v

    product = list(
        dict(zip(iterable_variables.keys(), values))
        for values in itertools.product(*iterable_variables.values())
    )
    for x in product:
        x.update(normal_variables)
    return product


# %% Parse DRLP


def parse_drlp(drlp: str, depth: int, kwgs: dict = {}):
    filename, drlp = read_drlp(drlp)
    drlp_v, drlp = split_drlp_vpq(drlp)
    drlp_p, drlp_q = split_drlp_pq(drlp)

    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    # util.log(astpretty.pformat(ast_root_p, show_offsets=False))

    (ast_root_p, ast_root_q), input_size, output_size = transform((ast_root_p, ast_root_q), depth, kwgs)

    # Make and save
    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    dnnp_root = make_dnnp(ast_root_p, ast_root_q)
    code, unwinded_dnnp_filename = save_dnnp(dnnp_root, filename, depth)

    return code, unwinded_dnnp_filename


# %% Parse DRLP for k-induction
def parse_drlp_induction(drlp: str, depth: int, kwargs: dict = {}):
    filename, drlp = read_drlp(drlp)
    drlp_v, drlp = split_drlp_vpq(drlp)
    drlp_p, drlp_q = split_drlp_pq(drlp)

    # k+1 Precondition
    depth += 1
    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    # util.log(astpretty.pformat(ast_root_p,show_offsets=False))

    (ast_root_p, ast_root_q), input_size, output_size = transform((ast_root_p, ast_root_q), depth, kwargs)

    transformer_indction = DRLPTransformer_Induction(depth, input_size, output_size, to_fix_subscript=False)
    ast_root_p = transformer_indction.visit(ast_root_p)

    # Assume i~i+k holds
    depth -= 1

    ast_root_q_ = ast.parse(drlp_q)
    (ast_root_q_,), ins, outs = transform((ast_root_q_,), depth, kwargs)

    transformer_indction = DRLPTransformer_Induction(depth, input_size, output_size, to_fix_subscript=True)
    ast_root_q_ = transformer_indction.visit(ast_root_q_)

    # Make and save
    ast_root_p.body += ast_root_q_.body
    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    dnnp_root = make_dnnp(ast_root_p, ast_root_q)
    code, unwinded_dnnp_filename = save_dnnp(dnnp_root, filename.rsplit(".")[0] + "!ind", depth)

    return code, unwinded_dnnp_filename


def parse_drlps(drlp: str, depth: int, to_induct: bool = False):
    filename, drlp = read_drlp(drlp)
    drlp_v, drlp_pq = split_drlp_vpq(drlp)

    varibles = exec_drlp_v(drlp_v)
    kwargss = get_product(varibles)

    codes = []
    unwinded_dnnp_filenames = []

    for kwargs in kwargss:
        if not to_induct:
            code, unwinded_dnnp_filename = parse_drlp(drlp_pq, depth, kwargs)
        else:
            code, unwinded_dnnp_filename = parse_drlp_induction(drlp_pq, depth, kwargs)

        codes.append(code)
        unwinded_dnnp_filenames.append(unwinded_dnnp_filename)

    return codes, unwinded_dnnp_filenames


def parse_drlps_induction(drlp: str, depth: int):
    return parse_drlps(drlp, depth, True)


if __name__ == "__main__":
    parse_drlp(src, 3)
    # parse_drlp_induction(src, 3)
