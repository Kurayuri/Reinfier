from .. import util
from .. import CONSTANT
import ast

import astpretty
import copy
import numpy
import yapf
import os
import astor
import sys


class DRLPParsingError(Exception):
    def __init__(self, msg: str, *args: object, lineno=None, col_offset=None):
        if lineno is not None:
            prefix = f"line {lineno}"
            if col_offset is not None:
                prefix = f"{prefix}, col {col_offset}"
            msg = f"{prefix}: {msg}"
        super().__init__(msg, *args)


class DRLPTransformer(ast.NodeTransformer):
    input_size_id = "x_size"
    output_size_id = "y_size"
    input_id = "x"
    output_id = "y"
    expectation_delimiter = "@Exp"
    precondition_delimiter = "@Pre"
    dnnp_input_id = "x"
    dnnp_network_alias = "N"
    dnnp_output_id = "%s(%s)" % (
        dnnp_network_alias, dnnp_input_id)
    unwinding_id = "k"

    dnnp_and_id = "And"
    dnnp_or_id = "Or"
    dnnp_impiles_id = "Implies"
    dnnp_forall_id = "Forall"
    dnnp_shape_of_dim_0 = "[0]"

    def __init__(self, depth):
        self.depth = depth

        self.input_size = None
        self.output_size = None

        self.iter_vals = {}
        self.iter_ids = []

    def calculate(self, node):
        expr = ast.Expression(body=node)
        ast.fix_missing_locations(expr)
        src = eval(compile(expr, filename="", mode="eval"))
        src = str(src)
        root = ast.parse(src)
        return root.body[0].value

    def flatten_list(self, lst):
        return [x for y in lst for x in y]

    def is_dim_n_List(self, node, n: int):
        if isinstance(node, ast.List):
            if n == 1:
                # if len(node.elts) != 0 and isinstance(node.elts[0], ast.Constant):
                if len(node.elts) != 0 and self.is_Constant(node.elts[0]):
                    return True
                else:
                    return False
            if len(node.elts) != 0 and isinstance(node.elts[0], ast.List):
                return self.is_dim_n_List(node.elts[0], n - 1)
        return False

    def is_Constant(self, node):
        if (isinstance(node, ast.Constant) or
                (isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant))):
            return True
        return False

# %% 1. Get input_size and output_size
#   2. Replace parameters
#   3. Calculate expression
#   4. Unroll For and With


class DRLPTransformer_Init(DRLPTransformer):
    def __init__(self, depth, kwargs):
        self.kwargs = kwargs
        super().__init__(depth)

    # Read input_size and output_size
    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        for target in node.targets:
            if target.id == self.input_size_id:
                if self.input_size is None:
                    self.input_size = node.value.value
                    return None
                else:
                    assert self.input_size == node.value.value, "input sizes are not equal"
            if target.id == self.output_size_id:
                if self.output_size is None:
                    self.output_size = node.value.value
                    return None
                else:
                    assert self.output_size == node.value.value, "output sizes are not equal"
        return node

    #
    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        if ((isinstance(node.left, ast.List) and isinstance(node.right, ast.Constant)) or
                (isinstance(node.right, ast.List) and isinstance(node.left, ast.Constant))):
            if isinstance(node.op, ast.Mult):
                return self.calculate(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            return self.calculate(node)
        return node

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        elements = [node.left] + node.comparators

        io_element = None
        for element in elements:
            if isinstance(element, ast.Name):
                if element.id == self.input_id or element.id == self.output_id:
                    io_element = element
        size = None
        if io_element is not None:
            for element in elements:
                if element is not io_element and self.is_dim_n_List(element, 2):
                    size = len(element.elts[0].elts)
        if size is not None:
            if io_element.id == self.input_id:
                if self.input_size is None:
                    self.input_size = size
                else:
                    assert self.input_size == size, "input sizes are not equal"
            if io_element.id == self.output_id:
                if self.output_size is None:
                    self.output_size = size
                else:
                    assert self.output_size == size, "output sizes are not equal"
        return node

    # Loop unroll
    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        iter_id = node.target.id
        self.iter_ids.append(iter_id)
        if node.iter.func.id == "range" or node.iter.func.id == "orange":
            range_func = node.iter
            iter_start = range_func.args[0].value
            try:
                iter_stop = range_func.args[1].value
            except BaseException:
                iter_stop = iter_start
                iter_start = 0
            try:
                iter_step = range_func.args[2].value
            except BaseException:
                iter_step = 1
            args = []
            for i in range(iter_start, iter_stop, iter_step):
                self.iter_vals[iter_id] = i
                tmp = copy.deepcopy(node)
                tmp = self.generic_visit(tmp)
                args.append(tmp.body)

            if node.iter.func.id == "range":
                args = self.flatten_list(args)
                id = self.dnnp_and_id
                node = ast.Expr(
                    ast.Call(
                        func=ast.Name(id=id, ctx=ast.Load()),
                        args=args,
                        keywords=[]
                    ))
            elif node.iter.func.id == "orange":
                args_connect = []
                for arg in args:
                    if len(arg) > 1:
                        args_connect.append(ast.Call(
                            func=ast.Name(id=self.dnnp_and_id, ctx=ast.Load()),
                            args=arg,
                            keywords=[]
                        ))
                    else:
                        args_connect.append(arg[0])
                id = self.dnnp_or_id
                node = ast.Expr(
                    ast.Call(
                        func=ast.Name(id=id, ctx=ast.Load()),
                        args=args_connect,
                        keywords=[]
                    ))
        self.iter_ids.pop()

        return node

    def visit_Name(self, node: ast.Name):
        if node.id == self.unwinding_id:
            return ast.Constant(
                value=self.depth
            )
        if node.id in self.iter_ids:
            return ast.Constant(
                value=self.iter_vals[node.id]
            )
        if node.id in self.kwargs.keys():
            return ast.Constant(
                value=self.kwargs[node.id]
            )
        return node

    # With unroll
    def visit_With(self, node: ast.With):
        node = self.generic_visit(node)
        if node.items[0].context_expr.id == "orange":
            # if node.name=="orange":
            node = ast.Expr(
                ast.Call(
                    func=ast.Name(id=self.dnnp_or_id, ctx=ast.Load()),
                    args=node.body,
                    keywords=[]
                ))
        elif node.items[0].context_expr.id == "orange":
            # if node.name=="range":
            node = ast.Expr(
                ast.Call(
                    func=ast.Name(id=self.dnnp_and_id, ctx=ast.Load()),
                    args=node.body,
                    keywords=[]
                ))
        return node

# %% Subscript Transform


class DRLPTransformer_1(DRLPTransformer):
    def __init__(self, depth, input_size, output_size):
        super().__init__(depth)
        self.input_size = input_size
        self.output_size = output_size

    def visit_Subscript(self, node: ast.Subscript):

        index = None

        if sys.version_info >= (3, 9):
            # Dim 1
            if isinstance(node.value, ast.Name):
                node = self.generic_visit(node)
                if node.value.id == self.input_id or node.value.id == self.output_id:
                    if isinstance(node.slice, ast.Constant):
                        index = node.slice.value
                        if node.value.id == self.input_id:
                            lower = index * self.input_size
                            upper = lower + self.input_size
                        if node.value.id == self.output_id:
                            lower = index * self.output_size
                            upper = lower + self.output_size

                if index is not None:
                    if lower + 1 == upper:
                        node.slice = ast.Constant(value=lower)
                    else:
                        node.slice = ast.Slice(
                            lower=ast.Constant(value=lower),
                            upper=ast.Constant(value=upper)
                        )
            # Dim 2
            elif isinstance(node.value, ast.Subscript):
                self.generic_visit(node.value.value)
                self.generic_visit(node.value.slice)
                self.generic_visit(node.slice)
                if node.value.value.id == self.input_id or node.value.value.id == self.output_id:
                    if isinstance(node.value.slice, ast.Constant):
                        index = node.value.slice.value
                        if node.value.value.id == self.input_id:
                            lower = index * self.input_size
                        if node.value.value.id == self.output_id:
                            lower = index * self.output_size
                if index is not None:
                    if isinstance(node.slice, ast.Constant):
                        lower = lower + node.slice.value
                        upper = lower + 1
                    if isinstance(node.slice, ast.Slice):
                        upper = lower + node.slice.upper.value
                        lower = lower + node.slice.lower.value
                    node.value = node.value.value
                    # node.value=ast.Name(id=node.value.value.id,ctx=ast.Load())

                    if lower + 1 == upper:
                        node.slice = ast.Constant(value=lower)
                    else:
                        node.slice = ast.Slice(
                            lower=ast.Constant(value=lower),
                            upper=ast.Constant(value=upper)
                        )

        elif sys.version_info >= (3, 8):
            # Dim 1
            if isinstance(node.value, ast.Name):
                node = self.generic_visit(node)
                if node.value.id == self.input_id or node.value.id == self.output_id:
                    if isinstance(node.slice, ast.Slice):
                        index = node.slice.lower.value
                        if node.value.id == self.input_id:
                            lower = node.slice.lower.value * self.input_size
                            upper = node.slice.upper.value * self.input_size
                        if node.value.id == self.output_id:
                            lower = node.slice.lower.value * self.output_size
                            upper = node.slice.upper.value * self.output_size
                    # if isinstance(node.slice,ast.Slice):
                    #     pass
                    elif isinstance(node.slice.value, ast.Constant):
                        index = node.slice.value.value
                        if node.value.id == self.input_id:
                            lower = index * self.input_size
                            upper = lower + self.input_size
                        if node.value.id == self.output_id:
                            lower = index * self.output_size
                            upper = lower + self.output_size
                if index is not None:
                    if lower + 1 == upper:
                        node.slice = ast.Index(ast.Constant(value=lower))
                    else:
                        node.slice = ast.Slice(
                            lower=ast.Constant(value=lower),
                            upper=ast.Constant(value=upper),
                            step=None
                        )
            # Dim 2
            elif isinstance(node.value, ast.Subscript):
                self.generic_visit(node.value.value)
                self.generic_visit(node.value.slice)
                self.generic_visit(node.slice)
                if node.value.value.id == self.input_id or node.value.value.id == self.output_id:
                    if isinstance(node.value.slice.value, ast.Constant):
                        index = node.value.slice.value.value
                        if node.value.value.id == self.input_id:
                            lower = index * self.input_size
                        if node.value.value.id == self.output_id:
                            lower = index * self.output_size
                if index is not None:
                    if isinstance(node.slice, ast.Index):
                        lower = lower + node.slice.value.value
                        upper = lower + 1
                    if isinstance(node.slice, ast.Slice):
                        upper = lower + node.slice.upper.value
                        lower = lower + node.slice.lower.value
                    node.value = node.value.value
                    # node.value=ast.Name(id=node.value.value.id,ctx=ast.Load())

                    if lower + 1 == upper:
                        node.slice = ast.Constant(value=lower)
                    else:
                        node.slice = ast.Slice(
                            lower=ast.Constant(value=lower),
                            upper=ast.Constant(value=upper),
                            step=None
                        )

        return node

    def visit_List(self, node):
        node = self.generic_visit(node)
        if self.is_dim_n_List(node, 2):
            elements = [j for i in node.elts for j in i.elts]
            node.elts = elements
        return node


# %% Name and List Transform
class DRLPTransformer_2(DRLPTransformer):
    def __init__(self, depth):
        super().__init__(depth)

    def visit_Name(self, node: ast.Name):
        if node.id == self.input_id:
            return ast.Name(
                id=self.dnnp_input_id + self.dnnp_shape_of_dim_0,
                ctx=ast.Load()
            )
        if node.id == self.output_id:
            return ast.Name(
                id=self.dnnp_output_id + self.dnnp_shape_of_dim_0,
                ctx=ast.Load()
            )
        return node

    def visit_List(self, node):
        if len(node.elts) > 1:
            node = ast.Call(
                func=ast.Name(id="array", ctx=ast.Load()),
                # args=[ast.List(elts=[node],ctx=ast.Load())],
                args=[node],
                keywords=[]
            )
        else:
            node = node.elts[0]
        return node


# %% Fix Input and Output Subscript Size Transform (k-induction)
class DRLPTransformer_Induction(DRLPTransformer):
    def __init__(self, depth, input_size, output_size, to_fix_subscript=True):
        super().__init__(depth)
        self.input_size = input_size
        self.output_size = output_size
        self.fix_subsript = to_fix_subscript
    # Remove Init Constraint

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Compare):
            elements = [node.value.left] + node.value.comparators
            init_element = None
            for element in elements:
                if isinstance(element, ast.Subscript):
                    if element.value.id == self.dnnp_input_id + self.dnnp_shape_of_dim_0:
                        if (element.slice.lower.value == self.input_size * 0 and
                                element.slice.upper.value == self.input_size * 1):
                            init_element = element
            is_init = True
            if init_element is not None:
                for element in elements:
                    if element is not init_element:
                        if isinstance(element, ast.Call):
                            if element.func.id != "array":
                                is_init = False
                        else:
                            is_init = False

                if is_init == True:
                    return None
        return node

    # Fix Input and Output Subscript Size Transform
    def visit_Name(self, node: ast.Name):
        if self.fix_subsript == True:
            if node.id == self.dnnp_input_id + self.dnnp_shape_of_dim_0:
                return ast.Name(
                    id=node.id +
                    "[0:%d]" % (self.depth * self.input_size),
                    ctx=ast.Load()
                )
            if node.id == self.dnnp_output_id + self.dnnp_shape_of_dim_0:
                return ast.Name(
                    id=node.id +
                    "[0:%d]" % (self.depth * self.output_size),
                    ctx=ast.Load()
                )
        return node

    def visit_Subscript(self, node: ast.Subscript):
        return node
