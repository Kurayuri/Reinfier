import ast
from distutils.command.bdist import show_formats
from textwrap import indent
import astpretty
import copy
import numpy
import yapf
import os
import astor
import sys

src = '''
y_size=1

[[-1]*2]*k <= x <= [[1]*2]*k

[0]*2 <= x[0] <= [0]*2

for i in range(0,k):
    Implies(y[i] > [1],  x[i]+0.5 >= x[i+1] >= x[i])
    Implies(y[i] <= [1], x[i]-0.5 <= x[i+1] <= x[i])

# Exp
y >= [[0]]*k
'''

style = '''
[style]
based_on_style = pep8
column_limit=90
dedent_closing_brackets = true
'''

class DRLPParsingError(Exception):
    def __init__(self, msg: str, *args: object, lineno=None, col_offset=None):
        if lineno is not None:
            prefix = f"line {lineno}"
            if col_offset is not None:
                prefix = f"{prefix}, col {col_offset}"
            msg = f"{prefix}: {msg}"
        super().__init__(msg, *args)


class DRLPTransformer(ast.NodeTransformer):
    def __init__(self, unwinding):
        self.unwinding = unwinding

        self.input_size = None
        self.output_size = None

        self.iter_vals = {}
        self.iter_ids = []

        self.input_size_id = "x_size"
        self.output_size_id = "y_size"
        self.input_id = "x"
        self.output_id = "y"
        self.dnnp_input_id = "x"
        self.dnnp_network_alias = "N"
        self.dnnp_output_id = "%s(%s)" % (
            self.dnnp_network_alias, self.dnnp_input_id)
        self.unwinding_id = "k"

        self.dnnp_and_id = "And"
        self.dnnp_or_id = "Or"
        self.dnnp_impiles_id = "Implies"
        self.dnnp_forall_id = "Forall"
        self.dnnp_shape_of_dim_0="[0]"

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
                return self.is_dim_n_List(node.elts[0], n-1)
        return False

    def is_Constant(self, node):
        if (isinstance(node, ast.Constant) or
                (isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant))):
            return True
        return False


class DRLPTransformer_1(DRLPTransformer):
    def __init__(self, unwinding):
        super().__init__(unwinding)

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        for target in node.targets:
            if target.id == self.input_size_id:
                if self.input_size == None:
                    self.input_size = node.value.value
                    return None
                else:
                    assert self.input_size == node.value.value, "input sizes are not equal"
            if target.id == self.output_size_id:
                if self.output_size == None:
                    self.output_size = node.value.value
                    return None
                else:
                    assert self.output_size == node.value.value, "output sizes are not equal"
        return node

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
        elements = [node.left]+node.comparators

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
        if size != None:
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

    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        iter_id = node.target.id
        self.iter_ids.append(iter_id)
        if node.iter.func.id == "range" or node.iter.func.id == "orange":
            range_func = node.iter
            iter_start = range_func.args[0].value
            try:
                iter_stop = range_func.args[1].value
            except:
                iter_stop =iter_start
                iter_start=0
            try:
                iter_step = range_func.args[2].value
            except:
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
                value=self.unwinding
            )
        if node.id in self.iter_ids:
            return ast.Constant(
                value=self.iter_vals[node.id]
            )
        return node

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

# Subscript Transform


class DRLPTransformer_2(DRLPTransformer):
    def __init__(self, unwinding, input_size, output_size):
        super().__init__(unwinding)
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
                            lower = index*self.input_size
                            upper = lower+self.input_size
                        if node.value.id == self.output_id:
                            lower = index*self.output_size
                            upper = lower+self.output_size

                if index is not None:
                    if lower+1 == upper:
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
                            lower = index*self.input_size
                        if node.value.value.id == self.output_id:
                            lower = index*self.output_size
                if index is not None:
                    if isinstance(node.slice, ast.Constant):
                        lower = lower+node.slice.value
                        upper = lower+1
                    if isinstance(node.slice, ast.Slice):
                        upper = lower+node.slice.upper.value
                        lower = lower+node.slice.lower.value
                    node.value = node.value.value
                    # node.value=ast.Name(id=node.value.value.id,ctx=ast.Load())

                    if lower+1 == upper:
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
                    if isinstance(node.slice,ast.Slice):
                        index=node.slice.lower.value
                        if node.value.id == self.input_id:
                            lower = node.slice.lower.value*self.input_size
                            upper = node.slice.upper.value*self.input_size
                        if node.value.id == self.output_id:
                            lower = node.slice.lower.value*self.output_size
                            upper = node.slice.upper.value*self.output_size
                    # if isinstance(node.slice,ast.Slice):
                    #     pass
                    elif isinstance(node.slice.value, ast.Constant):
                        index = node.slice.value.value
                        if node.value.id == self.input_id:
                            lower = index*self.input_size
                            upper = lower+self.input_size
                        if node.value.id == self.output_id:
                            lower = index*self.output_size
                            upper = lower+self.output_size
                if index is not None:
                    if lower+1 == upper:
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
                            lower = index*self.input_size
                        if node.value.value.id == self.output_id:
                            lower = index*self.output_size
                if index is not None:
                    if isinstance(node.slice, ast.Index):
                        lower = lower+node.slice.value.value
                        upper = lower+1
                    if isinstance(node.slice, ast.Slice):
                        upper = lower+node.slice.upper.value
                        lower = lower+node.slice.lower.value
                    node.value = node.value.value
                    # node.value=ast.Name(id=node.value.value.id,ctx=ast.Load())

                    if lower+1 == upper:
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


# Name and List Transform
class DRLPTransformer_3(DRLPTransformer):
    def __init__(self, unwinding):
        super().__init__(unwinding)

    def visit_Name(self, node: ast.Name):
        if node.id == self.input_id:
            return ast.Name(
                id=self.dnnp_input_id+self.dnnp_shape_of_dim_0,
                ctx=ast.Load()
            )
        if node.id == self.output_id:
            return ast.Name(
                id=self.dnnp_output_id+self.dnnp_shape_of_dim_0,
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


# Fix Input and Output Subscript Size Transform (k-induction)
class DRLPTransformer_4(DRLPTransformer):
    def __init__(self, unwinding, input_size, output_size,fix_subscript=True):
        super().__init__(unwinding)
        self.input_size = input_size
        self.output_size = output_size
        self.fix_subsript=fix_subscript
    # Remove Init Constraint
    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Compare):
            elements = [node.value.left]+node.value.comparators
            init_element = None
            for element in elements:
                if isinstance(element, ast.Subscript):
                    if element.value.id == self.dnnp_input_id+self.dnnp_shape_of_dim_0:
                        if (element.slice.lower.value == self.input_size*0 and
                                element.slice.upper.value == self.input_size*1):
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
        if self.fix_subsript==True:
            if node.id == self.dnnp_input_id+self.dnnp_shape_of_dim_0:
                return ast.Name(
                    id=node.id +
                    "[0:%d]" % (self.unwinding*self.input_size),
                    ctx=ast.Load()
                )
            if node.id == self.dnnp_output_id+self.dnnp_shape_of_dim_0:
                return ast.Name(
                    id=node.id +
                    "[0:%d]" % (self.unwinding*self.output_size),
                    ctx=ast.Load()
                )
        return node

    def visit_Subscript(self, node: ast.Subscript):
        return node


def parse_drlp(drlp: str, unwinding: int):
    filename = drlp
    try:
        with open(filename) as f:
            drlp = f.read()
    except Exception:
        filename = "drlp"

    try:
        drlp = drlp.split('\n# Exp\n')
        if len(drlp) != 2:
            raise Exception
    except Exception:
        raise DRLPParsingError("invalid DRLP format")

    drlp_p = drlp[0]
    drlp_q = drlp[1]

    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)

    # astpretty.pprint(ast_root_p, show_offsets=False)

    transformer = DRLPTransformer(unwinding)

    transformer_1 = DRLPTransformer_1(unwinding)
    ast_root_p = transformer_1.visit(ast_root_p)
    ast_root_q = transformer_1.visit(ast_root_q)

    transformer_2 = DRLPTransformer_2(
        unwinding,
        transformer_1.input_size,
        transformer_1.output_size
    )
    ast_root_p = transformer_2.visit(ast_root_p)
    ast_root_q = transformer_2.visit(ast_root_q)

    transformer_3 = DRLPTransformer_3(unwinding)
    ast_root_p = transformer_3.visit(ast_root_p)
    ast_root_q = transformer_3.visit(ast_root_q)

    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    # Add And
    if len(ast_root_p.body) < 2:
        node_p = ast_root_p.body[0]
    else:
        node_p = ast.Call(
            func=ast.Name(id=transformer.dnnp_and_id, ctx=ast.Load()),
            args=ast_root_p.body,
            keywords=[]
        )
    if len(ast_root_q.body) < 2:
        node_q = ast_root_q.body[0]
    else:
        node_q = ast.Call(
            func=ast.Name(id=transformer.dnnp_and_id, ctx=ast.Load()),
            args=ast_root_q.body,
            keywords=[]
        )

    impiles_node = ast.Call(
        func=ast.Name(id=transformer.dnnp_impiles_id, ctx=ast.Load()),
        args=[node_p, node_q],
        keywords=[]
    )

    forall_node = ast.Expr(ast.Call(
        func=ast.Name(id=transformer.dnnp_forall_id, ctx=ast.Load()),
        args=[
            ast.Name(id=transformer.dnnp_input_id, ctx=ast.Load()),
            impiles_node],
        keywords=[]
    ))

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
                ast.Name(id=transformer.dnnp_network_alias, ctx=ast.Store)],
            value=ast.Name(id='''Network("N")''', ctx=ast.Load())
        ),
        forall_node]

    print("input size: %d\noutput size: %d" %
          (transformer_1.input_size, transformer_1.output_size))

    print("\n"*2)

    code = astor.to_source(dnnp_node)

    with open("./style.style_config", 'w') as f:
        f.write(style)
    code, changed = yapf.yapflib.yapf_api.FormatCode(
        code, style_config="./style.style_config")
    os.remove("./style.style_config")
    print(code)

    filename = filename.rsplit(".")
    unwinded_dnnp_filename = filename[0]+"_step_%d" % (unwinding)+".dnnp"
    with open(unwinded_dnnp_filename, "w") as f:
        f.write(code)
    print(unwinded_dnnp_filename)
    return code, unwinded_dnnp_filename


def parse_drlp_induction(drlp: str, unwinding: int):
    filename = drlp
    try:
        with open(filename) as f:
            drlp = f.read()
    except Exception:
        filename = "tmp.drlp"

    try:
        drlp = drlp.split('\n# Exp\n')
        if len(drlp) != 2:
            raise Exception
    except Exception:
        raise DRLPParsingError("invalid DRLP format")

    drlp_p = drlp[0]
    drlp_q = drlp[1]

    unwinding += 1
    ast_root_p = ast.parse(drlp_p)
    ast_root_q = ast.parse(drlp_q)
    # astpretty.pprint(ast_root_p, show_offsets=False)

    transformer = DRLPTransformer(unwinding)

    transformer_1 = DRLPTransformer_1(unwinding)
    ast_root_p = transformer_1.visit(ast_root_p)
    ast_root_q = transformer_1.visit(ast_root_q)

    transformer_2 = DRLPTransformer_2(
        unwinding,
        transformer_1.input_size,
        transformer_1.output_size
    )
    input_size = transformer_1.input_size
    output_size = transformer_1.output_size
    ast_root_p = transformer_2.visit(ast_root_p)
    ast_root_q = transformer_2.visit(ast_root_q)

    transformer_3 = DRLPTransformer_3(unwinding)
    ast_root_p = transformer_3.visit(ast_root_p)
    ast_root_q = transformer_3.visit(ast_root_q)

    transformer_4 = DRLPTransformer_4(unwinding, input_size, output_size,False)
    ast_root_p = transformer_4.visit(ast_root_p)

    ast_root_p = ast.fix_missing_locations(ast_root_p)
    ast_root_q = ast.fix_missing_locations(ast_root_q)

    # k
    unwinding -= 1
    ast_root_q_ = ast.parse(drlp_q)
    t = DRLPTransformer(unwinding)

    t_1 = DRLPTransformer_1(unwinding)
    ast_root_q_ = t_1.visit(ast_root_q_)

    t_2 = DRLPTransformer_2(
        unwinding,
        input_size,
        output_size
    )
    ast_root_q_ = t_2.visit(ast_root_q_)

    t_3 = DRLPTransformer_3(unwinding)
    ast_root_q_ = t_3.visit(ast_root_q_)

    t_4 = DRLPTransformer_4(
        unwinding,
        input_size,
        output_size,
        True
    )
    ast_root_q_ = t_4.visit(ast_root_q_)

    ast_root_q_ = ast.fix_missing_locations(ast_root_q_)

    ast_root_p.body += ast_root_q_.body

    # Add And
    if len(ast_root_p.body) < 2:
        node_p = ast_root_p.body[0]
    else:
        node_p = ast.Call(
            func=ast.Name(id=transformer.dnnp_and_id, ctx=ast.Load()),
            args=ast_root_p.body,
            keywords=[]
        )

    if len(ast_root_q.body) < 2:
        node_q = ast_root_q.body[0]
    else:
        node_q = ast.Call(
            func=ast.Name(id=transformer.dnnp_and_id, ctx=ast.Load()),
            args=ast_root_q.body,
            keywords=[]
        )

    impiles_node = ast.Call(
        func=ast.Name(id=transformer.dnnp_impiles_id, ctx=ast.Load()),
        args=[node_p, node_q],
        keywords=[]
    )

    forall_node = ast.Expr(ast.Call(
        func=ast.Name(id=transformer.dnnp_forall_id, ctx=ast.Load()),
        args=[
            ast.Name(id=transformer.dnnp_input_id, ctx=ast.Load()),
            impiles_node],
        keywords=[]
    ))

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
                ast.Name(id=transformer.dnnp_network_alias, ctx=ast.Store)],
            value=ast.Name(id='''Network("N")''', ctx=ast.Load())
        ),
        forall_node]

    print("input size: %d\noutput size: %d" %
          (input_size, output_size))

    print("\n"*2)


    code = astor.to_source(dnnp_node)

    with open("./style.style_config", 'w') as f:
        f.write(style)
    code, changed = yapf.yapflib.yapf_api.FormatCode(
        code, style_config="./style.style_config")
    os.remove("./style.style_config")
    print(code)

    filename = filename.rsplit(".")
    unwinded_dnnp_filename = filename[0]+"_induction_step_%d" % (unwinding)+".dnnp"
    with open(unwinded_dnnp_filename, "w") as f:
        f.write(code)
    print(unwinded_dnnp_filename)
    return code, unwinded_dnnp_filename


if __name__ == "__main__":
    # parse_drlp(src, 3)
    parse_drlp_induction(src, 3)
