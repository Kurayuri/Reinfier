from .error import *
from .import auxiliary
from ..common.Feature import Dynamic, Static
import astor
import copy
import ast
import sys


class DRLPTransformer(ast.NodeTransformer):
    INPUT_SIZE_ID = "x_size"
    OUTPUT_SIZE_ID = "y_size"
    INPUT_ID = "x"
    OUTPUT_ID = "y"
    PRECONDITION_DELIMITER = "@Pre"
    EXPECTATION_DELIMITER = "@Exp"
    DNNP_INPUT_ID = "x"
    DNNP_NETWORK_ALIAS = "N"
    DNNP_OUTPUT_ID = "%s(%s)" % (
        DNNP_NETWORK_ALIAS, DNNP_INPUT_ID)
    DEPTH_ID = "k"

    DNNP_AND_ID = "And"
    DNNP_OR_ID = "Or"
    DNNP_IMPILES_ID = "Implies"
    DNNP_FORALL_ID = "Forall"
    DNNP_SHAPE_OF_DIM_0 = "[0]"

    def __init__(self):
        self.depth = 0

        self.input_size = None
        self.output_size = None

        self.iter_vals = {}
        self.iter_ids = []

        self.variables = set()

    def calculate(self, node):
        try:
            expr = ast.Expression(body=node)
            ast.fix_missing_locations(expr)
            ans = eval(compile(expr, filename="", mode="eval"))
        except BaseException:
            ans = eval(astor.to_source(node))
        src = str(ans)
        root = ast.parse(src)
        return ans, root.body[0].value

    def flatten_list(self, lst):
        flat_list = []
        for item in lst:
            if isinstance(item, list):
                flat_list.extend(self.flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    def flatten_List(self, node):
        flat_list = []

        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.List):
                    flat_list.extend(self.flatten_List(item))
                else:
                    flat_list.append(item)
        else:
            flat_list.append(node)
        return flat_list

    def get_io_element(self, elements):
        io_element = None
        for element in elements:
            id = self.get_Name(element)
            if id == self.INPUT_ID or id == self.OUTPUT_ID:
                io_element = element
                is_input = True if id == self.INPUT_ID else False
        return io_element, is_input

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

    def get_dim_List(self, node):
        dim = 0
        if isinstance(node, ast.List):
            while dim < 10:
                dim += 1
                if self.is_dim_n_List(node, dim):
                    break
        return dim

    def get_dim_Subscrpit(self, node):
        dim = 0
        if isinstance(node, ast.Subscript):
            while dim < 10:
                dim += 1
                if self.is_dim_n_Subscrpit(node, dim):
                    break
        return dim

    def is_dim_n_Subscrpit(self, node, n: int):
        if isinstance(node, ast.Subscript):
            if sys.version_info >= (3, 8):
                if n == 1:
                    if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant):
                        return True
                    else:
                        return False
                if isinstance(node.value, ast.Subscript):
                    return self.is_dim_n_Subscrpit(node.value, n - 1)
        return False

    def get_Name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return self.get_Name(node.value)

    def is_Constant(self, node):
        if (isinstance(node, ast.Constant) or
                (isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant))):
            return True
        return False

    def visit_Name(self, node: ast.Name):
        self.variables.add(node.id)
        return node


class DRLPTransformer_Concretize(DRLPTransformer):
    '''
    Concretize all variables, list(Subscript),, BinOp
    All variables are replaced by given dict "kwargs"
    e.g.
        kwargs = {a=1,b=[2],c=3}
        a -> 1
        b[0] -> 2
        [c][0] -> 3
        a*b[0] -> 2
    '''
    SUBSCRIPT = 0b01
    BINOP = 0b10

    def __init__(self, kwargs: dict = {}, flag: int = SUBSCRIPT | BINOP):
        super().__init__()
        if kwargs is None:
            self.kwargs = {}
        self.kwargs = kwargs
        self.flag = flag

    def visit_Name(self, node: ast.Name):
        if node.id in self.kwargs.keys():
            return ast.Constant(
                value=self.kwargs[node.id]
            )
        return node

    def visit_Subscript(self, node: ast.Subscript):
        node = self.generic_visit(node)
        if self.flag & self.SUBSCRIPT:
            if isinstance(node.value, ast.Constant) or isinstance(node.value, ast.List):
                try:
                    __, node = self.calculate(node)
                except BaseException:
                    pass
        return node

    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        if self.flag & self.BINOP:
            try:
                __, node = self.calculate(node)
            except BaseException:
                pass
        return node


class DRLPTransformer_Init(DRLPTransformer):
    '''
    1. Get input_size and output_size
    2. Replace key parameters
    3. Calculate expression
    4. Unroll For
    5. Process With
    6. Replace Variables
    '''

    def __init__(self, depth, kwargs: dict = {}):
        super().__init__()
        self.depth = depth
        if kwargs is None:
            self.kwargs = {}
        self.kwargs = kwargs

    def visit_Assign(self, node: ast.Assign):
        '''Read input_size and output_size'''
        node = self.generic_visit(node)
        for target in node.targets:
            if target.id == self.INPUT_SIZE_ID:
                if self.input_size is None:
                    self.input_size = node.value.value
                    return None
                else:
                    if self.input_size != node.value.value:
                        raise DRLPParsingError("Input sizes are not equal")
            if target.id == self.OUTPUT_SIZE_ID:
                if self.output_size is None:
                    self.output_size = node.value.value
                    return None
                else:
                    if self.output_size != node.value.value:
                        raise DRLPParsingError("Output sizes are not equal")
        return node

    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        if ((isinstance(node.left, ast.List) and isinstance(node.right, ast.Constant)) or
                (isinstance(node.right, ast.List) and isinstance(node.left, ast.Constant))):
            if isinstance(node.op, ast.Mult):
                return self.calculate(node)[1]
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            return self.calculate(node)[1]
        return node

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        elements = [node.left] + node.comparators

        io_element = None
        for element in elements:
            if isinstance(element, ast.Name):
                if element.id == self.INPUT_ID or element.id == self.OUTPUT_ID:
                    io_element = element
        size = None
        if io_element is not None:
            for element in elements:
                if element is not io_element and self.is_dim_n_List(element, 2):
                    size = len(element.elts[0].elts)
        if size is not None:
            if io_element.id == self.INPUT_ID:
                if self.input_size is None:
                    self.input_size = size
                else:
                    pass
                    # if self.input_size != node.value.value:
                    #     raise DRLPParsingError("Input sizes are not equal")
            if io_element.id == self.OUTPUT_ID:
                if self.output_size is None:
                    self.output_size = size
                else:
                    pass
                    # if self.output_size != node.value.value:
                    #     raise DRLPParsingError("Output sizes are not equal")
        return node

    def visit_For(self, node: ast.For):
        '''Loop unroll'''
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
                id = self.DNNP_AND_ID
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
                            func=ast.Name(id=self.DNNP_AND_ID, ctx=ast.Load()),
                            args=arg,
                            keywords=[]
                        ))
                    else:
                        args_connect.append(arg[0])
                id = self.DNNP_OR_ID
                node = ast.Expr(
                    ast.Call(
                        func=ast.Name(id=id, ctx=ast.Load()),
                        args=args_connect,
                        keywords=[]
                    ))
        self.iter_ids.pop()

        return node

    def visit_Name(self, node: ast.Name):
        if node.id == self.DEPTH_ID:
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

    def visit_With(self, node: ast.With):
        '''With unroll'''
        node = self.generic_visit(node)
        if node.items[0].context_expr.id == "orange":
            # if node.name=="orange":
            node = ast.Expr(
                ast.Call(
                    func=ast.Name(id=self.DNNP_OR_ID, ctx=ast.Load()),
                    args=node.body,
                    keywords=[]
                ))
        elif node.items[0].context_expr.id == "range":
            # if node.name=="range":
            node = ast.Expr(
                ast.Call(
                    func=ast.Name(id=self.DNNP_AND_ID, ctx=ast.Load()),
                    args=node.body,
                    keywords=[]
                ))
        return node


class DRLPTransformer_1(DRLPTransformer):
    '''
    1. Transform Subscript
    2. Process If

    '''

    def __init__(self, depth, input_size, output_size):
        super().__init__()
        self.depth = depth
        self.input_size = input_size
        self.output_size = output_size

    def visit_Subscript(self, node: ast.Subscript):

        index = None
        try:
            if sys.version_info >= (3, 9):
                # Dim 1
                if isinstance(node.value, ast.Name):
                    node = self.generic_visit(node)
                    if node.value.id == self.INPUT_ID or node.value.id == self.OUTPUT_ID:
                        if isinstance(node.slice, ast.Constant):
                            index = node.slice.value
                            if node.value.id == self.INPUT_ID:
                                lower = index * self.input_size
                                upper = lower + self.input_size
                            if node.value.id == self.OUTPUT_ID:
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
                    if node.value.value.id == self.INPUT_ID or node.value.value.id == self.OUTPUT_ID:
                        if isinstance(node.value.slice, ast.Constant):
                            index = node.value.slice.value
                            if node.value.value.id == self.INPUT_ID:
                                lower = index * self.input_size
                            if node.value.value.id == self.OUTPUT_ID:
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
                    if node.value.id == self.INPUT_ID or node.value.id == self.OUTPUT_ID:
                        if isinstance(node.slice, ast.Slice):
                            index = node.slice.lower.value
                            if node.value.id == self.INPUT_ID:
                                lower = node.slice.lower.value * self.input_size
                                upper = node.slice.upper.value * self.input_size
                            if node.value.id == self.OUTPUT_ID:
                                lower = node.slice.lower.value * self.output_size
                                upper = node.slice.upper.value * self.output_size
                        # if isinstance(node.slice,ast.Slice):
                        #     pass
                        elif isinstance(node.slice.value, ast.Constant):
                            index = node.slice.value.value
                            if node.value.id == self.INPUT_ID:
                                lower = index * self.input_size
                                upper = lower + self.input_size
                            if node.value.id == self.OUTPUT_ID:
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
                    if node.value.value.id == self.INPUT_ID or node.value.value.id == self.OUTPUT_ID:
                        if isinstance(node.value.slice.value, ast.Constant):
                            index = node.value.slice.value.value
                            if node.value.value.id == self.INPUT_ID:
                                lower = index * self.input_size
                            if node.value.value.id == self.OUTPUT_ID:
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
        except TypeError:
            raise DRLPParsingError("Input size or output size are not specified or cannot be auto-inferred")
        return node

    def visit_List(self, node):
        node = self.generic_visit(node)
        if self.is_dim_n_List(node, 2):
            elements = [j for i in node.elts for j in i.elts]
            node.elts = elements
        return node

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        if self.calculate(node.test)[0] == True:
            node = ast.Call(
                func=ast.Name(id=self.DNNP_AND_ID, ctx=ast.Load()),
                args=node.body,
                keywords=[]
            )
        else:
            node = None
        return node


class DRLPTransformer_2(DRLPTransformer):
    '''
    Name and List Transform
    '''

    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def visit_Name(self, node: ast.Name):
        if node.id == self.INPUT_ID:
            return ast.Name(
                id=self.DNNP_INPUT_ID + self.DNNP_SHAPE_OF_DIM_0,
                # id=self.DNNP_INPUT_ID,
                ctx=ast.Load()
            )
        if node.id == self.OUTPUT_ID:
            return ast.Name(
                id=self.DNNP_OUTPUT_ID + self.DNNP_SHAPE_OF_DIM_0,
                # id=self.DNNP_OUTPUT_ID,
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


class DRLPTransformer_Induction(DRLPTransformer):
    '''
    Fix Input and Output Subscript Size Transform (k-induction)
    '''

    def __init__(self, depth, input_size, output_size, to_fix_subscript=True):
        super().__init__()
        self.depth = depth
        self.input_size = input_size
        self.output_size = output_size
        self.fix_subsript = to_fix_subscript

    def visit_Expr(self, node: ast.Expr):
        '''Remove Init Constraint'''
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Compare):
            elements = [node.value.left] + node.value.comparators
            init_element = None
            for element in elements:
                if isinstance(element, ast.Subscript):
                    if element.value.id == self.DNNP_INPUT_ID + self.DNNP_SHAPE_OF_DIM_0:
                        if isinstance(element.slice, ast.Constant):
                            if element.slice.value < self.input_size:
                                init_element = element
                        elif isinstance(element.slice, ast.Slice):
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

    def visit_Name(self, node: ast.Name):
        '''Fix Input and Output Subscript Size Transform'''
        if self.fix_subsript == True:
            if node.id == self.DNNP_INPUT_ID + self.DNNP_SHAPE_OF_DIM_0:
                return ast.Name(
                    id=node.id +
                    "[0:%d]" % (self.depth * self.input_size),
                    ctx=ast.Load()
                )
            if node.id == self.DNNP_OUTPUT_ID + self.DNNP_SHAPE_OF_DIM_0:
                return ast.Name(
                    id=node.id +
                    "[0:%d]" % (self.depth * self.output_size),
                    ctx=ast.Load()
                )
        return node

    def visit_Subscript(self, node: ast.Subscript):
        return node


class DRLPTransformer_RSC(DRLPTransformer):
    '''
    Remove Simplifiable Call
    Simplifiable Call: And/Or Call with 0/1 arg1
    '''

    def __init__(self):
        super().__init__()

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        try:
            if node.value is None:
                return None
        except BaseException:
            return None
        return node

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            return node
        if (node.func.id == self.DNNP_AND_ID or node.func.id == self.DNNP_OR_ID):
            if len(node.args) == 0:
                return None
            elif len(node.args) == 1:
                return node.args[0]
        return node


class DRLPTransformer_RIC(DRLPTransformer):
    '''
    Remove Init Constraint
    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def visit_Expr(self, node: ast.Expr):
        '''Remove Init Constraint'''
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Compare):
            elements = [node.value.left] + node.value.comparators
            init_element = None
            for element in elements:
                if isinstance(element, ast.Subscript):
                    if self.is_dim_n_Subscrpit(element, 1):
                        if element.value.id == self.INPUT_ID and element.slice.value.value == 0:
                            init_element = element
                    elif self.is_dim_n_Subscrpit(element, 2):
                        if element.value.value.id == self.INPUT_ID and element.value.slice.value.value == 0:
                            init_element = element
            is_init = True
            if init_element is not None:
                for element in elements:
                    if element is not init_element:
                        try:
                            __, elementi = self.calculate(element)
                        except BaseException:
                            elementi = element
                        if isinstance(elementi, ast.List) or isinstance(elementi, ast.Constant):
                            pass
                        else:
                            is_init = False
                if is_init == True:
                    return None
        return node


class DRLPTransformer_Boundary(DRLPTransformer):
    '''
    Get boundary of each feature
    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.depth = 1
        self.input_dynamics = {idx: Dynamic() for idx in range(input_size)}
        self.input_statics = {}
        self.output_dynamics = {idx: Dynamic() for idx in range(output_size)}
        self.output_statics = {}

    def fill(self, elems, is_lower, is_closed, idxs, is_input):
        elems = self.flatten_List(elems)
        dynamics = self.input_dynamics if is_input else self.output_dynamics
        statics = self.output_dynamics if is_input else self.output_statics
        values = []

        for elem in elems:
            if isinstance(elem, ast.Constant):
                values.append(elem.value)
            elif isinstance(elem, ast.UnaryOp) and isinstance(elem.op, ast.USub):
                values.append(-elem.operand.value)
            else:
                values.append(astor.to_source(elem))

        if is_lower:
            for i in range(len(idxs)):
                dynamics[idxs[i]].lower = values[i]
                dynamics[idxs[i]].lower_closed = is_closed
        else:
            for i in range(len(idxs)):
                dynamics[idxs[i]].upper = values[i]
                dynamics[idxs[i]].upper_closed = is_closed

    def compr(self, left, right, op, io_element, dim, is_input):
        size = self.input_size if is_input else self.output_size
        if dim == 0:
            idxs = [i for i in range(size * self.depth)]
        elif dim == 1:
            idxs = [i for i in range(size)]
        elif dim == 2:

            if isinstance(io_element.slice, ast.Constant):
                idxs = [io_element.slice.value]
            elif isinstance(io_element.slice, ast.Slice):
                idxs = [i for i in range(io_element.slice.lower, io_element.slice.upper)]

        if left is io_element:
            if isinstance(op, ast.Lt):
                self.fill(right, False, False, idxs, is_input)
            elif isinstance(op, ast.LtE):
                self.fill(right, False, True, idxs, is_input)
            elif isinstance(op, ast.Gt):
                self.fill(right, True, False, idxs, is_input)
            elif isinstance(op, ast.GtE):
                self.fill(right, True, True, idxs, is_input)

        elif right is io_element:
            if isinstance(op, ast.Lt):
                self.fill(left, True, False, idxs, is_input)
            elif isinstance(op, ast.LtE):
                self.fill(left, True, True, idxs, is_input)
            elif isinstance(op, ast.Gt):
                self.fill(left, False, False, idxs, is_input)
            elif isinstance(op, ast.GtE):
                self.fill(left, False, True, idxs, is_input)

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        elements = [node.left] + node.comparators
        ops = node.ops

        io_element, is_input = self.get_io_element(elements)
        dim = self.get_dim_Subscrpit(io_element)

        size = None
        if io_element is not None:
            for i in range(len(ops)):
                self.compr(elements[i], elements[i + 1], ops[i], io_element, dim, is_input)
        return node


class DRLPTransformer_SplitCompare(DRLPTransformer):
    '''
    Split Compare with miltiple ops to multiple Compare with single ops
    '''

    def __init__(self):
        super().__init__()

    # def visit_List(self, node):
    #     if len(node.elts) > 1:
    #         node = ast.Call(
    #             func=ast.Name(id="np.array", ctx=ast.Load()),
    #             # args=[ast.List(elts=[node],ctx=ast.Load())],
    #             args=[node],
    #             keywords=[]
    #         )
    #     else:
    #         node = node.elts[0]
    #     return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        elements = [node.left] + node.comparators
        ops = node.ops

        io_element, is_input = self.get_io_element(elements)
        dim = self.get_dim_Subscrpit(io_element)
        if len(ops) == 1:
            return node

        values = []
        for i in range(len(ops)):
            values.append(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr='all',
                        ctx=ast.Load()
                    ),
                    args=[ast.Compare(
                        left=elements[i],
                        comparators=[elements[i + 1]],
                        ops=[ops[i]]
                    )],
                    keywords=[]
                )
            )

        return ast.BoolOp(
            op=ast.And(),
            values=values
        )
