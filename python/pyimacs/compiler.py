import ast
import warnings
from typing import *

import pyimacs._C.libpyimacs.pyimacs as _pyimacs
import pyimacs.lang as pyl

from .runtime import JITFunction

ir = _pyimacs.ir
target = _pyimacs.target


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, context: ir.context, prototype, function_name: str, gscope: Dict[str, Any], module=None,
                 is_kernel=True, function_types={}):
        self.builder = ir.builder(context)
        self.module = self.builder.create_module() if module is None else module
        self.function_types = function_types
        self.prototype = prototype
        self.is_kernel = is_kernel

        self.function_name = function_name
        self.gscope = gscope
        self.lscope: Dict[str, Any] = {
            'int': int,
            'float': float,
        }
        self.local_defs: Dict[str, pyl.Value] = {}
        self.global_uses: Dict[str, pyl.Value] = {}

    def get_value(self, name: str) -> pyl.Value:
        if name in self.lscope:
            return self.lscope[name]
        if name in self.gscope:
            return self.gscope
        raise ValueError(f'{name} is not defined')

    def set_value(self, name: str, value) -> None:
        self.lscope[name] = value
        self.local_defs[name] = value

    def is_value(self, value: Any) -> bool:
        return isinstance(value, pyl.Value)

    def visit_Module(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Return(self, node) -> Any:
        ret_value = self.visit(node.value)
        if ret_value is None:
            self.builder.ret([])
            return None
        if isinstance(ret_value, tuple):
            assert NotImplementedError()
        else:
            ret = pyl.to_value(ret_value, self.builder)
            self.builder.ret([ret.handle])
            return ret.type

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        arg_names, kwarg_names = self.visit(node.args)
        # initialize defaults
        for i, default_value in enumerate(node.args.defaults):
            arg_node = node.args.args[-i - 1]
            annotation = arg_node.annotation
            name = arg_node.arg
            st_target = ast.Name(id=name, ctx=ast.Store())
            if annotation is None:
                init_node = ast.Assign(
                    targets=[st_target], value=default_value)
            else:
                init_node = ast.AnnAssign(
                    target=st_target, value=default_value, annotation=annotation)
            self.visit(init_node)

        visibility = "public" if self.is_kernel else "private"
        fn = self.builder.get_or_insert_function(self.module, self.function_name, self.prototype.to_ir(self.builder),
                                                 visibility)
        self.module.push_back(fn)
        entry = fn.add_entry_block()
        arg_values = []
        idx = 0
        for i, arg_name in enumerate(arg_names):
            arg_values.append(
                pyl.Value(fn.args(idx), self.prototype.param_types[idx]))

        insert_pt = self.builder.get_insertion_block()
        for arg_name, arg_value in zip(arg_names, arg_values):
            self.set_value(arg_name, arg_value)
        self.builder.set_insertion_point_to_start(entry)
        # visit function body
        has_ret = self.visit_compound_statement(node.body)
        # finalize function
        if not has_ret:
            self.builder.ret([])
        else:
            # update return type
            if isinstance(self.last_ret_type, tuple):
                self.prototype.ret_types = list(self.last_ret_type)
                fn.reset_type(self.prototype.to_ir(self.builder))
            else:
                self.prototype.ret_types = [self.last_ret_type]
                fn.reset_type(self.prototype.to_ir(self.builder))
        if insert_pt:
            self.builder.set_insertion_point_to_end(insert_pt)

    def visit_arguments(self, node: ast.arguments) -> Any:
        arg_names = []
        for arg in node.args:
            arg_names += [self.visit(arg)]
        kwarg_names = []
        if node.kwarg:
            kwarg_names = self.visit(node.kwarg)
        return arg_names, kwarg_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def visit_Assign(self, node):
        _names = []
        for target in node.targets:
            _names += [self.visit(target)]
        assert len(_names) == 1
        names = _names[0]
        values = self.visit(node.value)
        if not isinstance(names, tuple):
            names = [names]
        if not isinstance(values, tuple):
            values = [values]
        for name, value in zip(names, values):
            if not isinstance(value, pyl.Value):
                value = pyl.to_tensor(value, self.builder)
            self.set_value(name, value)

    def visit_Assign(self, node):
        _names = []
        for target in node.targets:
            _names += [self.visit(target)]
        assert len(_names) == 1
        names = _names[0]
        values = self.visit(node.value)
        if not isinstance(names, tuple):
            names = [names]
        if not isinstance(values, tuple):
            values = [values]
        for name, value in zip(names, values):
            # by default, constexpr are assigned into python variable
            if not isinstance(value, pyl.Value):
                value = pyl.to_value(value, self.builder)
            self.set_value(name, value)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        fn = {
            ast.Add: '__add__',
            ast.Sub: '__sub__',
            ast.Mult: '__mul__',
            ast.Div: '__truediv__',
            ast.FloorDiv: '__floordiv__',
            ast.Mod: '__mod__',
            ast.Pow: '__pow__',
            ast.LShift: '__lshift__',
            ast.RShift: '__rshift__',
            ast.BitAnd: '__and__',
            ast.BitOr: '__or__',
            ast.BitXor: '__xor__',
        }[type(node.op)]
        if self.is_value(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            return node.id
        return self.get_value(node.id)

    def visit_compound_statement(self, stmts):
        for stmt in stmts:
            self.last_ret_type = self.visit(stmt)
            if isinstance(stmt, ast.Return):
                break
        return stmts and isinstance(stmt, ast.Return)

    def visit_Pass(self, node: ast.Pass) -> Any:
        pass

    def visit(self, node):
        if node is not None:
            self.last_node = node

        with warnings.catch_warnings():
            # The ast library added visit_Constant and deprecated some other
            # methods but we can't move to that without breaking Python 3.6 and 3.7.
            warnings.simplefilter("ignore", DeprecationWarning)  # python 3.9
            warnings.simplefilter(
                "ignore", PendingDeprecationWarning)  # python 3.8
            return super().visit(node)

    def generic_visit(self, node: ast.AST) -> Any:
        if node is None:
            return None
        typename = type(node).__name__
        raise NotImplementedError(f"Unsupported node: {typename}")


def compile(fn: JITFunction, **kwargs):
    '''
    :param fn: An
    :param kwargs:
    :return:
    '''
    ctx = ir.context()
    ctx.load_pyimacs()

    configs = kwargs.get("configs", None)
    signature = kwargs["signature"]
    kwargs["configs"] = configs
    name = fn.__name__
    kwargs["signature"] = signature

    mod = translate_ast_to_lispir(fn, signature, specialization=None)
    lisp_code = translate_lispir_to_lispcode(mod)
    return lisp_code


def translate_ast_to_lispir(fn, signature, specialization):
    mod, _ = build_pyimacs_ir(fn, signature, specialization)
    return mod


def translate_lispir_to_lispcode(mod):
    return target.to_lisp_code(mod)


def str_to_ty(name):
    tys = {
        "f": pyl.Float,
        "i": pyl.Int,
        "b": pyl.Bool,
        "s": pyl.String,
        "void": pyl.Void,
    }
    assert name in tys
    return tys[name]


class CompilationError(Exception):
    def __init__(self, src, node):
        self.message = f'at {node.lineno}:{node.col_offset}:\n'
        self.message += '\n'.join(src.split('\n')[:node.lineno])
        self.message += '\n' + ' ' * node.col_offset + '^'
        self.src = src
        self.node = node
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.src, self.node))


def get_function_type_from_signature(signature: str) -> pyl.FunctionType:
    '''
    signature: "i,f -> void"
    '''
    input, output = signature.split("->")
    ins = [str_to_ty(ty) for ty in input.strip().split(",")]
    ous = [str_to_ty(ty) for ty in output.strip().split(",")]
    return pyl.FunctionType(ret_types=ous, param_types=ins)


def build_pyimacs_ir(fn, signature: str, specialization):
    '''
    signature: str, "i,f -> v"
    '''
    context = ir.context()
    context.load_pyimacs()

    # create kernel prototype
    def cst_key(i): return fn.arg_names.index(i) if isinstance(i, str) else i

    # visit kernel AST
    gscope = fn.__globals__.copy()
    function_name = fn.__name__

    prototype = get_function_type_from_signature(signature)
    generator = CodeGenerator(context, prototype, gscope=gscope, function_name=function_name,
                              is_kernel=True)
    generator.visit(fn.parse())
    # try:
    # except Exception as e:
    # node = generator.last_node
    # if node is None or isinstance(e, (NotImplementedError, CompilationError)):
    # raise e
    # raise CompilationError(fn.src, node) from e
    ret = generator.module
    # module takes ownership of the context
    ret.context = context
    return ret, generator
