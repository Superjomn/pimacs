from dataclasses import dataclass, field
from typing import *

from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.target import elisp_ast as ast


@dataclass
class SymbolTable:
    # value's hash -> Var
    frames: List[Dict[int, ast.Var]] = field(default_factory=list)

    def push(self):
        self.frames.append({})

    def pop(self):
        self.frames = self.frames[:-1]

    def cur(self) -> Dict[str, ast.Var]:
        return self.frames[-1]

    def get(self, val: ir.Value) -> ast.Var:
        ''' Recursively find the value. '''
        res = self.get_unsafe(val)
        assert res, f"{repr(val)} doesn't exists in SymbolTable"
        return res

    def get_unsafe(self, var: ir.Value) -> Optional[ast.Var]:
        for frame in reversed(self.frames):
            if hash(var) in frame:
                return frame[hash(var)]

    def add(self, val: ir.Value, var: Optional[ast.Var] = None) -> ast.Var:
        var = var if var else ast.Var("arg%d" % len(self.cur()))
        self.cur()[hash(val)] = var
        return var

    @property
    def symbols(self):
        res = set()
        for frame in self.frames:
            res.update([hash(key) for key in frame.keys()])
        return res


class MlirToAstTranslator:
    '''
    Translate from MLIR to Elisp AST
    '''

    def __init__(self):
        self.symbol_table = SymbolTable()

    def run(self, mod: ir.Module) -> List[ast.Function]:
        funcs = []
        for func_name in mod.get_function_names():
            func = self.visit_Function(mod.get_function(func_name))
            funcs.append(func)
        return funcs

    def get_or_set_value(self, var: ir.Value) -> ast.Var:
        ret = self.symbol_table.get_unsafe(var)
        if ret is None:
            ret = self.symbol_table.add(var)
        return ret

    def visit_Function(self, op: ir.Function) -> ast.Function:
        with SymbolTableGuard(self.symbol_table) as s:
            name = op.get_name()
            arg_vars = []
            args = [op.args(i) for i in range(op.num_args())]
            for no, arg in enumerate(args):
                arg_vars.append(self.symbol_table.add(arg))

            region = op.body()
            assert region.size() == 1, "Only 1 block is supported"
            block = region.blocks(0)
            assert len(args) == block.get_num_arguments()
            ret_body = self.visit_Block(block)

            func = ast.Function(name=name, args=arg_vars, body=ret_body)
            return func

    def visit_Block(self, op: ir.Block) -> ast.LetExpr:
        # the arguments of the block should be pushed into the symbol table already.
        # TODO[Superjomn]: add a pass to inline some expressions

        let_args = []

        for op_ in op.operations():
            for idx in range(op_.get_num_results()):
                res = op_.get_result(idx)
                var = self.get_or_set_value(res)
                let_args.append(var)

        body = []

        for op_ in op.operations():
            body.append(self.visit_Operation(op_))

        let = ast.LetExpr(vars=let_args, body=body)
        return let

    def visit_Operation(self, op: ir.Operation) -> ast.Expression:
        if op.name() in ("arith.addi", "arith.addf",
                         "arith.subi", "arith.subf",
                         "arith.muli", "arith.mulf",
                         "arith.divs", "arith.divf",
                         ):
            return self.visit_binary(op)
        if op.name() == "arith.constant":
            return self.visit_Constant(op)
        if op.name() == "std.return":
            return self.visit_Ret(op)

        raise NotImplementedError(op.name())

    bin_op = {
        "arith.addi": "+",
        "arith.addf": "+",
        "arith.subi": "-", "arith.subf": "-",
        "arith.muli": "*", "arith.mulf": "*",
        "arith.divs": "/", "arith.divf": "/",
    }

    def visit_binary(self, op: ir.Operation) -> ast.Expression:
        assert op.num_operands() == 2
        lhs = op.get_operand(0).get()
        rhs = op.get_operand(1).get()

        lhs = self.symbol_table.get(lhs)
        rhs = self.symbol_table.get(rhs)
        assert lhs
        assert rhs
        res = ast.Expression(ast.Symbol(self.bin_op[op.name()]), lhs, rhs)
        return self.setq(op.get_result(0), res)

    def visit_Ret(self, op: ir.Operation) -> ast.Expression:
        if op.num_operands() == 0:
            return ast.Expression()
        # TODO[Superjomn]: Unify Var to Expr
        if op.num_operands() == 1:
            return self.symbol_table.get(op.get_operand(0).get())

    def visit_Region(self, op: ir.Region) -> ast.Expression:
        blocks = []
        for i in range(op.size()):
            block = op.blocks(i)
            blocks.append(self.visit(block))
        return ast.Expression(*blocks)

    def visit_Constant(self, op: ir.Operation):
        value = op.get_attr("value")
        if value.is_float():
            value = value.to_float()
        elif value.is_int():
            value = value.to_int()
        elif value.is_bool():
            value = value.to_bool()
        else:
            value = value.to_string()

        value = ast.Token(value)
        return self.setq(op.get_result(0), value)

    def setq(self, var: ir.Value, val: Any) -> ast.Expression:
        return ast.Expression(ast.Symbol("setq"), self.symbol_table.get(var), val)


@dataclass(init=True)
class SymbolTableGuard(object):
    symbol_table: SymbolTable

    def __enter__(self):
        self.symbol_table.push()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.symbol_table.pop()
