from dataclasses import dataclass
from typing import *

from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.target import elisp_ast as ast


class SymbolTable:
    def __init__(self):
        self.frames: List[Dict[ir.Value, ast.Var]] = []

    def push(self):
        self.frames.append({})

    def pop(self):
        self.frames = self.frames[:-1]

    def cur(self) -> Dict[str, ast.Var]:
        return self.frames[-1]

    def get(self, val: ir.Value) -> Optional[ast.Var]:
        ''' Recursively find the value. '''
        n = len(self.frames)
        for frame_id in range(n):
            frame = self.frames[n - 1 - frame_id]
            if val in frame:
                return frame[val]

    def add(self, val: ir.Value, var: Optional[ast.Var] = None) -> ast.Var:
        var = var if var else ast.Var("arg%d" % len(self.cur()))
        self.cur()[val] = var
        return var


class MlirToAstTranslator:
    '''
    Translate from MLIR to Elisp AST
    '''

    def __init__(self):
        self.mod = []
        self.symbol_table = SymbolTable()

    def run(self, mod: ir.Module):
        for func_name in mod.get_function_names():
            func = self.visit_Function(mod.get_function(func_name))
            self.mod.append(func)

    def get_or_set_value(self, var: ir.Value) -> ast.Var:
        ret = self.symbol_table.get(var)
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

            ret_body = []
            region = op.body()
            assert region.size() == 1, "Only 1 block is supported"
            block = region.blocks(0)
            assert len(args) == block.get_num_arguments()
            ret_body.append(self.visit_Block(block))

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
        return ast.Expression([ast.Token(self.bin_op[op.name()]), self.symbol_table.get(lhs), self.symbol_table.get(rhs)])

    def visit_Ret(self, op: ir.Operation) -> ast.Expression:
        if op.num_operands() == 0:
            return ast.Expression()
        # TODO[Superjomn]: Unify Var to Expr
        if op.num_operands() == 1:
            return self.get_or_set_value(op.get_operand(0))

    def visit_Region(self, op: ir.Region) -> ast.Expression:
        blocks = []
        for i in range(op.size()):
            block = op.blocks(i)
            blocks.append(self.visit(block))
        return ast.Expression(blocks)

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
        return self.setq(op.get_result(0), value)

    def setq(self, var: ir.Value, val: Any) -> ast.Expression:
        return ast.Expression([ast.Token("setq"), self.symbol_table.get(var), val])


@dataclass(init=True)
class SymbolTableGuard(object):
    symbol_table: SymbolTable

    def __enter__(self):
        self.symbol_table.push()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.symbol_table.pop()
