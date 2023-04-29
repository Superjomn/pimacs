from dataclasses import dataclass, field
from typing import *

from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.target import elisp_ast as ast


@dataclass
class SymbolTable:
    # value's hash -> Var
    frames: List[Dict[int, ast.Var]] = field(default_factory=list)
    instant_frames: List[Dict[int, ast.Token]] = field(default_factory=list)

    def push(self):
        self.frames.append({})
        self.instant_frames.append({})

    def pop(self):
        self.frames = self.frames[:-1]
        self.instant_frames = self.instant_frames[:-1]

    def cur(self) -> Tuple[Dict[str, ast.Var], Dict[str, ast.Var]]:
        return self.frames[-1], self.instant_frames[-1]

    @property
    def cur_frame(self) -> Dict[str, ast.Var]:
        return self.frames[-1]

    @property
    def cur_instant_frame(self) -> Dict[str, ast.Token]:
        return self.instant_frames[-1]

    def get(self, val: ir.Value) -> ast.Var:
        ''' Recursively find the value. '''
        res = self.get_unsafe(val)
        assert res, f"{repr(val)} doesn't exists in SymbolTable"
        return res

    def get_unsafe(self, var: ir.Value) -> Optional[ast.Var]:
        for frame in reversed(self.frames):
            if hash(var) in frame:
                return frame[hash(var)]

    def add(self, val: ir.Value, var: Optional[ast.Var] = None, instant_val: Optional[ast.Token] = None) -> ast.Var:
        var = var if var else ast.Var("arg%d" % len(self.cur_frame))
        frame, instant_frame = self.cur()
        frame[hash(val)] = var
        if instant_val is not None:
            instant_frame[hash(val)] = instant_val
        return var

    def get_instant_value(self, val: ir.Value) -> Optional[ast.Token]:
        for frame in reversed(self.instant_frames):
            if hash(val) in frame:
                return frame[hash(val)]

    @property
    def symbols(self):
        res = set()
        for frame in self.frames:
            res.update(frame.values())
        return res


class MlirToAstTranslator:
    '''
    Translate from MLIR to Elisp AST
    '''

    def __init__(self):
        self.symbol_table = SymbolTable()

        self._constant_val = None

    def run(self, mod: ir.Module) -> List[ast.Function]:
        print("mlir:\n", mod)
        funcs = []
        for func_name in mod.get_function_names():
            func = self.visit_Function(mod.get_llvm_function(func_name))
            if func is not None:
                funcs.append(func)
        return funcs

    def get_or_set_value(self, var: ir.Value, instant_val: Optional[Any] = None) -> ast.Var:
        ret = self.symbol_table.get_unsafe(var)
        if ret is None:
            ret = self.symbol_table.add(var, instant_val=instant_val)
        return ret

    def visit_Function(self, op: ir.Function) -> ast.Function:
        if op.body().size() == 0:
            # get an Func declaration, do nothing
            return None

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
                if op_.name() == "arith.constant":
                    value = self.get_consant_value(op_)
                    var = self.get_or_set_value(res, instant_val=value)
                else:
                    var = self.get_or_set_value(res)
                let_args.append(var)

        body = []

        for op_ in op.operations():
            if op_.name() == 'scf.yield':
                continue
            op_ = self.visit_Operation(op_)
            # ignore the empty Return op
            if op_ is not None:
                body.append(op_)

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
        if op.name() == "lisp.get_null":
            return self.visit_GetNull(op)
        if op.name() == "func.return":
            return self.visit_Ret(op)
        if op.name() == "scf.if":
            return self.visit_If(op)
        # if op.name() == "std.call":
        #     return self.visit_Call(op)
        if op.name() == "llvm.call":
            return self.visit_LLVMCall(op)
        if op.name() == "lisp.make_tuple":
            return self.visit_MakeTuple(op)
        if op.name() == "lisp.make_symbol":
            return self.visit_MakeSymbol(op)
        if op.name() == "lisp.guard":
            return self.visit_Guard(op)
        if op.name() == "lisp.string_eq":
            return self.visit_StringEq(op)

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

    def visit_StringEq(self, op: ir.Operation) -> ir.Value:
        assert op.num_operands() == 2
        lhs = op.get_operand(0).get()
        rhs = op.get_operand(1).get()

        lhs = self.symbol_table.get(lhs)
        rhs = self.symbol_table.get(rhs)
        assert lhs
        assert rhs
        res = ast.Expression(ast.Symbol("string="), lhs, rhs)
        return self.setq(op.get_result(0), res)

    def visit_MakeTuple(self, op: ir.Operation) -> ast.Expression:
        assert op.num_operands() > 0
        res = ast.Expression(ast.Symbol("list"))
        for i in range(op.num_operands()):
            res.append(self.symbol_table.get(op.get_operand(i).get()))
        return self.setq(op.get_result(0), res)

    def visit_MakeSymbol(self, op: ir.Operation) -> ast.Expression:
        assert op.num_operands() == 1
        name = self.symbol_table.get_instant_value(op.get_operand(0).get())
        is_keyword = op.get_attr("is_keyword").to_bool()
        name = name if not is_keyword else f":{name}"
        return self.setq(op.get_result(0), ast.Symbol(name))

    def visit_Ret(self, op: ir.Operation) -> ast.Expression:
        if op.num_operands() == 0:
            # Ignore the empty return. It is always inserted by default incase no return statement is provided in MLIR.
            # To avoid it break the emitting of IfOp.
            return None
        # TODO[Superjomn]: Unify Var to Expr
        if op.num_operands() == 1:
            return self.symbol_table.get(op.get_operand(0).get())

    def visit_Guard(self, op: ir.Operation) -> ast.Expression:
        op: ir.GuardOp = op.to_guard_op()
        name = op.get_name()
        args = op.get_args()
        args = [self.symbol_table.get(arg) for arg in args]
        body = self.visit_Region(op.get_body())
        assert isinstance(body, ast.Expression)
        assert len(body.symbols) == 1
        body = body.symbols[0]

        return ast.Guard(name, args, body)

    def visit_Region(self, op: ir.Region) -> ast.Expression:
        blocks = []
        for i in range(op.size()):
            block = op.blocks(i)
            blocks.append(self.visit_Block(block))
        return ast.Expression(*blocks)

    def visit_Constant(self, op: ir.Operation):
        value = self.get_consant_value(op)
        value = ast.Token(value)

        return self.setq(op.get_result(0), value)

    def visit_GetNull(self, op: ir.Operation):
        value = op.get_result(0).get_type()
        if value.is_float():
            return self.setq(op.get_result(0), ast.Token(0.0))
        if value.is_int():
            return self.setq(op.get_result(0), ast.Token(0))
        if value.is_bool():
            return self.setq(op.get_result(0), ast.Token("nil"))
        if value.is_string():
            return self.setq(op.get_result(0), ast.Token(""))
        raise NotImplementedError(value)

    def get_consant_value(self, op: ir.Operation):
        value = op.get_attr("value")
        if value.is_float():
            value = value.to_float()
        elif value.is_int():
            value = value.to_int()
        elif value.is_bool():
            value = value.to_bool()
        else:
            value = value.to_string()
        return value

    def visit_If(self, op: ir.Operation):
        cond = op.get_operand(0)
        cond = self.symbol_table.get(cond.get())

        if_op = op.to_if_op()
        then_block = if_op.get_then_block()
        else_block = if_op.get_else_block()

        then_block = self.visit_Block(then_block)
        if else_block:
            else_block = self.visit_Block(else_block)

        return ast.IfElse(cond, then_block, else_block)

    # def visit_Call(self, op: ir.Operation):
    #     callee = op.to_call_op().get_callee()
    #     args = []
    #     for arg in op.operands():
    #         args.append(self.symbol_table.get(arg))
    #     call = ast.Call(ast.Symbol(callee), args)
    #     if op.get_num_results() == 0:
    #         return call
    #     return self.setq(op.get_result(0), call)

    def visit_LLVMCall(self, op: ir.Operation):
        callee = op.to_llvm_call_op().get_callee()
        is_void_call = op.get_num_results() == 0
        args = []
        for arg in op.operands():
            args.append(self.symbol_table.get(arg))
        call = ast.Call(ast.Symbol(callee), args, is_void_call=is_void_call)
        if op.get_num_results() == 0:
            return call
        return self.setq(op.get_result(0), call)

    def setq(self, var: ir.Value, val: Any) -> ast.Expression:
        return ast.Expression(ast.Symbol("setq"), self.symbol_table.get(var), val)


@dataclass(init=True)
class SymbolTableGuard(object):
    symbol_table: SymbolTable

    def __enter__(self):
        self.symbol_table.push()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.symbol_table.pop()
