from typing import *

class MLIRContext:
    def load_pyimacs(self) -> None: ...


class Type:
    def is_integer(self, bits:int) -> bool: ...

    def is_int(self) -> bool: ...

    def is_float(self) -> bool: ...

    def is_string(self) -> bool: ...

    def is_object(self) -> bool: ...


class Value:
    def set_attr(self, name: str, attr: "Attribute"): ...

    def replace_all_uses_with(self, new_value: Value): ...

    def get_type(self) -> Type: ...


class BlockArgument:
    pass


class Region:
    def get_parent_region(self) -> "Region": ...

    def size(self) -> int: ...

    def blocks(self, idx: int) -> Block: ...

    def empty(self) -> bool: ...


class Block:
    def arg(self, idx: int) -> BlockArgument: ...

    def get_num_arguments(self) -> int: ...

    def dump(self) -> None: ...

    def move_before(self, other: "Block"):  ...

    def insert_before(self, block: "Block"): ...

    def get_parent(self, block: "Block") -> Region: ...

    def merge_block_before(self, dst: "Block") -> None: ...

    def replace_use_in_block_with(self, v: Value, newValue: Value): ...

    def operations(self) -> List[Operation]: ...


class Attribute:
    def is_int(self) -> bool: ...

    def is_float(self) -> bool: ...

    def is_string(self) -> bool: ...

    def is_bool(self) -> bool: ...

    def to_int(self) -> int: ...

    def to_float(self) -> float: ...

    def to_string(self) -> str: ...

    def to_bool(self) -> bool: ...


class IntegerAttr:
    pass


class BoolAttr:
    pass


class OpState:
    def set_attr(self, name: string, attr: Attribute) -> None: ...

    def get_num_results(self) -> int: ...

    def get_region(self, idx: int) -> Region: ...

    def dump(self) -> None: ...

    def __str__(self) -> str: ...

    def append_operand(self, val: Value): ...

    def verify(self) -> bool: ...


class OpOperand:
    def __len__(self) -> int: ...

    def result_number(self) -> int: ...

    def get(self) -> Value: ...


class Operation:
    def operands(self) -> List[Value]: ...

    def name(self) -> str: ...

    def num_operands(self) -> int: ...

    def get_operand(self, idx: int) -> OpOperand: ...

    def get_num_results(self) -> int: ...

    def get_result(self, idx: int) -> Value: ...

    def get_attr(self, name: str) -> Attribute: ...

    def to_if_op(self) -> IfOp: ...

    def to_call_op(self) -> CallOp: ...


class ForOp(OpState):
    def get_induction_var(self) -> Value: ...


class IfOp(OpState):
    def get_then_block(self) -> Block: ...

    def get_else_block(self) -> Block: ...

    def get_then_yield(self) -> "YieldOp": ...

    def get_else_yield(self) -> "YieldOp": ...

class CallOp(OpState):
    def get_callee(self) -> str: ...


class YieldOp(OpState):
    ...


class WhileOp(OpState):
    def get_before(self) -> Region: ...

    def get_after(self) -> Region: ...


class ConditionOp(OpState):
    ...


class Module(OpState):
    def dump(self) -> None: ...

    def __str__(self) -> str: ...

    def push_back(self, func: "FuncOp") -> None: ...

    def has_function(self, func_name: str) -> bool: ...

    def get_function(self, func_name: str) -> Function: ...

    def get_body(self, block_id: int) -> Block: ...

    def get_function_names(self) -> List[str]: ...

    def parse_mlir_module(
            file_name: str, context: MLIRContext) -> "ModuleOp": ...


class Function(OpState):
    def args(self, idx: int) -> Value: ...

    def add_entry_block(self) -> Block: ...

    def num_args(self) -> int: ...

    def get_name(self) -> str: ...

    def set_arg_attr(self, arg_no: int, name: str, val: int) -> None: ...

    def reset_type(self, new_type: Type): ...

    def body(self) -> Region: ...


class InsertPoint:
    ...


class Builder:
    def __init__(self, context: MLIRContext): ...

    def create_module(self) -> Module: ...

    # Create a return
    def ret(self, vals: List[Value]) -> None: ...

    def call(self, func: Function, args: List[Value]) -> Operation: ...

    def extern_call(self, ret_type: Type, callee: str,
                    args: List[Value]) -> OpState: ...

    def set_insertion_point_to_start(self, block: Block) -> None: ...

    def set_insertion_point_to_end(self, block: Block) -> None: ...

    def get_insertion_block(self) -> Block: ...

    def get_insertion_point(self) -> InsertPoint: ...

    def restore_insertion_point(self, x:InsertPoint) -> None: ...

    def get_bool_attr(self, v: bool) -> BoolAttr: ...

    def get_int32_attr(self, v: int) -> IntegerAttr: ...

    def get_int1(self, v: bool) -> Value: ...

    def get_int32(self, v: int) -> Value: ...

    def get_string(self, v: str) -> Value: ...

    def get_float32(self, v: float) -> Value: ...

    def get_null_value(self, v: bool) -> Value: ...

    # The null(nil in lisp) also need to be typed.
    def get_null_as_int(self) -> Value: ...

    def get_null_as_float(self) -> Value: ...

    def get_null_as_string(self) -> Value: ...

    def get_null_as_object(self) -> Value: ...

    def get_all_ones_value(self, type: Type) -> Value: ...

    # tupes
    def get_void_ty(self) -> Type: ...

    def get_int1_ty(self) -> Type: ...

    def get_int8_ty(self) -> Type: ...

    def get_int16_ty(self) -> Type: ...

    def get_int32_ty(self) -> Type: ...

    def get_int64_ty(self) -> Type: ...

    def get_float_ty(self) -> Type: ...

    def get_double_ty(self) -> Type: ...

    def get_string_ty(self) -> Type: ...

    def get_object_ty(self) -> Type: ...

    def get_block_ty(self) -> Type: ...

    def get_function_ty(
            self, in_types: List[Type], out_types: List[Type]) -> Type: ...

    def get_or_insert_function(self, module: Module, func_name: str,
                               func_type: Type, visibility: bool) -> Function: ...

    def create_block(self) -> Block: ...

    def create_block_with_parent(
            self, parent: Region, arg_types: List[Type]) -> Block: ...

    def new_block(self) -> Block: ...

    def create_for_op(self, lb: Value, ub: Value, step: Value,
                      init_args: List[Value]) -> ForOp: ...

    def creat_if_op(
            self, ret_types: List[Type], condition: Value, with_else: bool) -> IfOp: ...

    def create_yield_op(self, yields: List[Value]) -> YieldOp: ...

    def create_while_op(
            self, ret_types: List[Type], init_args: List[Value]) -> WhileOp: ...

    # binary ops
    def create_sle(lhs: Value, rhs: Value) -> Value: ...

    def create_slt(lhs: Value, rhs: Value) -> Value: ...

    def create_sge(lhs: Value, rhs: Value) -> Value: ...

    def create_sgt(lhs: Value, rhs: Value) -> Value: ...

    def create_ule(lhs: Value, rhs: Value) -> Value: ...

    def create_ult(lhs: Value, rhs: Value) -> Value: ...

    def create_uge(lhs: Value, rhs: Value) -> Value: ...

    def create_ugt(lhs: Value, rhs: Value) -> Value: ...

    def create_eq(lhs: Value, rhs: Value) -> Value: ...

    def create_neq(lhs: Value, rhs: Value) -> Value: ...

    def create_olt(lhs: Value, rhs: Value) -> Value: ...

    def create_ogt(lhs: Value, rhs: Value) -> Value: ...

    def create_ole(lhs: Value, rhs: Value) -> Value: ...

    def create_oge(lhs: Value, rhs: Value) -> Value: ...

    def create_oeq(lhs: Value, rhs: Value) -> Value: ...

    def create_one(lhs: Value, rhs: Value) -> Value: ...
