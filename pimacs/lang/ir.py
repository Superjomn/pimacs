from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import pimacs.lang.type as _type
from pimacs.lang.type import Type


@dataclass
class IrNode(ABC):
    loc: Optional["Location"] = None

    @abstractmethod
    def verify(self):
        pass


@dataclass
class Expr(IrNode, ABC):
    @abstractmethod
    def get_type(self) -> Type:
        pass


@dataclass
class Stmt(IrNode):
    pass


@dataclass
class FileName:
    __slots__ = ['filename', 'module']
    filename: str
    module: Optional[str] = None


@dataclass
class Location:
    __slots__ = ['line', 'column', 'filename']
    line: int
    column: int
    filename: FileName

    def __str__(self):
        return f"{self.filename}:{self.line}:{self.column}"


@dataclass
class VarDecl(Stmt):
    __slots__ = ['name', 'type', 'init']
    name: str
    type: Optional[Type] = None
    init: Optional[Expr] = None

    def verify(self):
        if self.type is None and self.init is None:
            raise Exception(
                f"{self.loc}:\nType or init value must be provided")
        if type is None:
            self.type = self.init.get_type()
        if self.init is not None and self.type != self.init.get_type():
            raise Exception(
                f"{self.loc}:\n var declaration type mismatch: type is {self.type} but the init is {self.init.get_type()}")


@dataclass
class Arg:
    __slots__ = ['name', 'type', 'default']
    name: str
    type: Optional[Type] = None
    default: Optional[Expr] = None

    def verify(self):
        if self.default is not None and self.type != self.default.get_type():
            raise Exception(
                f"{self.loc}:\nArg type mismatch: type is {self.type} but the default is {self.default.get_type()}")


@dataclass
class Block(Stmt):
    __slots__ = ['stmts']
    stmts: List[Stmt] = field(default_factory=list)

    def verify(self):
        for stmt in self.stmts:
            stmt.verify()


@dataclass
class FuncDecl(Stmt):
    __slots__ = ['name', 'params', 'return_type', 'body']
    name: str
    params: List[Arg]
    return_type: Optional[Type] = None
    body: Block

    def verify(self):
        for param in self.params:
            param.verify()
        self.body.verify()


@dataclass
class IfStmt(Stmt):
    __slots__ = ['condition', 'then_branch', 'else_branch']
    condition: Expr
    then_branch: Block
    else_branch: Optional[Block] = None

    def verify(self):
        self.condition.verify()
        self.then_branch.verify()
        if self.else_branch is not None:
            self.else_branch.verify()


@dataclass
class WhileStmt(Stmt):
    __slots__ = ['condition', 'body']
    condition: Expr
    body: Block

    def verify(self):
        self.condition.verify()
        self.body.verify()


@dataclass
class ForStmt(Stmt):
    __slots__ = ['init', 'condition', 'increment', 'body']
    init: VarDecl
    condition: Expr
    increment: Expr
    body: Block

    def verify(self):
        self.init.verify()
        self.condition.verify()
        self.increment.verify()
        self.body.verify()


@dataclass
class Constant(Expr):
    __slots__ = ['value']
    value: Any

    def get_type(self) -> Type:
        if isinstance(self.value, int):
            return _type.Int
        if isinstance(self.value, float):
            return _type.Float
        if isinstance(self.value, str):
            return _type.Str
        if isinstance(self.value, bool):
            return _type.Bool
        raise Exception(f"Unknown constant type: {self.value}")

    def verify(self):
        pass


@dataclass
class BinaryOp(Expr):
    __slots__ = ['left', 'op', 'right']
    left: Expr
    op: "BinaryOperator"
    right: Expr

    def get_type(self) -> Type:
        if self.left.get_type() == self.right.get_type():
            return self.left.get_type()
        # handle type conversion
        type_conversions = {
            (Type.INT, Type.FLOAT): _type.Float,
            (Type.FLOAT, Type.INT): _type.Float,
            (Type.INT, Type.BOOL): _type.Int,
            (Type.BOOL, Type.INT): _type.Int,
        }
        ret = type_conversions.get(
            (self.left.get_type(), self.right.get_type()), None)
        if ret is None:
            raise Exception(
                f"{self.loc}:\nType mismatch: {self.left.get_type()} and {self.right.get_type()}")

    def verify(self):
        self.left.verify()
        self.right.verify()


class BinaryOperator(Enum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    EQUALS = "=="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    LESS_THAN_EQUALS = "<="
    GREATER_THAN = ">"
    GREATER_THAN_EQUALS = ">="
    AND = "&&"
    OR = "||"
