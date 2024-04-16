from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import pimacs.lang.type as _type
from pimacs.lang.type import Type


@dataclass
class IrNode(ABC):
    loc: Optional["Location"]

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


@dataclass(slots=True)
class FileName:
    filename: str
    module: Optional[str] = None


@dataclass(slots=True)
class Location:
    filename: FileName
    line: int
    column: int

    def __str__(self):
        return f"{self.filename}:{self.line}:{self.column}"


@dataclass(slots=True)
class VarDecl(Stmt):
    name: str = ""
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


@dataclass(slots=True)
class ArgDecl(Stmt):
    name: str
    type: Optional[Type] = None
    default: Optional[Expr] = None

    def verify(self):
        if self.default is not None and self.type != self.default.get_type():
            raise Exception(
                f"{self.loc}:\nArg type mismatch: type is {self.type} but the default is {self.default.get_type()}")


@dataclass(slots=True)
class Block(Stmt):
    stmts: List[Stmt] = field(default_factory=list)

    def verify(self):
        for stmt in self.stmts:
            stmt.verify()


@dataclass(slots=True)
class FuncDecl(Stmt):
    name: str
    args: List[ArgDecl]
    body: Block
    return_type: Optional[Type] = None

    def verify(self):
        for arg in self.args:
            arg.verify()
        self.body.verify()


@dataclass(slots=True)
class IfStmt(Stmt):
    condition: Expr
    then_branch: Block
    else_branch: Optional[Block] = None

    def verify(self):
        self.condition.verify()
        self.then_branch.verify()
        if self.else_branch is not None:
            self.else_branch.verify()


@dataclass(slots=True)
class WhileStmt(Stmt):
    condition: Expr
    body: Block

    def verify(self):
        self.condition.verify()
        self.body.verify()


@dataclass(slots=True)
class ForStmt(Stmt):
    init: VarDecl
    condition: Expr
    increment: Expr
    body: Block

    def verify(self):
        self.init.verify()
        self.condition.verify()
        self.increment.verify()
        self.body.verify()


@dataclass(slots=True)
class Constant(Expr):
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


@dataclass(slots=True)
class BinaryOp(Expr):
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


@dataclass(slots=True)
class CallParam(Expr):
    name: str
    value: Expr

    def verify(self):
        self.value.verify()

    def get_type(self) -> Type:
        return self.value.get_type()


@dataclass(slots=True)
class FuncCall(Expr):
    func: FuncDecl | str
    args: Optional[List[CallParam | Expr]] = None

    def get_type(self) -> Type:
        return self.func.return_type

    def verify(self):
        self.func.verify()
        for arg in self.args:
            arg.verify()


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
