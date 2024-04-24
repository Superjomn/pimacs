import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import pimacs.lang.type as _type
from pimacs.lang.type import Type, TypeId


@dataclass
class IrNode(ABC):
    loc: Optional["Location"] = field(repr=False)
    # The nodes using this node
    users: List["IrNode"] = field(default_factory=list, repr=False, init=False)

    @abstractmethod
    def verify(self):
        pass

    def add_user(self, user: "IrNode"):
        if user not in self.users:
            self.users.append(user)


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
class PlainCode:
    content: str

    def __post_init__(self):
        assert isinstance(self.content, str)


@dataclass(slots=True)
class Location:
    source: FileName | PlainCode
    line: int
    column: int

    def __post_init__(self):
        assert isinstance(self.source, FileName) or isinstance(
            self.source, PlainCode)

    def __str__(self):
        if isinstance(self.source, PlainCode):
            file_line = self.source.content.splitlines()[
                self.line - 1].rstrip()
            marker = " " * (self.column - 1) + "^"
            return f"<code>:{self.line}:{self.column}\n{file_line}\n{marker}"

        if os.path.exists(self.source.filename):
            file_line = open(self.source.filename).readlines()[
                self.line - 1].rstrip()
            marker = " " * (self.column - 1) + "^"
            return f"{self.source.filename}:{self.line}:{self.column}\n{file_line}\n{marker}"
        else:
            return f"{self.source.filename}:{self.line}:{self.column}"


@dataclass(slots=True)
class VarDecl(Stmt):
    name: str = ""
    type: Optional[Type] = None
    init: Optional[Expr] = None
    # If not mutable, it is a constant declared by `let`
    mutable: bool = True

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
class VarRef(Expr):
    ''' A placeholder for a variable.
    It is used by lexer to represent a variable. Then in the parser, it will be replaced by one with VarDecl and other meta data.
    '''
    decl: Optional[VarDecl] = None
    value: Optional[Expr] = None
    type: Optional[Type] = None
    name: str = ""

    @property
    def is_placeholder(self) -> bool:
        return bool(self.name)

    @property
    def is_ref(self) -> bool:
        return self.decl is not None

    @property
    def is_lisp(self) -> bool:
        return self.name.startswith('%')

    def get_type(self) -> Type:
        assert self.decl is not None and self.decl.type is not None
        return self.decl.type

    def verify(self):
        pass


@dataclass
class LispVarRef(VarRef):
    def __init__(self, name: str, loc: Location):
        self.loc = loc
        self.name = name[1:]
        self.type = _type.LispType

    def get_type(self) -> Type:
        return _type.LispType

    def __repr__(self):
        return f"LispVal({self.name})"


@dataclass(slots=True)
class ArgDecl(Stmt):
    name: str
    type: Optional[Type] = None
    default: Optional[Expr] = None

    class Kind(Enum):
        normal = 0
        # For class methods
        cls_placeholder = 1
        self_placeholder = 2

    kind: Kind = Kind.normal

    def verify(self):
        if self.default is not None and self.type != self.default.get_type():
            raise Exception(
                f"{self.loc}:\nArg type mismatch: type is {self.type} but the default is {self.default.get_type()}")


@dataclass(slots=True)
class Block(Stmt):
    stmts: List[Stmt] = field(default_factory=list)
    doc_string: Optional["DocString"] = None

    def verify(self):
        for stmt in self.stmts:
            stmt.verify()
        if self.doc_string:
            self.doc_string.verify()


@dataclass(slots=True)
class File(Stmt):
    stmts: List[Stmt]

    def verify(self):
        for stmt in self.stmts:
            stmt.verify()


@dataclass(slots=True)
class FuncDecl(Stmt):
    name: str
    args: List[ArgDecl]
    body: Block
    return_type: Optional[Type] = None
    decorators: List["Decorator"] = field(default_factory=list)

    class Kind(Enum):
        Unknown = -1
        Func = 0
        Method = 1  # class method

    kind: Kind = field(default=Kind.Func, repr=False)

    def verify(self):
        for arg in self.args:
            arg.verify()
        self.body.verify()

        self._verify_decorators_unique()
        for decorator in self.decorators:
            decorator.verify()

        if self.is_staticmethod and self.is_property:
            raise Exception(
                f"{self.loc}:\nA method can't be both a staticmethod and a property")

    @property
    def is_staticmethod(self) -> bool:
        "Return True if the function is a @staticmethod."
        for decorator in self.decorators:
            if decorator.action == "staticmethod":
                return True
        return False

    @property
    def is_property(self) -> bool:
        "Return True if the function is a @property."
        for decorator in self.decorators:
            if decorator.action == "property":
                return True
        return False

    @property
    def is_classmethod(self) -> bool:
        "Return True if the function is a @classmethod."
        for decorator in self.decorators:
            print(f"**decorators: {decorator.action}")
            if decorator.action == "classmethod":
                return True
        return False

    def _verify_decorators_unique(self):
        seen = set()
        for decorator in self.decorators:
            if decorator.action in seen:
                raise Exception(
                    f"{self.loc}:\nDuplicate decorator: {decorator.action}")
            seen.add(decorator.action)


@dataclass(slots=True)
class IfStmt(Stmt):
    cond: Expr
    then_branch: Block
    else_branch: Optional[Block] = None

    def verify(self):
        self.cond.verify()
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


class BinaryOperator(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "&&"
    OR = "||"


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
            (TypeId.INT, TypeId.FLOAT): _type.Float,
            (TypeId.FLOAT, TypeId.INT): _type.Float,
            (TypeId.INT, TypeId.BOOL): _type.Int,
            (TypeId.BOOL, TypeId.INT): _type.Int,
        }
        ret = type_conversions.get(
            (self.left.get_type().type_id, self.right.get_type().type_id), _type.Unk)
        if ret is None:
            raise Exception(
                f"{self.loc}:\nType mismatch: {self.left.get_type()} and {self.right.get_type()}")
        return _type.Unk

    def verify(self):
        self.left.verify()
        self.right.verify()


class UnaryOperator(Enum):
    NEG = "-"
    NOT = "not"


@dataclass(slots=True)
class UnaryOp(Expr):
    op: UnaryOperator
    value: Expr

    def get_type(self) -> Type:
        return self.value.get_type()

    def verify(self):
        self.value.verify()


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
    # ArgDecl as func for self/cls placeholder cases
    func: FuncDecl | str | ArgDecl
    args: Optional[List[CallParam | Expr]] = None

    def get_type(self) -> Type:
        assert isinstance(self.func, FuncDecl)
        assert self.func.return_type is not None
        return self.func.return_type

    def verify(self):
        self.func.verify()
        for arg in self.args:
            arg.verify()


@dataclass(slots=True)
class ReturnStmt(Stmt):
    value: Optional[Expr] = None

    def verify(self):
        if self.value is not None:
            self.value.verify()


@dataclass(slots=True)
class Decorator(Stmt):
    action: FuncCall | str

    def verify(self):
        self.action.verify()


@dataclass(slots=True)
class AssignStmt(Stmt):
    target: VarRef
    value: Expr

    def verify(self):
        self.target.verify()
        self.value.verify()


@dataclass
class ClassDef(Stmt):
    name: str
    body: List[Stmt]

    def verify(self):
        for stmt in self.body:
            stmt.verify()


@dataclass(slots=True)
class DocString(Stmt):
    content: str

    def verify(self):
        pass


@dataclass(slots=True)
class SelectExpr(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def get_type(self) -> Type:
        if self.then_expr.get_type() == self.else_expr.get_type():
            return self.then_expr.get_type()
        raise Exception(
            f"{self.loc}:\nType mismatch: {self.then_expr.get_type()} and {self.else_expr.get_type()}")

    def verify(self):
        self.cond.verify()
        if self.cond.get_type() != _type.Bool:
            raise Exception(
                f"{self.loc}:\nCondition type must be bool, but got {self.cond.get_type()}")
        if self.then_expr.get_type() != self.else_expr.get_type():
            raise Exception(
                f"{self.loc}:\nType mismatch: {self.then_expr.get_type()} and {self.else_expr.get_type()}")

        self.then_expr.verify()
        self.else_expr.verify()


@dataclass(slots=True)
class GuardStmt(Stmt):
    header: FuncCall
    body: Block

    def verify(self):
        self.header.verify()
        self.body.verify()
