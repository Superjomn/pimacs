"""
The ast module contains the AST nodes for Pimacs syntax.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, List, Optional, Set, Tuple, Union

import pimacs.ast.type as _type
from pimacs.ast.type import Type, TypeId


@dataclass
class Node(ABC):
    loc: Optional["Location"] = field(repr=False, compare=False)
    # The nodes using this node
    users: List["Node"] = field(default_factory=list, repr=False, init=False)
    sema_failed: bool = field(default=False, repr=False, init=False)

    def add_user(self, user: "Node"):
        if user not in self.users:
            self.users.append(user)


@dataclass(slots=True)
class VisiableSymbol:
    '''
    @pub or not
    '''
    is_public: bool = field(default=False, repr=False, init=False)


# The Expr or Stmt are not strickly separated in the AST now.

@dataclass
class Expr(Node, ABC):
    @abstractmethod
    def get_type(self) -> Type:
        pass


@dataclass
class Stmt(Node):
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
class VarDecl(Stmt, VisiableSymbol):
    '''
    VarDecl is a variable declaration, such as `var a: int = 1` or `let a = 1`.
    '''
    name: str = ""
    type: Type = _type.Unk
    init: Optional[Expr] = None
    # If not mutable, it is a constant declared by `let`
    mutable: bool = True

    decorators: List["Decorator"] = field(default_factory=list)

    def __repr__(self):
        return f"var {self.name}: {self.type}" + f" = {self.init}" if self.init else ""


@dataclass(slots=True)
class VarRef(Expr):
    """A placeholder for a variable.
    """

    target: Optional[Union[VarDecl, "Arg"]] = None
    type: Optional[Type] = None
    name: str = ""

    @property
    def is_placeholder(self) -> bool:
        return bool(self.name)

    @property
    def is_ref(self) -> bool:
        return self.target is not None

    @property
    def is_lisp(self) -> bool:
        return self.name.startswith("%")

    def get_type(self) -> Type:
        if isinstance(self.target, UVarRef):
            return self.target.target_type
        else:
            assert self.target is not None and self.target.type is not None
            return self.target.type


@dataclass(slots=True)
class Arg(Stmt):
    name: str
    type: Type = field(default_factory=lambda: _type.Unk)
    default: Optional[Expr] = None
    is_variadic: bool = False

    class Kind(Enum):
        normal = 0
        # For class methods
        cls_placeholder = 1
        self_placeholder = 2

    @property
    def is_cls_placeholder(self) -> bool:
        return self.kind == Arg.Kind.cls_placeholder

    @property
    def is_self_placeholder(self) -> bool:
        return self.kind == Arg.Kind.self_placeholder

    kind: Kind = Kind.normal


@dataclass(slots=True)
class Block(Stmt):
    stmts: List[Stmt] = field(default_factory=list)
    doc_string: Optional["DocString"] = None
    # If get a return value, return_type is set
    return_type: List[Type] = field(default_factory=list)


@dataclass(slots=True)
class File(Stmt):
    stmts: List[Stmt]


@dataclass(slots=True)
class Function(Stmt, VisiableSymbol):
    name: str
    body: Block
    args: List[Arg] = field(default_factory=list)
    return_type: Type = _type.Unk
    decorators: List["Decorator"] = field(default_factory=list)

    class Kind(Enum):
        Unknown = -1
        Func = 0
        Method = 1  # class method

    kind: Kind = field(default=Kind.Func, repr=False)

    def __post_init__(self):
        assert self.args is not None

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
            if decorator.action == "classmethod":
                return True
        return False

    def _verify_decorators_unique(self):
        seen = set()
        for decorator in self.decorators:
            if decorator.action in seen:
                raise Exception(f"{self.loc}:\nDuplicate decorator: {
                                decorator.action}")
            seen.add(decorator.action)


@dataclass(slots=True)
class If(Stmt):
    cond: Expr
    then_branch: Block
    elif_branches: List[Tuple[Expr, Block]] = field(default_factory=list)
    else_branch: Optional[Block] = None


@dataclass(slots=True)
class While(Stmt):
    condition: Expr
    body: Block


@dataclass(slots=True)
class For(Stmt):
    init: VarDecl
    condition: Expr
    increment: Expr
    body: Block


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
        if self.value is None:
            return _type.Nil
        raise Exception(f"Unknown constant type: {self.value}")


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

    type: Optional[_type.Type] = field(init=False, default=None)

    # handle type conversion
    type_conversions: ClassVar[Any] = {
        (TypeId.INT, TypeId.FLOAT): _type.Float,
        (TypeId.FLOAT, TypeId.INT): _type.Float,
        (TypeId.INT, TypeId.BOOL): _type.Int,
        (TypeId.BOOL, TypeId.INT): _type.Int,
    }

    def get_type(self) -> Type:
        if self.type:
            return self.type

        if self.left.get_type() == self.right.get_type():
            return self.left.get_type()

        ret = BinaryOp.type_conversions.get(
            (self.left.get_type().type_id, self.right.get_type().type_id), _type.Unk
        )
        if ret is None:
            raise Exception(
                f"{self.loc}:\nType mismatch: {self.left.get_type()} and {
                    self.right.get_type()}"
            )
        return _type.Unk


class UnaryOperator(Enum):
    NEG = "-"
    NOT = "not"


@dataclass(slots=True)
class UnaryOp(Expr):
    op: UnaryOperator
    value: Expr

    def get_type(self) -> Type:
        return self.value.get_type()


@dataclass(slots=True)
class CallParam(Expr):
    name: str
    value: Expr

    def get_type(self) -> Type:
        return self.value.get_type()


custom_types: Set[_type.Type] = set()


def get_custom_type(class_def: "Class"):
    # TODO: consider module name
    ret = _type.Type(TypeId.CUSTOMED, class_def.name)
    if ret not in custom_types:
        custom_types.add(ret)
    return ret


@dataclass(slots=True)
class Call(Expr):
    # ArgDecl as func for self/cls placeholder cases
    func: Union[Function, "UFunction", str]
    args: List[CallParam | Expr] = field(default_factory=list)
    type_spec: List[Type] = field(default_factory=list)

    def get_type(self) -> Type:
        if isinstance(self.func, Function):
            assert self.func.return_type is not None
            return self.func.return_type
        elif isinstance(self.func, Class):
            return get_custom_type(self.func)
        elif isinstance(self.func, Arg):
            return _type.Unk
        elif isinstance(self.func, UFunction):
            return _type.Unk
        else:
            raise Exception(f"Unknown function type: {
                            type(self.func)}: {self.func}")


@dataclass(slots=True)
class Template:
    types: List[_type.Type]


@dataclass(slots=True)
class Return(Stmt):
    value: Optional[Expr] = None


@dataclass(slots=True)
class Decorator(Stmt):
    action: Call | str | Template


@dataclass(slots=True)
class Assign(Stmt):
    '''
    a = 1
    '''
    target: VarRef
    value: Expr


@dataclass(slots=True)
class Attribute(Expr):
    value: VarRef
    attr: str


@dataclass
class Class(Stmt, VisiableSymbol):
    name: str
    body: List[Stmt]
    decorators: List["Decorator"] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"class {self.name}"


@dataclass(slots=True)
class DocString(Stmt):
    content: str


@dataclass(slots=True)
class Select(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def get_type(self) -> Type:
        if self.then_expr.get_type() == self.else_expr.get_type():
            return self.then_expr.get_type()
        return _type.Unk


@dataclass(slots=True)
class Guard(Stmt):
    header: Call
    body: Block


@dataclass
class MemberRef(Expr):
    obj: VarRef
    member: VarRef

    def get_type(self) -> Type:
        if isinstance(self.member, VarRef):
            return self.member.get_type()
        return _type.Unk


def make_const(value: int | float | str | bool | None, loc: Location) -> Constant:
    return Constant(value=value, loc=loc)


@dataclass(slots=True)
class UVarRef(Expr):
    ''' Unresolved variable reference, such as `a` in `a.b`. '''
    name: str
    target_type: Type = field(default_factory=lambda: _type.Unk)

    def get_type(self) -> Type:
        return self.target_type


@dataclass(slots=True)
class UAttr(Expr):
    ''' Unresolved attribute, such as `a.b` in `a.b.c` '''
    value: VarRef | UVarRef
    attr: str

    def get_type(self) -> Type:
        return _type.Unk


@dataclass(slots=True)
class UClass(Stmt):
    ''' Unresolved class, such as `A` in `A.b`. '''
    name: str


@dataclass(slots=True)
class UFunction(Stmt):
    ''' Unresolved function, such as `f` in `f()`. '''
    name: str
    return_type: Type = _type.Unk


class LispList(Expr):
    def __init__(self, items: List[Expr], loc: Location):
        self.items = items
        self.loc = loc

    def get_type(self) -> Type:
        return _type.LispType
