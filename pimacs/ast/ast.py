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
class IrNode(ABC):
    loc: Optional["Location"] = field(repr=False, compare=False)
    # The nodes using this node
    users: List["IrNode"] = field(default_factory=list, repr=False, init=False)
    sema_failed: bool = field(default=False, repr=False, init=False)

    def add_user(self, user: "IrNode"):
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
class VarDecl(Stmt, VisiableSymbol):
    '''
    VarDecl is a variable declaration, such as `var a: int = 1` or `let a = 1`.
    '''
    name: str = ""
    type: Optional[Type] = None
    init: Optional[Expr] = None
    # If not mutable, it is a constant declared by `let`
    mutable: bool = True

    decorators: List["Decorator"] = field(default_factory=list)

    def __repr__(self):
        return f"var {self.name}: {self.type}"


@dataclass(slots=True)
class VarRef(Expr):
    """A placeholder for a variable.
    It is used by lexer to represent a variable. Then in the parser, it will be replaced by one with VarDecl and other meta data.
    """

    decl: Optional[Union[VarDecl, "ArgDecl", "UnresolvedVarRef"]] = None
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
        return self.name.startswith("%")

    def get_type(self) -> Type:
        if isinstance(self.decl, UnresolvedVarRef):
            return self.decl.target_type
        else:
            assert self.decl is not None and self.decl.type is not None
            return self.decl.type


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
        return self.kind == ArgDecl.Kind.cls_placeholder

    @property
    def is_self_placeholder(self) -> bool:
        return self.kind == ArgDecl.Kind.self_placeholder

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
class FuncDecl(Stmt, VisiableSymbol):
    name: str
    body: Block
    args: List[ArgDecl] = field(default_factory=list)
    return_type: Optional[Type] = None
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
class IfStmt(Stmt):
    cond: Expr
    then_branch: Block
    elif_branches: List[Tuple[Expr, Block]] = field(default_factory=list)
    else_branch: Optional[Block] = None


@dataclass(slots=True)
class WhileStmt(Stmt):
    condition: Expr
    body: Block


@dataclass(slots=True)
class ForStmt(Stmt):
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


def get_custom_type(class_def: "ClassDef"):
    # TODO: consider module name
    ret = _type.Type(TypeId.CUSTOMED, class_def.name)
    if ret not in custom_types:
        custom_types.add(ret)
    return ret


@dataclass(slots=True)
class FuncCall(Expr):
    # ArgDecl as func for self/cls placeholder cases
    func: Union[FuncDecl, "UnresolvedFuncDecl", str]
    args: List[CallParam | Expr] = field(default_factory=list)
    type_spec: List[Type] = field(default_factory=list)

    def get_type(self) -> Type:
        if isinstance(self.func, FuncDecl):
            assert self.func.return_type is not None
            return self.func.return_type
        elif isinstance(self.func, ClassDef):
            return get_custom_type(self.func)
        elif isinstance(self.func, ArgDecl):
            return _type.Unk
        elif isinstance(self.func, UnresolvedFuncDecl):
            return _type.Unk
        else:
            raise Exception(f"Unknown function type: {
                            type(self.func)}: {self.func}")


@dataclass(slots=True)
class Template:
    types: List[_type.Type]


@dataclass(slots=True)
class LispFuncCall(Expr):
    """Call a lisp function, such as `%format("hello")` or

    ```
    let format = %format
    format("hello)
    ```
    """

    func: LispVarRef
    args: List[CallParam | Expr] = field(default_factory=list)

    def __post_init__(self):
        assert self.args is not None

    def get_type(self) -> Type:
        return _type.LispType


class LispList(Expr):
    def __init__(self, items: List[Expr], loc: Location):
        self.items = items
        self.loc = loc

    def get_type(self) -> Type:
        return _type.LispType


@dataclass(slots=True)
class ReturnStmt(Stmt):
    value: Optional[Expr] = None


@dataclass(slots=True)
class Decorator(Stmt):
    action: FuncCall | str | Template


@dataclass(slots=True)
class Assign(Stmt):
    '''
    a = 1
    '''
    target: VarRef
    value: Expr


@dataclass(slots=True)
class UnresolvedAttr(Expr):
    value: VarRef
    attr: str

    def get_type(self) -> Type:
        return _type.Unk


@dataclass(slots=True)
class Attribute(Expr):
    value: VarRef
    attr: str


@dataclass
class ClassDef(Stmt, VisiableSymbol):
    name: str
    body: List[Stmt]
    decorators: List["Decorator"] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"class {self.name}"


@dataclass(slots=True)
class DocString(Stmt):
    content: str


@dataclass(slots=True)
class SelectExpr(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def get_type(self) -> Type:
        if self.then_expr.get_type() == self.else_expr.get_type():
            return self.then_expr.get_type()
        return _type.Unk


@dataclass(slots=True)
class GuardStmt(Stmt):
    header: FuncCall
    body: Block


@dataclass
class MemberRef(Expr):
    obj: VarRef
    member: VarRef

    def get_type(self) -> Type:
        if isinstance(self.member, VarRef):
            return self.member.get_type()
        return _type.Unk


@dataclass(slots=True)
class UnresolvedFuncDecl(Stmt):
    name: str
    return_type: Optional[Type] = None


@dataclass(slots=True)
class UnresolvedVarRef(Expr):
    name: str
    target_type: Type = field(default_factory=lambda: _type.Unk)

    def get_type(self) -> Type:
        return self.target_type


@dataclass(slots=True)
class UnresolvedClassDef(Stmt):
    name: str


def make_const(value: int | float | str | bool | None, loc: Location) -> Constant:
    return Constant(value=value, loc=loc)
