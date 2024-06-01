"""
The ast module contains the AST nodes for Pimacs syntax.
"""

__all__ = ['Node',
           'Expr',
           'Stmt',
           'FileName',
           'PlainCode',
           'Location',
           'VarDecl',
           'VarRef',
           'Arg',
           'Block',
           'File',
           'Function',
           'If',
           'While',
           'For',
           'Constant',
           'BinaryOperator',
           'BinaryOp',
           'UnaryOperator',
           'UnaryOp',
           'CallParam',
           'Call',
           'Return',
           'Decorator',
           'Assign',
           'Attribute',
           'Class',
           'DocString',
           'Select',
           'Guard',
           'Unresolved',
           'UVarRef',
           'UAttr',
           'UClass',
           'UFunction',
           'LispList',
           ]

import os
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, List, Optional, Set, Tuple, Union

import pimacs.ast.type as _type
from pimacs.ast.type import Type, TypeId

from .utils import WeakSet


@dataclass
class Node(ABC):
    loc: Optional["Location"] = field(repr=False, compare=False)
    # The nodes using this node
    users: WeakSet = field(default_factory=WeakSet,
                           repr=False, init=False, hash=False)
    sema_failed: bool = field(
        default=False, init=False, hash=False, repr=False, compare=False)

    # Whether the symbol is resolved
    resolved: bool = field(default=True, repr=False, init=False, hash=False)

    def add_user(self, user: "Node"):
        if user not in self.users:
            self.users.add(user)

    def replace_all_uses_with(self, new: "Node"):
        for user in self.users:
            user.replace_child(self, new)
        for user in self.users:
            new.add_user(user)
        self.users.clear()

    @contextmanager
    def write_guard(self):
        yield
        self._refresh_users()

    @abstractmethod
    def _refresh_users(self):
        pass

    @abstractmethod
    def replace_child(self, old, new):
        ''' Replace a child with a new node. '''
        raise NotImplementedError()


@dataclass(slots=True)
class VisiableSymbol:
    '''
    @pub or not
    '''
    is_public: bool = field(default=False, repr=False, init=False)


# The Expr or Stmt are not strickly separated in the AST now.

@dataclass
class Expr(Node, ABC):
    type: Optional[Type] = field(default=None, init=False, hash=False)

    @property
    def type_determined(self) -> bool:
        return self.type not in (None, _type.Unk)

    def get_type(self) -> Type:
        return self.type or _type.Unk


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


@dataclass(slots=True, unsafe_hash=True)
class VarDecl(Stmt, VisiableSymbol):
    '''
    VarDecl is a variable declaration, such as `var a: int = 1` or `let a = 1`.
    '''
    name: str = ""
    type: Type = _type.Unk
    init: Optional[Expr] = None
    # If not mutable, it is a constant declared by `let`
    mutable: bool = True

    decorators: Tuple["Decorator", ...] = field(default_factory=tuple)

    def replace_child(self, old, new):
        if self.init == old:
            with self.write_guard():
                self.init = new

    def __repr__(self):
        return f"var {self.name}: {self.type}" + f" = {self.init}" if self.init else ""

    def __post_init__(self):
        self._refresh_users()

        if is_unk(self.type) and self.init:
            self.type = self.init.type

    def _refresh_users(self):
        if self.init:
            self.init.add_user(self)


@dataclass(slots=True, unsafe_hash=True)
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

    def replace_child(self, old, new):
        if self.target == old:
            with self.write_guard():
                self.target = new

    def __post_init__(self):
        self._refresh_users()

        if self.target:
            self.type = self.target.type

    def _refresh_users(self):
        if self.target:
            self.target.add_user(self)


@dataclass(slots=True, unsafe_hash=True)
class Arg(Expr):
    name: str = field(default="")
    type: Type = field(default_factory=lambda: _type.Unk)
    default: Optional[Expr] = None
    is_variadic: bool = False

    class Kind(Enum):
        normal = 0
        # For class methods
        cls_placeholder = 1
        self_placeholder = 2

    kind: Kind = Kind.normal

    @property
    def is_cls_placeholder(self) -> bool:
        return self.kind == Arg.Kind.cls_placeholder

    @property
    def is_self_placeholder(self) -> bool:
        return self.kind == Arg.Kind.self_placeholder

    def replace_child(self, old, new):
        with self.write_guard():
            if self.default == old:
                self.default = new

    def __post_init__(self):
        self._refresh_users()

        if is_unk(self.type) and self.default:
            self.type = self.default.type

    def _refresh_users(self):
        if self.default:
            self.default.add_user(self)


@dataclass(slots=True, unsafe_hash=True)
class Block(Stmt):
    stmts: Tuple[Stmt, ...] = field(default_factory=tuple)
    doc_string: Optional["DocString"] = None

    # If get a return value, return_type is set
    return_type: List[Type] = field(default_factory=list, hash=False)

    # TODO: Does the container node need to refresh users?
    def _refresh_users(self):
        # For the Call node, we need Block to be its user to replace a Call without a return value
        # e.g. Block(Call("some_fn")) -> Block(Call("other fn")), the Call node should have at least one user for replacing
        for stmt in self.stmts:
            stmt.add_user(self)

    def replace_child(self, old, new):
        with self.write_guard():
            stmts = list(self.stmts)
            for i, stmt in enumerate(stmts):
                if stmt == old:
                    stmts[i] = new
            self.stmts = tuple(stmts)


@dataclass(slots=True)
class File(Stmt):
    stmts: List[Stmt]

    def _refresh_users(self):
        self.users.clear()
        for stmt in self.stmts:
            stmt.add_user(self)

    def __post_init__(self):
        self._refresh_users()

    def replace_child(self, old, new):
        stmts = list(self.stmts)
        with self.write_guard():
            for i, stmt in enumerate(stmts):
                if stmt == old:
                    stmts[i] = new
        self.stmts = list(stmts)  # TODO: make stmts always a list


@dataclass(slots=True, unsafe_hash=True)
class Function(Stmt, VisiableSymbol):
    name: str
    body: Block
    args: Tuple[Arg, ...] = field(default_factory=tuple)
    return_type: Type = _type.Unk
    decorators: Tuple["Decorator", ...] = field(default_factory=tuple)

    class Kind(Enum):
        Unknown = -1
        Func = 0
        Method = 1  # class method

    class Annotation(Enum):
        Class_constructor = 0
        Class_method = 1

    kind: Kind = field(default=Kind.Func, repr=False)
    annotation: Annotation = field(
        default=Annotation.Class_constructor, repr=False, init=False, compare=False)

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

    def replace_child(self, old, new):
        with self.write_guard():
            if self.body == old:
                self.body = new
            else:
                args = list(self.args)
                for i, arg in enumerate(args):
                    if arg == old:
                        args[i] = new
                self.args = tuple(args)

            decorators = list(self.decorators)
            for i, decorator in enumerate(self.decorators):
                if decorator == old:
                    decorators[i] = new
            self.decorators = tuple(decorators)

    def _verify_decorators_unique(self):
        seen = set()
        for decorator in self.decorators:
            if decorator.action in seen:
                raise Exception(f"{self.loc}:\nDuplicate decorator: {
                                decorator.action}")
            seen.add(decorator.action)

    def _refresh_users(self):
        pass  # container node, no need to refresh users


@dataclass(slots=True, unsafe_hash=True)
class If(Stmt):
    cond: Expr
    then_branch: Block
    elif_branches: Tuple[Tuple[Expr, Block], ...] = field(
        default_factory=tuple)
    else_branch: Optional[Block] = None

    def __post_init__(self):
        self._refresh_users()

    def replace_child(self, old, new):
        with self.write_guard():
            if self.cond == old:
                self.cond = new
            if self.then_branch == old:
                self.then_branch = new

            elif_branches = list(self.elif_branches)
            for i, (cond, block) in enumerate(self.elif_branches):
                if cond == old:
                    elif_branches[i] = (new, block)
                elif block == old:
                    elif_branches[i] = (cond, new)
            self.elif_branches = tuple(elif_branches)

            if self.else_branch == old:
                self.else_branch = new

    def _refresh_users(self):
        self.cond.add_user(self)
        self.then_branch.add_user(self)
        for cond, block in self.elif_branches:
            cond.add_user(self)
            block.add_user(self)
        if self.else_branch:
            self.else_branch.add_user(self)


@dataclass(slots=True)
class While(Stmt):
    condition: Expr
    body: Block

    def _refresh_users(self):
        self.condition.add_user(self)

    def __post_init__(self):
        self._refresh_users()

    def replace_child(self, old, new):
        with self.write_guard():
            if self.condition == old:
                self.condition = new
            if self.body == old:
                self.body = new


@dataclass(slots=True)
class For(Stmt):
    init: VarDecl
    condition: Expr
    increment: Expr
    body: Block

    def __post_init__(self):
        self._refresh_users()

    def _refresh_users(self):
        self.init.add_user(self)
        self.condition.add_user(self)
        self.increment.add_user(self)

    def replace_child(self, old, new):
        with self.write_guard():
            if self.init == old:
                self.init = new
            if self.condition == old:
                self.condition = new
            if self.increment == old:
                self.increment = new
            if self.body == old:
                self.body = new


@dataclass(slots=True, unsafe_hash=True)
class Constant(Expr):
    value: Any

    def _get_type(self) -> Type:
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

    def __post_init__(self):
        self._refresh_users()

        self.type = self._get_type()

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        return super().replace_child(old, new)


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


@dataclass(slots=True, unsafe_hash=True)
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

    def _get_type(self) -> Type:
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

    def replace_child(self, old, new):
        with self.write_guard():
            if self.left == old:
                self.left = new
            if self.right == old:
                self.right = new

    def __post_init__(self):
        self._refresh_users()

        self.type = self._get_type()

    def _refresh_users(self):
        self.left.add_user(self)
        self.right.add_user(self)


class UnaryOperator(Enum):
    NEG = "-"
    NOT = "not"


@dataclass(slots=True, unsafe_hash=True)
class UnaryOp(Expr):
    op: UnaryOperator
    operand: Expr

    def _get_type(self) -> Type:
        return self.operand.get_type()

    def __post_init__(self):
        self._refresh_users()

        self.type = self._get_type()

    def _refresh_users(self):
        self.operand.add_user(self)

    def replace_child(self, old, new):
        with self.write_guard():
            if self.operand == old:
                self.operand = new


@dataclass(slots=True, unsafe_hash=True)
class CallParam(Expr):
    name: str
    value: Expr

    def __post_init__(self):
        self._refresh_users()

        self.type = self.value.get_type()

    def _refresh_users(self):
        self.value.add_user(self)

    def replace_child(self, old, new):
        with self.write_guard():
            if self.value == old:
                self.value = new


custom_types: Set[_type.Type] = set()


def get_custom_type(class_def: "Class"):
    # TODO: consider module name
    ret = _type.Type(TypeId.CUSTOMED, class_def.name)
    if ret not in custom_types:
        custom_types.add(ret)
    return ret


@dataclass(slots=True, unsafe_hash=True)
class Call(Expr):
    # ArgDecl as func for self/cls placeholder cases
    func: Union[Expr, str, Function, "UFunction"]
    args: Tuple[CallParam | Expr, ...] = field(default_factory=tuple)
    type_spec: Tuple[Type, ...] = field(default_factory=tuple)

    def _get_type(self) -> Type:
        if isinstance(self.func, Function):
            assert self.func.return_type is not None
            return self.func.return_type
        elif isinstance(self.func, Class):
            return get_custom_type(self.func)
        elif isinstance(self.func, Arg):
            return _type.Unk
        elif isinstance(self.func, UFunction):
            return _type.Unk
        elif isinstance(self.func, UAttr):
            return _type.Unk
        else:
            raise Exception(f"Unknown function type: {
                            type(self.func)}: {self.func}")

    def __post_init__(self):
        assert isinstance(self.args, tuple)
        self._refresh_users()

        self.type = self._get_type()

    def _refresh_users(self):
        if isinstance(self.func, Node):
            self.func.add_user(self)
        for arg in self.args:
            arg.add_user(self)

    def replace_child(self, old, new):
        if isinstance(self.func, Node) and self.func == old:
            with self.write_guard():
                self.func = new

        with self.write_guard():
            args = list(self.args)
            for i, arg in enumerate(args):
                if arg == old:
                    args[i] = new
            self.args = tuple(args)


@dataclass(slots=True)
class Template:
    types: Tuple[_type.Type, ...]


@dataclass(slots=True, unsafe_hash=True)
class Return(Stmt):
    value: Optional[Expr] = None

    def _refresh_users(self):
        if self.value:
            self.value.add_user(self)

    def __post_init__(self):
        self._refresh_users()

    def replace_child(self, old, new):
        with self.write_guard():
            if self.value == old:
                self.value = new


@dataclass(slots=True, unsafe_hash=True)
class Decorator(Stmt):
    action: Call | str | Template

    def _refresh_users(self):
        if isinstance(self.action, Node):
            self.action.add_user(self)

    def __post_init__(self):
        self._refresh_users()

    def replace_child(self, old, new):
        with self.write_guard():
            if self.action == old:
                self.action = new


@dataclass(slots=True, unsafe_hash=True)
class Assign(Expr):
    '''
    a = 1
    '''
    target: VarRef
    value: Expr

    def __post_init__(self):
        self._refresh_users()

        self.type = self.value.get_type()

    def _refresh_users(self):
        self.target.add_user(self)
        self.value.add_user(self)

    def replace_child(self, old, new):
        with self.write_guard():
            if self.target == old:
                self.target = new
            if self.value == old:
                self.value = new


@dataclass(slots=True, unsafe_hash=True)
class Attribute(Expr):
    value: VarRef
    attr: str

    def __post_init__(self):
        self._refresh_users()

    def _refresh_users(self):
        self.value.add_user(self)

        # TODO: set type

    def replace_child(self, old, new):
        with self.write_guard():
            if self.value == old:
                self.value = new


@dataclass(slots=True, unsafe_hash=True)
class Class(Stmt, VisiableSymbol):
    name: str
    body: Tuple[Stmt]
    decorators: Tuple["Decorator", ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return f"class {self.name}"

    def _refresh_users(self):
        self.body.add_user(self)
        for stmt in self.body:
            stmt.add_user(self)
        for decorator in self.decorators:
            decorator.add_user(self)

    def replace_child(self, old, new):
        with self.write_guard():
            body = list(self.body)
            for i, stmt in enumerate(body):
                if stmt == old:
                    body[i] = new
            self.body = tuple(body)

            decorators = list(self.decorators)
            for i, decorator in enumerate(self.decorators):
                if decorator == old:
                    decorators[i] = new
            self.decorators = tuple(decorators)


@dataclass(slots=True)
class DocString(Node):
    content: str

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        pass


@dataclass(slots=True, unsafe_hash=True)
class Select(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def _get_type(self) -> Type:
        if self.then_expr.get_type() == self.else_expr.get_type():
            return self.then_expr.get_type()
        return _type.Unk

    def __post_init__(self):
        self._refresh_users()

        self.type = self._get_type()

    def _refresh_users(self):
        self.cond.add_user(self)
        self.then_expr.add_user(self)
        self.else_expr.add_user(self)

    def replace_child(self, old, new):
        with self.write_guard():
            if self.cond == old:
                self.cond = new
            if self.then_expr == old:
                self.then_expr = new
            if self.else_expr == old:
                self.else_expr = new


@dataclass(slots=True, unsafe_hash=True)
class Guard(Stmt):
    header: Call
    body: Block

    def _refresh_users(self):
        self.header.add_user(self)
        self.body.add_user(self)

    def __post_init__(self):
        self._refresh_users()

    def replace_child(self, old, new):
        with self.write_guard():
            if self.header == old:
                self.header = new
            if self.body == old:
                self.body = new


def make_const(value: int | float | str | bool | None, loc: Location) -> Constant:
    return Constant(value=value, loc=loc)


@dataclass
class Unresolved:
    scope: Any = field(default=None, init=False, repr=False, hash=False)
    resolved: bool = field(default=False, init=False, repr=False, hash=False)

    def __post_init__(self):
        self.resolved = False


@dataclass(slots=True, unsafe_hash=True)
class UVarRef(Unresolved, Expr):
    ''' Unresolved variable reference, such as `a` in `a.b`. '''
    name: str
    target_type: Type = field(default_factory=lambda: _type.Unk)

    def get_type(self) -> Type:
        return self.target_type

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        pass


@dataclass(slots=True, unsafe_hash=True)
class UAttr(Unresolved, Expr):
    ''' Unresolved attribute, such as `a.b` in `a.b.c` '''
    value: VarRef | UVarRef
    attr: str

    def _refresh_users(self):
        self.value.add_user(self)

    def __post_init__(self):
        self._refresh_users()

        self.type = _type.Unk

    def replace_child(self, old, new):
        with self.write_guard():
            if self.value == old:
                self.value = new


@dataclass(slots=True, unsafe_hash=True)
class UClass(Unresolved, Stmt):
    ''' Unresolved class, such as `A` in `A.b`. '''
    name: str

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        pass


@dataclass(slots=True, unsafe_hash=True)
class UFunction(Unresolved, Stmt):
    ''' Unresolved function, such as `f` in `f()`. '''
    name: str
    return_type: Type = _type.Unk

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        pass


# TODO: To be deprecated
class LispList(Expr):
    def __init__(self, items: List[Expr], loc: Location):
        self.items = items
        self.loc = loc

    def get_type(self) -> Type:
        return _type.LispType

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        pass


def is_unk(type: Type) -> bool:
    return type == _type.Unk or type is None
