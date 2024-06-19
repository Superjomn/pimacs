import copy
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from multimethod import multimethod

import pimacs.ast.type as _ty
from pimacs.ast.ast import Arg, Call, CallParam, Expr, Function
from pimacs.ast.type import Type as _type
from pimacs.logger import logger

from .utils import (FuncSymbol, Scoped, ScopeKind, Symbol, SymbolItem,
                    print_colored)


# TODO: Make FuncDuplicationError accept the candiates, and make FuncOverloads.lookup return a single candidate
class FuncDuplicationError(Exception):
    pass


template_spec_t = Dict[_ty.Type, _ty.Type]  # type: ignore


@dataclass(slots=True, unsafe_hash=True)
class FuncSig:
    """Function signature"""

    symbol: FuncSymbol
    input_types: Tuple[Tuple[str, _ty.Type], ...]
    output_type: _ty.Type

    func: weakref.ref[Function] = field(hash=False)

    @classmethod
    def create(cls, func: "Function"):
        return_type = func.return_type if func.return_type else _ty.Void
        symbol = FuncSymbol(func.name)
        # TODO: Support the context
        return FuncSig(
            symbol=symbol,
            input_types=tuple((arg.name, arg.type) for arg in func.args),
            output_type=return_type,
            func=weakref.ref(func),
        )

    def all_param_types_concrete(self) -> bool:
        return not self.get_template_params()

    def get_template_params(self) -> Set[_ty.Type]:
        ret = set()

        def collect_type(arg_type):
            if not arg_type.is_concrete:
                if isinstance(arg_type, _ty.PlaceholderType):
                    ret.add(arg_type)
                elif isinstance(arg_type, _ty.CompositeType):
                    ret.update(arg_type.collect_placeholders())
                else:
                    raise NotImplementedError()

        for _, arg_type in self.input_types:
            collect_type(arg_type)
        collect_type(self.output_type)

        return ret

    def specialize(self, template_specs: Dict[_ty.Type, _ty.Type]) -> Optional["FuncSig"]:
        '''
        Specialize a Generic function signature with the template specs.
        Such as foo[T0, T1](x: T0, y: T1) -> T0 with mapping {T0: Int, T1: Float} will got
            foo[Int, Float](x: Int, y: Float) -> Int
        '''
        template_params = self.get_template_params()
        spec_params = set(template_specs.keys())
        if not template_params.issubset(spec_params):
            logger.debug(f"FuncSig.specialize failed: {
                         template_params} vs {template_specs}")
            return None

        ret = copy.copy(self)

        def specialize_type(arg_type) -> _ty.Type:
            if isinstance(arg_type, _ty.GenericType):
                if (new_arg_type := as_placeholder_type(arg_type)) and new_arg_type in template_specs:
                    return template_specs[new_arg_type]
                return arg_type
            elif arg_type in template_specs:
                return template_specs[arg_type]
            elif isinstance(arg_type, _ty.CompositeType):
                return arg_type.replace_with(template_specs)
            elif isinstance(arg_type, _ty.PlaceholderType):
                raise NotImplementedError(
                    f"Remaining placeholder type: {arg_type} in {self}")
            else:
                return arg_type

        # replace inputs
        input_types = []
        for no, (name, arg) in enumerate(self.input_types):
            arg = specialize_type(arg)
            input_types.append((name, arg))
        ret.input_types = tuple(input_types)

        ret.output_type = specialize_type(self.output_type)

        return ret

    def get_full_arg_list(self, call_params: List[CallParam | Expr],
                          template_param_spec: Optional[template_spec_t | List[_ty.Type]] = None) \
            -> None | Tuple[List[Tuple[str, _ty.Type]], template_spec_t]:
        ''' fill in the default values and get a full list of arguments
        Returns:
            None if the arguments do not match the signature
        '''
        # ref is out-of-date
        if self.func() is None:
            logger.debug(f"get_full_arg_list: Function is out-of-date")
            return None

        func_args = [arg for arg in self.func().args]  # type: ignore
        kwargs = {arg.name: arg for arg in func_args}   # type: ignore
        # mapping placeholder type to spec type
        # e.g. T0 => Int, T1 => Float
        template_specs: Dict[_ty.Type, _ty.Type] = {}

        if template_param_spec and isinstance(template_param_spec, (list, tuple)):
            if not self.func().template_params or len(template_param_spec) != len(self.func().template_params):  # type: ignore
                return None
            for param in template_param_spec:
                assert param.is_concrete, f"template param {
                    param} is not concrete"
            template_param_spec = {arg: spec for arg, spec in zip(
                self.func().template_params, template_param_spec)}  # type: ignore

        if template_param_spec:
            assert isinstance(template_param_spec, dict) or template_param_spec is None, f"template_param_spec: {
                template_param_spec}"
            template_specs.update(template_param_spec)

        logger.debug(f"get_full_arg_list: sig of {
                     self} get template_param_spec: {template_param_spec}")

        def set_tpl_param(param: _ty.Type, actual: _ty.Type):
            assert actual.is_concrete
            if param in template_specs:
                if template_specs[param] != actual:
                    return False
            template_specs[param] = actual
            return True

        def match_argument(arg: Arg, param: CallParam) -> bool:
            arg_type = as_placeholder_type(
                template_specs.get(arg.type, arg.type))
            logger.debug(f"get_full_arg_list: {
                         arg.name} -> {arg_type} of {type(arg_type)} vs {param.value.get_type()}")
            if not arg_type.is_concrete:
                logger.debug(f"get_full_arg_list: update template parameter: {
                             arg_type} to {param.value.get_type()}")
                return set_tpl_param(arg_type, param.value.get_type())
            elif not arg_type.can_accept(param.value.get_type()):
                logger.debug(f"get_full_arg_list: Type mismatch: {
                             arg_type} != {param.value.get_type()}")
                return False
            return True

        # ignore self
        # type: ignore
        if func_args and func_args[0].name == "self" and func_args[0].is_self_placeholder:
            del kwargs["self"]
            del func_args[0]
            # call_method always put the instance as the first argument
            call_params = call_params[1:]

        for no, param in enumerate(call_params):  # type: ignore
            if no >= len(func_args):
                return None  # too many arguments
            if isinstance(param, CallParam):
                if not param.name:
                    arg = func_args[no]
                else:
                    if param.name not in kwargs:
                        logger.debug(
                            f"get_full_arg_list: Unknown argument: {param.name}")
                    arg = kwargs[param.name]

            elif isinstance(param, Expr):
                arg: Arg = func_args[no]   # type: ignore

            logger.debug(f"get_full_arg_list: {arg.name} -> {param}")

            if match_argument(arg, param):  # type: ignore
                arg_name = func_args[no].name
                kwargs[arg_name] = param.value  # type: ignore
            else:
                return None

        for name, value in kwargs.items():
            # check if the required argument is missing
            if isinstance(value, Arg):
                if value.default is None:
                    logger.debug(f"Missing argument: {name}")
                    return None
                else:
                    kwargs[name] = value.default

        # check if all the arguments are filled
        if any(isinstance(value, Arg) for value in kwargs.values()):
            logger.debug(f"Missing argument")
            return None

        ret = [(arg.name, kwargs[arg.name])
               for arg in func_args]  # type: ignore

        return ret, template_specs  # type: ignore

    # TODO: replace the List[CallParam] with FuncCall to support overloading based on return_type

    def match_call(self, params: List[CallParam | Expr], template_specs: Optional[template_spec_t] = None) -> bool:
        ''' Check if the arguments match the signature with template specs '''
        return self.get_full_arg_list(params, template_specs)  # type: ignore


@dataclass(slots=True)
class FuncOverloads:
    """FuncOverloads represents the functions with the same name but different signatures.
    It could be records in the symbol table, and it could be scoped.
    """

    symbol: Symbol
    # funcs with the same name
    funcs: Dict[FuncSig, Function] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.symbol.name

    # TODO: Make it return an optional candidate for simplicity, if multiple, raise an error
    @multimethod
    def lookup(self, args: tuple, template_spec: Optional[template_spec_t] = None) -> List[Tuple[Function, FuncSig]]:
        """ Find the function that matches the arguments.
        Args:
            args: the arguments of the function call
            template_spec: the template specs of the function call

        Returns:
            A list of candidates, each candidate is a tuple of the function and the concrete signature.
        """
        candidates = []
        for sig, func in self.funcs.items():
            if ret := sig.get_full_arg_list(args, template_spec):
                _, template_specs = ret
                concrete_sig = sig.specialize(template_specs)
                candidates.append((func, concrete_sig))
        return candidates

    @multimethod  # type: ignore
    def lookup(self, sig: FuncSig) -> Optional[Function]:
        """Find the function that matches the signature"""
        return self.funcs.get(sig, None)

    def insert(self, func: Function):
        sig = FuncSig.create(func)
        if sig in self.funcs:
            raise FuncDuplicationError(f"Function {func.name} already exists")
        self.funcs[sig] = func

    def __iter__(self):
        return iter(self.funcs.values())

    def __add__(self, other: "FuncOverloads") -> "FuncOverloads":
        assert self.symbol == other.symbol
        assert self.funcs.keys().isdisjoint(other.funcs.keys())
        new_funcs = self.funcs.copy()
        new_funcs.update(other.funcs)
        return FuncOverloads(self.symbol, new_funcs)

    def __getitem__(self, sig: FuncSig) -> Function:
        return self.funcs[sig]

    def __contains__(self, sig: FuncSig) -> bool:
        return sig in self.funcs

    def __len__(self) -> int:
        return len(self.funcs)

    def __iter__(self):  # type: ignore
        return iter(self.funcs.values())

    def __repr__(self):
        return f"FuncOverloads[{self.symbol} x {len(self.funcs)}]"


def as_placeholder_type(type: _ty.Type) -> _ty.Type:
    # A trick to get the placeholder type with the same name. This is necessary since the lark parser get
    # GenericType[T] rather than PlaceholderType[T]
    # TODO: remove this trick, remove the buggy replace_types method from ast.
    if isinstance(type, _ty.PlaceholderType):
        return type
    elif isinstance(type, _ty.GenericType):
        return _ty.PlaceholderType(type.name)
    else:
        return type
