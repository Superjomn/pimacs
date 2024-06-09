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


class FuncDuplicationError(Exception):
    pass


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
        ret = {arg for _, arg in self.input_types if not arg.is_concrete}
        if not self.output_type.is_concrete:
            ret.add(self.output_type)
        return ret

    def specialize(self, template_specs: Dict[_ty.Type, _ty.Type]) -> Optional["FuncSig"]:
        template_params = self.get_template_params()
        spec_params = set(template_specs.keys())
        if not template_params.issubset(spec_params):
            return None

        ret = copy.copy(self)

        # replace inputs
        input_types = []
        for no, (name, arg) in enumerate(self.input_types):
            if arg in template_specs:
                arg = template_specs[arg]
            input_types.append((name, arg))
        ret.input_types = tuple(input_types)

        if output_type := template_specs.get(self.output_type, None):
            ret.output_type = output_type

        return ret

    def get_full_arg_list(self, call_params: List[CallParam | Expr]) -> None | Tuple[List[Tuple[str, _ty.Type]], Dict[_ty.Type, _ty.Type]]:
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
        template_specs: Dict[_ty.Type, _ty.Type] = {}

        def set_tpl_param(param: _ty.Type, actual: _ty.Type):
            assert actual.is_concrete
            if param in template_specs:
                if template_specs[param] != actual:
                    return False
            template_specs[param] = actual
            return True

        def match_argument(arg: Arg, param: CallParam) -> bool:
            if not arg.type.is_concrete:
                logger.debug("get_full_arg_list: Template parameter")
                return set_tpl_param(arg.type, param.value.get_type())
            elif not arg.type.can_accept(param.value.get_type()):
                logger.debug(f"get_full_arg_list: Type mismatch: {
                             arg.type} != {param.value.get_type()}")
                return False
            return True

        # ignore self
        # type: ignore
        if func_args and func_args[0].name == "self" and func_args[0].is_self_placeholder:
            del kwargs["self"]
            del func_args[0]
            # call_method always put the instance as the first argument
            call_params = call_params[1:]

        for no, param in enumerate(call_params):
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

    def match_call(self, params: List[CallParam | Expr]) -> bool:
        """Check if the arguments match the signature"""

        full_arg_list = self.get_full_arg_list(params)
        if full_arg_list is None:
            return False

        # TODO: process the variadic arguments
        return True


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

    @multimethod
    def lookup(self, args: tuple) -> List[Tuple[Function, FuncSig]]:
        """Find the function that matches the arguments"""
        candidates = []
        for sig, func in self.funcs.items():
            if ret := sig.get_full_arg_list(args):
                _, template_specs = ret
                concrete_sig = sig.specialize(template_specs)
                candidates.append((func, concrete_sig))
        return candidates

    @multimethod  # type: ignore
    def lookup(self, args: tuple, template_specs: Dict[_ty.Type, _ty.Type]):
        """Find the function that matches the arguments with template specs"""
        assert template_specs

        candidates = []
        for sig in self.funcs:
            if not sig.all_param_types_concrete():
                continue
            concrete_sig = sig.specialize(template_specs)
            if concrete_sig is None:
                continue
            candidate = self.funcs.get(sig, None), concrete_sig
            candidates.append(candidate)

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
