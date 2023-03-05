from dataclasses import dataclass, field
from typing import *

import pyimacs.lang as pyl
from pyimacs.elisp.core import make_symbol
from pyimacs.elisp.tuple import _make_tuple
from pyimacs.lang import ir
from pyimacs.lang.extension import Ext, builder, ctx, module, register_extern


class Struct(Ext):
    counter: int = 0
    names: Set[str] = set()

    def __init__(self, fields: List[str], name_hint: str = ""):
        self.fields: Set[str] = set(fields)
        self.name = name_hint if name_hint else "struct"
        if self.name in Struct.names:
            self.name += "-%d" % self.counter
            self.names.add(self.name)
            self.counter += 1

        fields = [make_symbol(x) for x in fields]
        _def_struct(make_symbol(self.name), _make_tuple(fields))

    def create(self, **kwargs) -> "Struct.Field":
        ''' Create an instance of the struct. '''
        assert kwargs
        kv = zip(kwargs.keys(), kwargs.values())
        flatten = []
        for item in kv:
            # TODO[Superjomn] fixit
            # assert k[i] in self.fields, f"Field {k[i]} not in the fields: {self.fields}"
            flatten.append(make_symbol(item[0]))
            flatten.append(item[1])
        flatten = _make_tuple(flatten)
        instance = _make_struct(self.name, flatten)
        return Struct.Field(instance, self.fields)

    @dataclass
    class Field(Ext):
        ''' A field of a struct. '''
        handle: object
        fields: Set[str]

        def __getattr__(self, field: str):
            ''' Catch the missing field access. '''
            # TODO[Superjomn] fixit
            # assert field in self.fields, f"Field {field} not in the fields: {self.fields}"
            return _get_field(self.handle, field)

# tuple and expand on arglist


class ExpandTuple:
    pass


@register_extern("cl-defstruct")
def _def_struct(name: object, fields: object) -> None: ...


@register_extern("pyimacs-makestruct")
def _make_struct(struct: str, args: object) -> object: ...


''' args is a tuple of arguments. '''


@register_extern("pyimacs-get-field")
def _get_field(o: object, field: str) -> object: ...
