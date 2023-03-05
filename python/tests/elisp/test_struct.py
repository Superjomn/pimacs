
import pyimacs.elisp as elisp
import pytest
from pyimacs.elisp.string import String
from pyimacs.runtime import jit

from pyimacs import compile


def test_struct_basic():
    @jit
    def fn() -> str:
        Person = elisp.Struct(["name", "age"], name_hint="Person")
        jojo = Person.create(name="Jojo", age=20)
        return jojo.name

    code = compile(fn, signature="void -> s")
    print(code)

    target = '''
    '''


test_struct_basic()
