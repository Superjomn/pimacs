
import pyimacs.elisp as elisp
import pytest
from pyimacs.aot import aot, get_context
from pyimacs.elisp.string import String
from pyimacs.lang import ir

from pyimacs import compile

ctx = get_context()


def test_struct_basic():
    @aot
    def fn() -> str:
        Person = elisp.Struct(["name", "age"], name_hint="Person")
        jojo = Person.create(name="Jojo", age=20)
        return jojo.name

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compile(fn, builder=builder, module=module)

    print(code)

    target = '''
(defun fn ()
    (let*
        ()
        (cl-defstruct Person (list name age))
        (pyimacs-get-field (pyimacs-makestruct "Person" (list :name "Jojo" :age 20)) "name")
    )
)
    '''
    assert code.strip() == target.strip()
