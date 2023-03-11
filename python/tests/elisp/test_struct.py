
import pyimacs.elisp as elisp
import pytest
from pyimacs.elisp.string import String
from pyimacs.runtime import aot

from pyimacs import compile


def test_struct_basic():
    @aot
    def fn() -> str:
        Person = elisp.Struct(["name", "age"], name_hint="Person")
        jojo = Person.create(name="Jojo", age=20)
        return jojo.name

    code = compile(fn, signature="void -> s")
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
