from pimacs.lang.type import *


def test_Type_str():
    assert str(Int) == "Int"
    assert str(Float) == "Float"
    assert str(Bool) == "Bool"
    assert str(Str) == "Str"
    assert str(Nil) == "nil"
    assert str(LispType) == "Lisp"
    assert str(Unk) == "Unk"
