import gc

from pimacs.ast.ast import Literal
from pimacs.ast.utils import *


class Example:
    def __init__(self, x):
        self.x = x


def test_WeakRef():
    x = Example(100)
    y = x  # y is a strong ref of x

    x_ref = WeakRef(x)

    del x
    gc.collect()

    assert y  # still alive

    del y  # all strong refs are gone
    gc.collect()

    assert x_ref() is None


def test_WeakSet():
    x = Example(100)
    y = Example(200)

    set = WeakSet()
    for i in range(10):
        set.add(x)
        set.add(y)

    assert len(set) == 2

    del x
    gc.collect()

    assert len(set) == 1

    del y
    gc.collect()
    assert len(set) == 0


def test_WeakSet_same_instance():
    x = Literal(value=100, loc=None)
    set = WeakSet()

    for i in range(10):
        set.add(x)
    assert len(set) == 1

    # Change the value, this should result in a different hash
    # But for WeakSet, the key is the instance itself, so it is already in the set
    x.value = 200
    for i in range(10):
        set.add(x)
    assert len(set) == 1
