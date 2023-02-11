from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.lang.core import *
from pyimacs.target.translate import Translator
from .utility import build_mod0


def test_basic():
    mod, ctx = build_mod0()
    print('mod', mod)
    trans = Translator()
    trans.run(mod)
    print(trans.mod)
