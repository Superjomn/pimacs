from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.lang.core import *
from pyimacs.target.translate import MlirToAstTranslator

from .utility import build_mod0


def test_basic():
    mod, ctx = build_mod0()
    trans = MlirToAstTranslator()
    trans.run(mod)
    print(trans.mod)
