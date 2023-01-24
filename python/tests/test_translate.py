from pyimacs.target.translate import Translator

from .utility import build_mod0


def test_basic():
    mod, ctx = build_mod0()
    trans = Translator()
    trans.run(mod)
