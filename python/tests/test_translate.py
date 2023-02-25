from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.lang.core import *
from pyimacs.target.translate import MlirToAstTranslator
from tests.utility import build_mod0


def test_basic():
    mod, ctx = build_mod0()
    trans = MlirToAstTranslator()
    funcs = trans.run(mod)
    assert len(funcs) == 1
    func = funcs[0]
    target = '''
(defun add (arg0)
    (let*
        (arg1 arg2)
        (setq arg1 32)
        (setq arg2 (+ arg0 arg1))
        arg2
    )
)
    '''
    assert str(func).strip() == target.strip()
