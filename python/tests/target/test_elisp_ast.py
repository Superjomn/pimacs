import pytest
from pyimacs.target.elisp_ast import *


def test_let():
    a = Var("a", default=0)
    assert str(a) == "a"
    b = Var("b", default=1)

    add = Expression("+", a, b)
    assert str(add) == "(+ a b)", str(add)

    let = LetExpr(vars=[a, b], body=[add])
    target = \
        '''
(let*
    ((a 0) (b 1))
    (+ a b)
)
    '''
    assert str(let).strip() == target.strip()


def test_call():
    func = Symbol("+")
    a = Var("a")
    b = Var("b")
    call = Expression(func, a, b)
    assert str(call) == "(+ a b)"


def test_recursive_call():
    func = Symbol("+")
    a = Var("a")
    b = Var("b")
    call0 = Call(func, [a, Call(func, [a, b])])
    print(call0)


def test_function():
    a = Var("a")
    b = Var("b")

    add = Expression(Symbol('+'), a, b)
    let = LetExpr(vars=[], body=[add])

    func = Function(name="sum", args=[a, b], body=let)
    print(func)
    target = \
        """
(defun sum (a b)
    (let*
        ()
        (+ a b)
    )
)
"""
    source = str(func).strip()
    target = target.strip()
    assert str(func).strip() == target.strip()


def test_ifelse():
    a = Var("a")
    b = Var("b")
    cond = Var('cond')
    expr0 = Expression(Symbol('+'), a, b)
    expr1 = Expression(Symbol('-'), a, b)
    ifelse = IfElse(cond=cond, then_body=expr0, else_body=expr1)
    target = '''
(if cond
    (+ a b)
    (- a b))
    '''
    assert str(ifelse).strip() == target.strip()


def test_ifelse_embed_in_expr():
    a = Var("a")
    b = Var("b")

    cond = Var('cond')
    expr0 = Expression(Symbol('+'), a, b)
    expr1 = Expression(Symbol('-'), a, b)
    ifelse = IfElse(cond=cond, then_body=expr0, else_body=expr1)

    let = LetExpr(vars=[], body=[ifelse])
    func = Function(name="something", args=[cond, a, b], body=let)
    target = '''
(defun something (cond a b)
    (let*
        ()
        (if cond
            (+ a b)
            (- a b))
    )
)
    '''
    assert str(func).strip() == target.strip()
