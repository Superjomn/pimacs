from pimacs.ast.ast import *


def test_basic():
    c = Constant(value=100, loc=None)
    v = VarDecl('x', init=c)

    assert v in c.users


def test_WeakSet_behavior():
    value0 = Constant(value=100, loc=None)
    value1 = Constant(value=200, loc=None)

    set = weakref.WeakSet()
    for i in range(10):
        set.add(value0)
        set.add(value1)

    assert len(set) == 2


def test_replace_child_VarDecl():
    value = Constant(value=100, loc=None)
    var = VarDecl('x', init=value)
    value1 = Constant(value=200, loc=None)
    var.replace_child(value, value1)
    assert var.init == value1

    assert len(value1.users) == 1
    value1.replace_all_uses_with(value)
    assert len(value1.users) == 0


def test_replace_child_BinaryOp():
    value = Constant(value=100, loc=None)
    value1 = Constant(value=200, loc=None)
    binop = BinaryOp(op='+', left=value, right=value1, loc=None)
    value2 = Constant(value=300, loc=None)
    binop.replace_child(value, value2)
    assert binop.left == value2

    assert len(value2.users) == 1
    value2.replace_all_uses_with(value)
    assert len(value2.users) == 0


def test_replace_child_UnaryOp():
    value = Constant(value=100, loc=None)
    unaryop = UnaryOp(op='-', operand=value, loc=None)
    value1 = Constant(value=200, loc=None)
    unaryop.replace_child(value, value1)
    assert unaryop.operand == value1

    assert len(value1.users) == 1
    value1.replace_all_uses_with(value)
    assert len(value1.users) == 0


def test_replace_child_Call():
    value = Constant(value=100, loc=None)
    func = UFunction(name='fn', return_type=None, loc=None)
    call = Call(func=func, args=(value,), loc=None)
    func1 = UFunction(name='fn1', return_type=None, loc=None)
    value1 = Constant(value=200, loc=None)
    call.replace_child(func, func1)
    assert call.func == func1

    for no, user in enumerate(func1.users):
        assert user == call

    assert len(func1.users) == 1
    func1.replace_all_uses_with(func)
    assert len(func1.users) == 0


def test_replace_child_If():
    test = Constant(value=100, loc=None)
    value0 = Constant(value=100, loc=None)
    value1 = Constant(value=200, loc=None)
    if_ = If(cond=test, then_branch=value0, else_branch=value1, loc=None)
    if_.replace_child(test, value1)
    assert if_.cond == value1

    assert len(value1.users) == 1
    value1.replace_all_uses_with(value0)
    assert len(value1.users) == 0


def test_replace_child_Function():

    arg0 = Arg(name='arg0', loc=None)
    var = VarDecl('x', init=arg0)
    var1 = VarDecl('y', init=arg0)
    body = Block(stmts=(var, var1), loc=None)
    func = Function(name='fn', args=(arg0), body=body, loc=None)

    # Block doesn't "use" the statements
    assert len(var.users) == 0
    assert len(arg0.users) == 2

    arg1 = Arg(name='arg1', loc=None)

    arg0.replace_all_uses_with(arg1)
    assert len(arg0.users) == 0
    assert len(arg1.users) == 2
