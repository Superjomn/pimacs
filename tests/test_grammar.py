import pytest

from pimacs.lang.parser import get_lark_parser, get_parser

lark_parser = get_lark_parser()
parser = get_parser()


def test_basic():
    basic_code = '''
def hello-world(name:Str) -> nil:
    print("Hello, world!")
'''
    for token in lark_parser.lex(basic_code):
        print('token:', repr(token))
    print("lark:\n", lark_parser.parse(basic_code).pretty())
    print(parser.parse(basic_code))


def test_constant():
    code = '''
var a: Bool = true
var b = false
var c = nil'''
    for token in lark_parser.lex(code):
        print('token:', repr(token))
    assert lark_parser.parse(code) is not None
    print(parser.parse(code))


def test_function_def():
    code = '''
def hello-world(name:Str):
    pass

def hello-world(name:Str, age:Int):
    pass

def hello-world(name:Str, age:Int) -> nil:
    # some comment
    pass

def hello-world(name:Str, age:Int) -> nil:
    "Some docs"
    # some comment
    return format("Hello %s, age: %d", name, age)
'''
    for token in lark_parser.lex(code):
        print('token:', repr(token))
    assert lark_parser.parse(code) is not None


def test_control_flow():
    code = '''
if x > 10:
    pass
elif x > 5:
    pass
else:
    pass

for i in range(10):
    var b = i * 2

while x > 0:
    var c = x - 1
'''
    for token in lark_parser.lex(code):
        print('token:', repr(token))
    assert lark_parser.parse(code) is not None


def test_optional():
    code = '''
var a : Int?
var b : Int? = 10
var c = a + b!
'''
    for token in lark_parser.lex(code):
        print('token:', repr(token))


def test_dict():
    code = '''
var a = { "a": 1, "b": 2 }
'''
    for token in lark_parser.lex(code):
        print('token:', repr(token))
    assert lark_parser.parse(code) is not None


def test_list():
    code = '''
var a = [1, 2, 3]
var b = [1,
2, 3]
var c = [1,
    2, 3,
    4, 5, 6]
'''
    tree = lark_parser.parse(code)
    assert tree
    print(tree.pretty())


def test_type():
    code = '''
var a: Int
var b: MyType[Int, Float]
var c: MyType[Int, MyType[Float, Bool]]
'''
    tree = lark_parser.parse(code)
    assert tree
    print(tree.pretty())


def test_decorator():
    code = '''
@decorator
def hello-world():
    pass

@decorator(1, 2, 3)
@decorator1
@decorator2
def hello-world():
    pass

@template [ T ]
def templated_fn(a: T):
    pass
'''
    for token in lark_parser.lex(code):
        print('token:', repr(token))
    tree = lark_parser.parse(code)
    assert tree
    print(tree.pretty())


test_basic()
