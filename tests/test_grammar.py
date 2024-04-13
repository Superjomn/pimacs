import pytest

from pimacs.lang.grammar import parser


def test_basic():
    basic_code = '''
def hello-world():
    print("Hello, world!")
'''
    for token in parser.lex(basic_code):
        print('token:', repr(token))
    assert parser.parse(basic_code) is not None


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
    for token in parser.lex(code):
        print('token:', repr(token))
    assert parser.parse(code) is not None


test_basic()
test_function_def()
