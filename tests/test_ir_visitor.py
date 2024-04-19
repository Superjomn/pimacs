from io import StringIO

import code_snippets
import pytest

from pimacs.lang.ir_visitor import IRPrinter, IRVisitor
from pimacs.lang.parser import get_lark_parser, get_parser, parse


class MyIRVisitor(IRVisitor):
    pass


def test_basic():
    parser = get_parser()
    res = parser.parse(code_snippets.var_case)
    print(res)

    visitor = MyIRVisitor()
    visitor.visit(res[0])


def test_IRPrinter_func():
    printer = IRPrinter(StringIO())
    file = parse(code_snippets.func_case)
    printer(file)
    output = printer.os.getvalue()

    print(output)

    assert output.strip() == \
        '''
def hello-0 (name :Str) -> nil:
    var a = "Hello " + name
    print("hello %s")


def fib (n :Int) -> Int:
    if n <= 1:
        return n

    return fib(n - 1) + fib(n - 2)
    '''.strip()


@pytest.mark.parametrize("code, target", [
    (code_snippets.decorator_case1,
     '''
@some-decorator(100)
@interactive("P:")
def hello (name :Str) -> nil:
    print("Hello %s")
'''),
    (code_snippets.decorator_case,
     '''
@interactive
def hello (name :Str) -> nil:
    print("Hello %s")
'''
     ),

    (code_snippets.var_case,
     '''
var a :Int
var b :Int = 1
var c = 1
var d :Float
var e :Float = 1.0
var f = 1.0
'''),

    (code_snippets.class_case,
     '''
class Person:
    var name :Str
    var age :Int
    def __init__ (self, name :Str, age :Int) -> nil:
        self.name = name
        self.age = age


    def get-name (self) -> Str:
        return self.name


    def get-age (self) -> Int:
        return self.age
'''
     )

])
def test_printer(code, target):
    printer = IRPrinter(StringIO())
    file = parse(code)
    print('file', file)
    printer(file)
    output = printer.os.getvalue()

    assert output.strip() == target.strip(), "\n"+output
