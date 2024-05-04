from io import StringIO

import pytest
from code_snippets import snippets

import pimacs.lang.ir as ir
from pimacs import BUILTIN_SOURCE_ROOT, SOURCE_ROOT
from pimacs.lang.ir_visitor import IRPrinter, IRVisitor
from pimacs.lang.parser import get_lark_parser, get_parser, parse


class MyIRVisitor(IRVisitor):
    pass


def test_basic():
    parser = get_parser()
    res = parser.parse(snippets.var_case)
    print(res)

    visitor = MyIRVisitor()
    visitor.visit(res[0])


def test_IRPrinter_func():
    printer = IRPrinter(StringIO())
    file = parse(code=snippets.func_case, build_ir=False)
    printer(file)
    output = printer.os.getvalue()

    print(output)

    assert output.strip() == \
        '''
def hello-0 (name :Str) -> nil:
    var a = "Hello " + name
    print("hello %s", a)


def fib (n :Int) -> Int:
    if n <= 1:
        return n

    return fib(n - 1) + fib(n - 2)
    '''.strip()


@pytest.mark.parametrize("snippet_key, target", [
    ("decorator_case1",
     '''
@some-decorator(100, 200)
@interactive("P:")
def hello (name :Str) -> nil:
    print("Hello %s", name)
'''
     ),
    ("decorator_case",
     '''
@interactive
def hello (name :Str) -> nil:
    print("Hello %s", name)
'''
     ),

    ("var_case",
     '''
var a :Int
var b :Int = 1
var c = 1
var d :Float
var e :Float = 1.0
var f = 1.0
'''),

    ("class_case",
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
     ),
    ("func_with_docstring_case",
        """
def hello (name :Str) -> nil:
    "Some docs"
    return
        """
     ),
])
def test_printer(snippet_key, target):
    code = snippets[snippet_key]
    printer = IRPrinter(StringIO())
    file = parse(code=code, build_ir=False)
    printer(file)
    output = printer.os.getvalue()

    assert output.strip() == target.strip(), "\n"+output
