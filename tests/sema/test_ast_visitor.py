from io import StringIO

import pytest
from code_snippets import snippets

import pimacs.ast.ast as ir
from pimacs import BUILTIN_SOURCE_ROOT, SOURCE_ROOT
from pimacs.ast.ast_printer import IRPrinter
from pimacs.ast.ast_visitor import IRVisitor
from pimacs.ast.parser import get_lark_parser, get_parser
from pimacs.transpiler.phases import parse_ast, perform_sema


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
    file = parse_ast(code=snippets.func_case)
    printer(file)
    output = printer.os.getvalue()

    print('output:\n', output)

    assert (
        output.strip()
        == """
def hello-0 (name :Str) -> nil:
    var a :Unk = "Hello " + name
    print("hello %s", a)


def fib (n :Int) -> Int:
    if n <= 1:
        return n

    return fib(n - 1) + fib(n - 2)
    """.strip()
    )


@pytest.mark.parametrize(
    "snippet_key, target",
    [
        (
            "decorator_case1",
            """
@some-decorator(100, 200)
@interactive("P:")
def hello (name :Str) -> nil:
    print("Hello %s", name)
""",
        ),
        (
            "decorator_case",
            """
@interactive
def hello (name :Str) -> nil:
    print("Hello %s", name)
""",
        ),
        (
            "var_case",
            """
var a :Int
var b :Int = 1
var c :Int = 1
var d :Float
var e :Float = 1.0
var f :Float = 1.0
""",
        ),
        (
            "class_case",
            """
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
""",
        ),
        (
            "func_with_docstring_case",
            """
def hello (name :Str) -> nil:
    "Some docs"
    return
        """,
        ),
    ],
)
def test_printer(snippet_key, target):
    code = snippets[snippet_key]
    printer = IRPrinter(StringIO())
    file = parse_ast(code=code)
    printer(file)
    output = printer.os.getvalue()

    assert output.strip() == target.strip(), "\n" + output
