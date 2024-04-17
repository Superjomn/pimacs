from io import StringIO

import code_snippets

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


def test_IRPrinter_var():
    printer = IRPrinter(StringIO())
    file = parse(code_snippets.var_case)
    printer(file)
    output = printer.os.getvalue()

    assert output.strip() == \
        '''
var a :Int
var b :Int = 1
var c = 1
var d :Float
var e :Float = 1.0
var f = 1.0
'''.strip()


def test_IRPrinter_func():
    printer = IRPrinter(StringIO())
    file = parse(code_snippets.func_case)
    printer(file)
    output = printer.os.getvalue()

    print(output)

    assert output.strip() == \
        '''
def hello-0 (name :String) -> nil:
    var a = "Hello " + name
    print("hello %s")


def fib (n :Int) -> Int:
    if n <= 1:
        return n

    return fib(n - 1) + fib(n - 2)
    '''.strip()


test_basic()
test_IRPrinter_var()
test_IRPrinter_func()
