from io import StringIO

import code_snippets

from pimacs.lang.ir_visitor import IRPrinter, IRVisitor
from pimacs.lang.parser import get_lark_parser, get_parser


class MyIRVisitor(IRVisitor):
    pass


def test_basic():
    parser = get_parser()
    res = parser.parse(code_snippets.var_case)
    print(res)

    visitor = MyIRVisitor()
    visitor.visit(res[0])


def test_IRPrinter():
    printer = IRPrinter(StringIO())
    parser = get_parser()
    nodes = parser.parse(code_snippets.var_case)
    for node in nodes:
        printer(node)
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


# test_basic()
test_IRPrinter()
