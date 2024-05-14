from typing import Optional

import pimacs.ast.ast as ast
from pimacs.ast.parser import get_parser
from pimacs.sema.sema import Sema


def parse_ast(
    code: str | None = None, filename: str = "<pimacs>", sema=False
) -> ast.File:
    """
    Parse the code and return the AST.
    """
    if code:
        source = ast.PlainCode(code)
        parser = get_parser(code=code)
    else:
        code = open(filename).read()
        source = ast.FileName(filename)  # type: ignore
        parser = get_parser(code=None, filename=filename)

    stmts = parser.parse(code)
    return ast.File(stmts=stmts, loc=ast.Location(source, 0, 0))


def perform_sema(the_ast: ast.File) -> ast.File | None:
    """
    Perform semantic analysis on the AST.

    It returns the IR if the semantic analysis succeeds.
    """
    sema = Sema()
    the_ir = sema.visit(the_ast)
    if sema.succeed:
        return the_ir

    return None
