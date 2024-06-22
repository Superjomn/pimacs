from io import StringIO
from pprint import pprint

from lark import UnexpectedToken

import pimacs.ast.ast as ast
import pimacs.lisp.ast as lisp_ast
from pimacs.ast.parser import get_parser
from pimacs.lisp.translator import LispTranslator
from pimacs.logger import logger
from pimacs.sema.context import ModuleContext
from pimacs.sema.file_sema import FileSema
from pimacs.sema.type_checker import amend_placeholder_types

from .codegen import Codegen


def parse_ast(
    code: str | None = None, filename: str = "<pimacs>"
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

    try:
        stmts = parser.parse(code)
    except UnexpectedToken as e:
        logger.error(f"Unexpected token: {e.token}")
        logger.error(f"Line: {e.line}, Column: {e.column}")
        logger.error(f"Expected token: {e.expected}")
        logger.error(f"Context: {e.get_context(code)}")

        raise e

    file = ast.File(stmts=stmts, loc=ast.Location(source, 0, 0))
    amend_placeholder_types(file)
    return file


def perform_sema(ctx: ModuleContext, the_ast: ast.File) -> ast.File | None:
    """
    Perform semantic analysis on the AST.

    It returns the IR if the semantic analysis succeeds.
    """
    sema = FileSema(ctx)
    the_ir = sema(the_ast)
    if sema.succeed:
        return the_ir

    pprint(the_ir)

    return None


def translate_to_lisp(the_ast: ast.File) -> lisp_ast.Module:
    """
    Translate the AST to Lisp AST.
    """
    translator = LispTranslator()
    return translator(the_ast)  # type: ignore


def gen_lisp_code(node: lisp_ast.Node) -> str:
    """
    Generate Lisp code from the Lisp AST.
    """
    os = StringIO()
    codegen = Codegen(os)
    return codegen(node)
