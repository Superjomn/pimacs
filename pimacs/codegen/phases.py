from io import StringIO

from lark import UnexpectedToken

import pimacs.ast.ast as ast
import pimacs.lisp.ast as lisp_ast
from pimacs.ast.parser import get_parser
from pimacs.lisp.translator import LispTranslator
from pimacs.logger import get_logger
from pimacs.sema.context import ModuleContext
from pimacs.sema.file_sema import FileSema
from pimacs.sema.type_checker import amend_compose_types_with_module

from .codegen import Codegen

logger = get_logger(__name__)


def parse_ast(
    code: str | None = None, file: str = "<pimacs>"
) -> ast.File:
    """
    Parse the code and return the AST.
    """
    if code:
        source = ast.PlainCode(code)
        parser = get_parser(code=code)
    else:
        code = open(file).read()
        source = ast.FileName(file)  # type: ignore
        parser = get_parser(code=None, filename=file)

    try:
        stmts = parser.parse(code)
    except UnexpectedToken as e:
        logger.error(f"Unexpected token: {e.token}")
        logger.error(f"Line: {e.line}, Column: {e.column}")
        logger.error(f"Expected token: {e.expected}")
        logger.error(f"Context: {e.get_context(code)}")

        raise e

    file = ast.File(stmts=stmts, loc=ast.Location(
        source, 0, 0))  # type: ignore
    return file  # type: ignore


def perform_sema(ctx: ModuleContext, the_ast: ast.File) -> ast.File | None:
    """
    Perform semantic analysis on the AST.

    It returns the IR if the semantic analysis succeeds.
    """
    module = ast.Module(name=ctx.name, path=None, loc=None)
    amend_compose_types_with_module(module, the_ast)

    sema = FileSema(ctx)

    the_ir = sema(the_ast)
    if sema.succeed:
        return the_ir

    return None


def translate_to_lisp(ctx: ModuleContext, the_ast: ast.File) -> lisp_ast.Module:
    """
    Translate the AST to Lisp AST.
    """
    translator = LispTranslator(ctx)
    return translator(the_ast)  # type: ignore


def gen_lisp_code(node: lisp_ast.Node) -> str:
    """
    Generate Lisp code from the Lisp AST.
    """
    os = StringIO()
    codegen = Codegen(os)
    return codegen(node)
