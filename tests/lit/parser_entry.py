#!/usr/bin/env python3
from pprint import pprint

import click

from pimacs.codegen.phases import (gen_lisp_code, parse_ast, perform_sema,
                                   translate_to_lisp)
from pimacs.sema.ast_visitor import print_ast
from pimacs.sema.context import ModuleContext


@click.command()
@click.argument("filename", type=str)
@click.option("--sema", type=bool, default=False)
@click.option("--mark-unresolved", type=bool, default=False)
@click.option("--enable-exception", type=bool, default=False)
@click.option("--display-ast", type=bool, default=False)
@click.option("--target", type=click.Choice(["ast", "lisp_ast", "lisp_code"]), default="ast")
def main(filename: str, sema: bool, mark_unresolved: bool, enable_exception: bool, display_ast: bool, target: str):
    the_ast = parse_ast(filename=filename)
    if display_ast:
        print("AST:\n")
        pprint(the_ast)

    if sema:
        ctx = ModuleContext(enable_exception=enable_exception)
        the_ast = perform_sema(ctx, the_ast)  # type: ignore

        if display_ast:
            print("SEMA:\n")
            pprint(the_ast)

    if the_ast:
        print_ast(the_ast)
    else:
        return

    if target in ("lisp_ast", "lisp_code"):
        the_ast = translate_to_lisp(the_ast)  # type: ignore
        print("LISP AST:\n")
        pprint(the_ast)

    if target == "lisp_code":
        code = gen_lisp_code(the_ast)  # type: ignore
        print("LISP Code:\n")
        print(code)


if __name__ == "__main__":
    main()
