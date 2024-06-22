#!/usr/bin/env python3
from pprint import pprint

import click

from pimacs.sema.ast_visitor import print_ast
from pimacs.sema.context import ModuleContext
from pimacs.transpiler.phases import parse_ast, perform_sema


@click.command()
@click.argument("filename", type=str)
@click.option("--sema", type=bool, default=False)
@click.option("--mark-unresolved", type=bool, default=False)
@click.option("--enable-exception", type=bool, default=False)
@click.option("--display-ast", type=bool, default=False)
def main(filename: str, sema: bool, mark_unresolved: bool, enable_exception: bool, display_ast: bool):
    file = parse_ast(filename=filename)
    if display_ast:
        print("AST:\n")
        pprint(file)

    if sema:
        ctx = ModuleContext(enable_exception=enable_exception)
        file = perform_sema(ctx, file)  # type: ignore

        if display_ast:
            print("SEMA:\n")
            pprint(file)

    if file:
        print_ast(file)


if __name__ == "__main__":
    main()
