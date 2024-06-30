#!/usr/bin/env python3
from pathlib import Path
from pprint import pprint

import click

from pimacs.codegen.phases import (gen_lisp_code, parse_ast, perform_sema,
                                   translate_to_lisp)
from pimacs.sema.ast_visitor import print_ast
from pimacs.sema.context import ModuleContext
from pimacs.sema.linker import Linker


@click.group()
def cli():
    pass


@click.command()
@click.argument("filename", type=str)
@click.option("--sema", type=bool, default=False)
@click.option("--mark-unresolved", type=bool, default=False)
@click.option("--enable-exception", type=bool, default=False)
@click.option("--display-ast", type=bool, default=False)
@click.option("--target", type=click.Choice(["ast", "lisp_ast", "lisp_code"]), default="ast")
def parse(filename: str, sema: bool, mark_unresolved: bool, enable_exception: bool, display_ast: bool, target: str):
    the_ast = parse_ast(filename=filename)
    ctx = ModuleContext(enable_exception=enable_exception)
    if target in ("lisp_ast", "lisp_code"):
        sema = True

    if display_ast:
        print("AST:\n")
        pprint(the_ast)

    if sema:
        the_ast = perform_sema(ctx, the_ast)  # type: ignore

        if display_ast:
            print("SEMA:\n")
            pprint(the_ast)

    if the_ast:
        print_ast(the_ast)
    else:
        return

    if target in ("lisp_ast", "lisp_code"):
        the_ast = translate_to_lisp(ctx, the_ast)  # type: ignore
        print("LISP AST:\n")
        pprint(the_ast)

    if target == "lisp_code":
        code = gen_lisp_code(the_ast)  # type: ignore
        print("LISP Code:\n")
        print(code)


@click.command()
@click.argument("root", type=str)
@click.option("--files", type=str)
@click.option("--target", type=click.Choice(["ast", "lisp_ast", "lisp_code"]), default="ast")
@click.option("--display-ast", type=bool, default=False)
def link(root: str, files: str, target: str, display_ast: bool):
    root = Path(root) if root else None  # type: ignore
    paths_ = list(map(Path, files.split(":")))

    linker = Linker()

    if not files:
        linker.add_module_root(root)
    else:
        for path in paths_:
            linker.add_module_path(path, root)

    linker()

    if display_ast:
        print("AST:\n")

        for record in linker.mapping.records:
            print(f"{record.sema.ctx.name}:\n\n")
            pprint(record.ast)

    if target in ("lisp_ast", "lisp_code"):
        for record in linker.mapping.records:
            ast = record.ast
            ast = translate_to_lisp(record.sema.ctx, ast)

            if target == "lisp_ast":
                print("LISP AST:\n")
                pprint(ast)

            if target == "lisp_code":
                code = gen_lisp_code(ast)
                print("LISP Code:\n")
                print(code)


if __name__ == "__main__":
    cli.add_command(parse)
    cli.add_command(link)
    cli()
