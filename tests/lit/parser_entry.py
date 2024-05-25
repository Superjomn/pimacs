#!/usr/bin/env python3
from io import StringIO

import click

from pimacs.sema.ast_visitor import IRPrinter
from pimacs.sema.context import ModuleContext
from pimacs.transpiler.phases import parse_ast, perform_sema


@click.command()
@click.argument("filename", type=str)
@click.option("--sema", type=bool, default=False)
def main(filename: str, sema: bool):
    file = parse_ast(filename=filename, sema=sema)
    if sema:
        ctx = ModuleContext()
        file = perform_sema(ctx, file)  # type: ignore

    if file:
        printer = IRPrinter(StringIO())
        printer(file)

        output = printer.os.getvalue()
        print(output)


if __name__ == "__main__":
    main()
