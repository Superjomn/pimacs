#!/usr/bin/env python3
from io import StringIO

import click

from pimacs.lang.ir_visitor import IRPrinter
from pimacs.lang.parser import parse


@click.command()
@click.argument("filename", type=str)
@click.option("--sema", type=bool, default=False)
def main(filename: str, sema: bool):
    file = parse(filename=filename, sema=sema)
    if file:
        printer = IRPrinter(StringIO())
        printer(file)

        output = printer.os.getvalue()
        print(output)


if __name__ == "__main__":
    main()
