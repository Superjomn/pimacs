#!/usr/bin/env python3
from pathlib import Path

import click

from pimacs.codegen.transpiler import ModulePath, Transpiler
from pimacs.logger import get_logger

logger = get_logger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("filename", type=str)
def transpile(filename: str):
    builtin_modules = [
        ModulePath(Path(__file__).parent /
                   "../pimacs/builtin", Path("dict.pim")),
        ModulePath(Path(__file__).parent /
                   "../pimacs/builtin", Path("list.pim")),
    ]
    transpiler = Transpiler(builtin_modules=builtin_modules)

    target = ModulePath(root=None, module_path=Path(filename))
    code = transpiler.run(target)
    print(code)


if __name__ == "__main__":
    cli.add_command(transpile)
    cli()
