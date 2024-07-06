#!/usr/bin/env python3
import os
from pathlib import Path
from pprint import pprint

import click

from pimacs.codegen.phases import (gen_lisp_code, parse_ast, perform_sema,
                                   translate_to_lisp)
from pimacs.logger import get_logger
from pimacs.sema.ast_visitor import print_ast
from pimacs.sema.context import ModuleContext
from pimacs.sema.linker import Linker
from pimacs.sema.utils import bcolors, print_colored

logger = get_logger(__name__)


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
    the_ast = parse_ast(file=filename)
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
        print_colored("LISP AST:\n", bcolors.OKGREEN)
        pprint(the_ast)

    if target == "lisp_code":
        code = gen_lisp_code(the_ast)  # type: ignore
        print_colored("LISP Code:\n", bcolors.OKGREEN)
        print(code)


@click.command()
@click.argument("paths", type=str)
@click.option("--target", type=click.Choice(["ast", "lisp_ast", "lisp_code"]), default="ast")
@click.option("--display-ast", type=bool, default=False)
@click.option("--modules-to-dump", type=str, default="")
def link(paths: str, target: str, display_ast: bool, modules_to_dump: str) -> None:
    '''
    Link the files.

    Args:
        paths: The paths to search for the modules. The format is "root0:path1:path2;root1:path1:path2..."
        target: The target to display. The options are "ast", "lisp_ast", "lisp_code"
        display_ast: Whether to display the AST
        modules_to_dump: The files to dump the AST and the lisp code, format: "module1:module2"
    '''
    linker = Linker()
    modules_to_dump = set(
        filter(None, modules_to_dump.split(":")))  # type: ignore

    for group in filter(None, paths.split(";")):
        files = list(map(Path, group.split(":")))
        root = files[0] if files[0].is_dir() else None
        if root is None:
            root = extract_root_from_paths(group)
        else:
            files = filter(None, files[1:])  # type: ignore

        if not files:
            assert root
            linker.add_module_root(root)
        else:
            for file in files:
                logger.info(f"Linker adding path: {file}, root: {root}")
                linker.add_module_path(file, root)

    linker()

    for record in linker.mapping.records:
        to_dump = (not modules_to_dump) or record.sema.ctx.name in modules_to_dump
        if display_ast:
            print(f"{record.sema.ctx.name}:\n\n")
            pprint(record.ast)

        if target in ("lisp_ast", "lisp_code"):
            print(f"** translate {record.sema.ctx.name} to lisp **")
            ast = translate_to_lisp(record.sema.ctx, record.ast)
            if target == "lisp_ast":
                if to_dump:
                    print("LISP AST:\n")
                    pprint(ast)
            elif target == "lisp_code":
                code = gen_lisp_code(ast)
                if to_dump:
                    print("LISP Code:\n")
                    print(code)


def extract_root_from_paths(paths: str) -> Path:
    """
    Extracts the root directory from a colon-separated list of paths.

    If the paths start with a root directory (e.g., "root0:path1:path2"), returns the root directory.
    Otherwise, returns the common parent directory of the paths.

    Args:
        paths: A colon-separated list of paths.

    Returns:
        The extracted root directory.
    """
    path_components = [Path(path).absolute() for path in paths.split(":")]

    common_prefix = os.path.commonprefix(
        [str(path) for path in path_components])

    if os.path.isdir(common_prefix):
        return Path(common_prefix)

    parent_dir = os.path.dirname(common_prefix)
    if os.path.isdir(parent_dir):
        return Path(parent_dir)

    # If the parent directory is not a directory, raise a ValueError
    raise ValueError(
        f"Failed to extract a valid root directory from the paths: {paths}")


if __name__ == "__main__":
    cli.add_command(parse)
    cli.add_command(link)
    cli()
