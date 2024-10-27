'''
This file contains all the functionalities for the transpiler.
'''

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pimacs.codegen.phases import gen_lisp_code, translate_to_lisp
from pimacs.logger import get_logger
from pimacs.sema.linker import Linker
from pimacs.sema.utils import bcolors, print_colored

logger = get_logger(__name__)


@dataclass
class ModulePath:
    root: Optional[Path]
    module_path: Optional[Path]

    @property
    def module_name(self) -> str:
        """
        Returns the module name as a string. If `root` is provided,
        the module name will be the relative path from the `root` to the `module_path`,
        with path separators replaced by dots. If `root` is not provided, the module name
        will be the stem of the `module_path`.

        :raises ValueError: If `module_path` is not provided.
        :return: The module name as a string.
        """
        if self.module_path is None:
            raise ValueError("module_path cannot be None")

        if self.root is None:
            return self.module_path.stem
        else:
            try:
                relative_path = self.module_path.relative_to(self.root)
                return str(relative_path).replace("/", ".")
            except ValueError as e:
                raise ValueError(
                    f"Cannot determine module name: module_path {
                        self.module_path} "
                    f"is not within root {self.root}"
                ) from e

    def __post_init__(self):
        if self.root is not None:
            if not self.root.is_dir() or not self.root.exists():
                raise ValueError(
                    f"root path {self.root} must be an existing directory"
                )


class Transpiler:
    def __init__(self, builtin_modules: Optional[List[ModulePath]] = None) -> None:
        self._builtin_modules = builtin_modules or self.get_system_modules()

    def run(self, target: ModulePath):
        linker = Linker()

        code_modules = []

        linker.add_module_path(target.module_path, target.root)

        for module in self._builtin_modules:
            linker.add_module_path(module.module_path, module.root)

        linker()

        for record in linker.mapping.records:
            if record.sema.ctx.name == target.module_name:
                print_colored(
                    f"Transpiling {record.sema.ctx.name} to lisp\n\n", color=bcolors.OKGREEN)
                ast = translate_to_lisp(record.sema.ctx, record.ast)
                code = gen_lisp_code(ast)
                code_modules.append(code)

        return '\n'.join(code_modules)

    def get_system_modules(self) -> Iterable[ModulePath]:
        ''' Get the default system modules. '''
        paths = os.environ.get("PIMACS_SYSTEM_MODULES", "")

        for group in filter(None, paths.split(";")):
            files = list(map(Path, group.split(":")))
            root = files[0] if files[0].is_dir() else None
            if root is None:
                root = extract_root_from_paths(group)
            else:
                files = filter(None, files[1:])  # type: ignore

            the_root = None

            if not files:
                assert root
                yield ModulePath(root, the_root)
            else:
                for file in files:
                    logger.info(f"Linker adding path: {file}, root: {root}")
                    yield ModulePath(root, file)


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
