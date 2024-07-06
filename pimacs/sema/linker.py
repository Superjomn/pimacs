'''
Linker helps to perform cross-module name binding.
'''
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tabulate import tabulate

from pimacs.ast.ast import Module, Node
from pimacs.logger import get_logger

from .context import ModuleContext
from .file_sema import FileSema

logger = get_logger(__name__)


class ModuleMapping:
    ''' ModuleMapping helps to map the module path to module instance. '''

    def __init__(self):
        # A mapping from the module path to the module instance. Note, the path should be absolute.
        self.mapping: Dict[Path, Module] = {}
        self.name_to_root = {}

    def add_module_root(self, module_root: Path):
        ''' Add the module root path. '''
        logger.debug(f"add module root: {module_root}")
        module_paths = ModuleMapping.collect_module_paths(module_root)
        logger.debug(f"module paths: {module_paths}")
        self._create_module_instances(module_root, module_paths)

    def add_module_path(self, module_path: Path, root: Optional[Path] = None):
        ''' Add the module path. '''
        logger.debug(f"add module path: {module_path}, root: {root}")
        root = root or module_path.parent

        self._create_module_instance(root, module_path)

    def __getitem__(self, module_path: Path) -> Module:
        ''' Get the module instance. '''
        return self.mapping[module_path]

    @property
    def module_paths(self) -> Iterable[Path]:
        ''' Get the module paths. '''
        return self.mapping.keys()

    def module_name_to_root(self, module_name: str) -> Path:
        ''' Get the module root path by the module name. '''
        return self.name_to_root.get(module_name, None)

    @property
    def modules(self) -> Iterable[Module]:
        return self.mapping.values()

    @staticmethod
    def collect_module_paths(module_root: Path):
        ''' Collect the module paths under the module_root.
        The module could be a file with suffix of `.pim` or a directory with a `__module__.pim` file.
        '''
        module_paths = []
        for root, dirs, files in os.walk(module_root):
            for file in files:
                if file.endswith('.pim'):
                    module_paths.append(Path(os.path.join(root, file)))
            for dir in dirs:
                dir_path = Path(os.path.join(root, dir))
                if (dir_path / '__module__.pim').exists():
                    module_paths.append(dir_path)
        return module_paths

    def _create_module_instances(self, module_root: Path, module_paths: List[Path]) -> None:
        ''' Create the module instances. '''
        assert module_paths, f"No module found under {module_root}."
        for module_path in module_paths:
            self._create_module_instance(module_root, module_path)

    def _create_module_instance(self, module_root: Path, module_path: Path) -> Module:
        ''' Create the module instance. '''
        module_root = module_root.absolute()
        module_path = module_root / module_path

        assert module_path.exists(), f"Module {module_path} does not exist."
        assert module_path not in self.mapping, f"Module {
            module_path} already exists."
        module_name = ModuleMapping.get_module_name(module_root, module_path)
        assert module_name
        self.name_to_root[module_name] = module_root
        logger.debug(f"got module: {module_name} from {module_path}")
        module = Module(name=module_name, path=module_path, loc=None)
        module.ctx = ModuleContext(module_name)
        self.mapping[module_path] = module
        return module

    @staticmethod
    def get_module_name(module_root: Path, module_path: Path) -> str:
        # The module name is the relative path to the module root.
        module_root = module_root.absolute()
        return module_path.relative_to(module_root).name.replace('/', '.')[:-4]

    def print_summary(self):
        ''' Print the summary of the module mapping. '''
        data = [[module_path, str(module)]
                for module_path, module in self.mapping.items()]
        print(tabulate(data, headers=[
              "Module Path", "Module"], tablefmt="grid"))


class FileSemaMapping:
    '''
    FileSemaMapping helps to map the file path to the sema instance.
    '''
    @dataclass
    class Record:
        sema: FileSema
        ast: Node

    def __init__(self):
        self._mapping: Dict[Path, FileSemaMapping.Record] = {}
        self._module_mapping = ModuleMapping()

    def add_root(self, path: Path):
        self._module_mapping.add_module_root(path)
        assert self._module_mapping.module_paths

    def add_path(self, path: Path, root: Optional[Path] = None):
        self._module_mapping.add_module_path(path, root)

    def _sema_file(self, path: Path):
        ''' Add the file path. '''
        from pimacs.codegen.phases import parse_ast

        logger.debug(f"sema file: {path}")

        assert path.exists()
        path = path.absolute()

        module = self._module_mapping[path]
        root = self._module_mapping.module_name_to_root(module.name)

        if path not in self._mapping:
            path = root / path
            logger.debug(f"sema file: {path}")
            ast = parse_ast(file=path)  # type: ignore
            # logger.info(f"parse_ast: {path}")
            # pprint(ast)
            sema = FileSema(module.ctx)
            ast = sema(ast)

            self._mapping[path] = FileSemaMapping.Record(sema=sema, ast=ast)

    @contextmanager
    def guard(self):
        ''' Guard the sema mapping. '''
        yield self

        self.freeze()

    def freeze(self):
        ''' Freeze the sema mapping. '''
        logger.debug(f"freeze sema mapping, module_paths: {
                     len(self._module_mapping.module_paths)}")
        for path in self._module_mapping.module_paths:
            self._sema_file(path)

    @property
    def modules(self) -> Iterable[Module]:
        return self._module_mapping.modules

    @property
    def semas(self) -> Iterable[FileSema]:
        for record in self._mapping.values():
            yield record.sema

    @property
    def records(self) -> Iterable["FileSemaMapping.Record"]:
        return self._mapping.values()

    @property
    def file_asts(self) -> Iterable[Node]:
        for path, record in self._mapping.items():
            yield record.ast

    def __getitem__(self, path: Path) -> "FileSemaMapping.Record":
        return self._mapping[path]

    def __len__(self):
        return len(self._mapping)


class Linker:
    def __init__(self):
        self.mapping = FileSemaMapping()

    def add_module_root(self, module_root: Path):
        ''' Add the module root. '''
        self.mapping.add_root(module_root)

    def add_module_path(self, module_path: Path, module_root: Optional[Path] = None):
        ''' Add the module path. '''
        self.mapping.add_path(module_path, module_root)

    def __call__(self):
        # Should add module root/path before freeze
        self.mapping.freeze()

        modules = self.mapping.modules
        for sema in self.mapping.semas:
            sema.link_modules(modules)
