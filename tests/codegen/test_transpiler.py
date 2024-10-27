from pathlib import Path

from pimacs.codegen.transpiler import ModulePath, Transpiler


def test_Transpiler():
    builtin_root = Path(__file__).parent / "../../pimacs/builtin"
    assert builtin_root.exists()

    builtin_modules = [
        ModulePath(builtin_root, Path("dict.pim")),
        ModulePath(builtin_root, Path("list.pim")),
    ]

    transpiler = Transpiler(builtin_modules=builtin_modules)
    test_module = ModulePath(root=None,
                             module_path=Path(__file__).parent / "test-basic.pim")
    code = transpiler.run(test_module)
    print(code)
