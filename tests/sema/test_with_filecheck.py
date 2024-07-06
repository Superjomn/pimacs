import subprocess
from typing import List

from ..utils import CURDIR, filecheck

workspace = CURDIR / "sema"


def test_basic():
    filecheck(workspace=workspace, file="./test-basic.pim")


def test_name_binding():
    filecheck(workspace=workspace, file="./test-name-binding.pim")


def test_class_template():
    filecheck(workspace=workspace, file="./test-class-template.pim")


def test_org_element():
    filecheck(workspace=workspace, file="./test-org-element.pim")


def test_imports():
    def abs_path(path): return str(workspace / path)
    path_group0 = [str(workspace), abs_path(
        "./test-imports.pim"), abs_path("./a-module.pim")]
    path_group1 = [str(workspace / "../../pimacs/builtin/dict.pim")]

    linkcheck(paths=[path_group0, path_group1],
              test_path=abs_path("./test-imports.pim"))


def linkcheck(paths: List[List[str]], test_path: str):
    # paths: root:path0:path1;root1:path0:path1
    parser_script = CURDIR / "parser_entry.py"

    the_paths = ';'.join([':'.join(path) for path in paths])

    command = f"{parser_script} link \"{
        the_paths}\" --target lisp_code | filecheck {test_path}"

    try:
        # Run the command and capture the output
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, text=True)
        print("Command output:", output)
    except subprocess.CalledProcessError as e:
        # If an error occurs, print the error output
        print("Error:", e.stdout)
        raise e


if __name__ == "__main__":
    test_imports()
