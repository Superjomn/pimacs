import os
import subprocess
from pathlib import Path
from typing import List

from ..utils import CURDIR, filecheck

workspace = Path(os.path.dirname(os.path.abspath(__file__)))


def test_basic():
    filecheck(workspace=workspace, file="./test-basic.pim")


def test_name_binding():
    filecheck(workspace=workspace, file="./test-name-binding.pim")


def test_class_template():
    filecheck(workspace=workspace, file="./test-class-template.pim")


def test_org_element():
    filecheck(workspace=workspace, file="./test-org-element.pim")


def test_imports():
    linkcheck(root='.', paths=["./test-imports.pim", "./a-module.pim"])


def linkcheck(root: str, paths: List[str]):
    parser_script = CURDIR / "parser_entry.py"

    for path in paths:
        command = f"{parser_script} link {
            root} --target lisp_code --files {':'.join(paths)} | filecheck {path}"
        try:
            # Run the command and capture the output
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT, text=True)
            print("Command output:", output)
        except subprocess.CalledProcessError as e:
            # If an error occurs, print the error output
            print("Error:", e.stdout)
