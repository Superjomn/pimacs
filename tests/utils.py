import os
import subprocess
from pathlib import Path

FILE = Path(os.path.abspath(__file__)).resolve()
CURDIR = FILE.parent


def check_codegen(workspace: Path, file: str):
    parser_script = CURDIR / "parser_entry.py"
    test_path = workspace / file
    command = f"{parser_script} parse --sema 1 --mark-unresolved 1 --display-ast 1 --enable-exception 1 --target lisp_code {
        test_path}"
    print(f"Command: {command}")
    try:
        # Run the command and capture the output
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, text=True)
        print("Command output:", output)
    except subprocess.CalledProcessError as e:
        # If an error occurs, print the error output
        print("Error:", e.stdout)
        raise e


def filecheck(workspace: Path, file: str):
    parser_script = CURDIR / "parser_entry.py"
    test_path = workspace / file
    command = f"{parser_script} parse --sema 1 --mark-unresolved 1 --display-ast 1 --enable-exception 1 --target lisp_code {
        test_path} | filecheck {test_path}"
    print(f"Command: {command}")
    try:
        # Run the command and capture the output
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, text=True)
        print("Command output:", output)
    except subprocess.CalledProcessError as e:
        # If an error occurs, print the error output
        print("Error:", e.stdout)
        raise e
