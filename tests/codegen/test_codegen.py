import os
import subprocess
from pathlib import Path

workspace = Path(os.path.dirname(os.path.abspath(__file__)))


def filecheck(file: str):
    parser_script = workspace / "../lit/parser_entry.py"
    test_path = workspace / file
    command = f"{parser_script} --sema 1 --mark-unresolved 1 --display-ast 1 --enable-exception 1 --target lisp_code {
        test_path} | filecheck {test_path}"
    try:
        # Run the command and capture the output
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, text=True)
        print("Command output:", output)
    except subprocess.CalledProcessError as e:
        # If an error occurs, print the error output
        print("Error:", e.stdout)


def test_basics_with_filecheck():
    filecheck("./test_basic.pim")


if __name__ == "__main__":
    test_basics_with_filecheck()
