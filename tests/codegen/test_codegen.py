import os
from pathlib import Path

from ..utils import filecheck

workspace = Path(os.path.dirname(os.path.abspath(__file__)))


def test_basics_with_filecheck():
    filecheck(workspace=workspace, file="./test-basic.pim")


def test_builtin_dict():
    filecheck(workspace=workspace, file="../../pimacs/builtin/dict.pim")


def test_builtin_list():
    filecheck(workspace=workspace, file="../../pimacs/builtin/list.pim")


if __name__ == "__main__":
    test_basics_with_filecheck()
