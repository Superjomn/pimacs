import os
from pathlib import Path

from ..utils import filecheck

workspace = Path(os.path.dirname(os.path.abspath(__file__)))


def test_basics_with_filecheck():
    filecheck(workspace=workspace, file="./test-basic.pim")


if __name__ == "__main__":
    test_basics_with_filecheck()
