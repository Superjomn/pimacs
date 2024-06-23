import os
from pathlib import Path

from ..utils import filecheck

workspace = Path(os.path.dirname(os.path.abspath(__file__)))


def test_basic():
    filecheck(workspace=workspace, file="./test_basic.pim")


def test_name_binding():
    filecheck(workspace=workspace, file="./test_name_binding.pim")


def test_class_template():
    filecheck(workspace=workspace, file="./test_class_template.pim")


def test_org_element():
    filecheck(workspace=workspace, file="./test_org_element.pim")
