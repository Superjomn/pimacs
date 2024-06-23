import os
from pathlib import Path

from ..utils import filecheck

workspace = Path(os.path.dirname(os.path.abspath(__file__)))


def test_basic():
    filecheck(workspace=workspace, file="./test-basic.pim")


def test_name_binding():
    filecheck(workspace=workspace, file="./test-name-binding.pim")


def test_class_template():
    filecheck(workspace=workspace, file="./test-class-template.pim")


def test_org_element():
    filecheck(workspace=workspace, file="./test-org-element.pim")
