from pyimacs.elisp.struct import struct


@struct
class Point:
    x: int
    y: int


def test_struct_basic():
    def fn():
        point = Point(1, 2)


test_struct_basic()
