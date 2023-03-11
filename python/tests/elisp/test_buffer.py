from pyimacs.elisp.buffer import Buffer
from pyimacs.runtime import aot

from pyimacs import compiler


@aot
def buffer_tester() -> str:
    buf = Buffer("test")
    name = buf.get_name()
    return name


# def test_Buffer_basic():
#     res = compiler.compile(buffer_tester, signature="void -> str")
#     print(res)
