from pyimacs.elisp.buffer import Buffer


def test_Buffer():
    buffer = Buffer(buf_name="hello")
    print(buffer._handle)
