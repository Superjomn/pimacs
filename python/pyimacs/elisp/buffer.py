from pyimacs.lang.extension import *


class Buffer:
    def __init__(self, buf_name: str = "", handle=None):
        self._handle = handle
        if buf_name:
            self._handle = buffer_get(buf_name)

    @staticmethod
    def current():
        ''' Get the current buffer. '''
        return Buffer(handle=current_buffer())

    @staticmethod
    def set_buffer(buf: "Buffer") -> None:
        set_buffer(buf._handle)


@register_extern("buffer-get")
def buffer_get(name: str) -> object: ...


@register_extern("current-buffer")
def current_buffer() -> object: ...


@register_extern("set-buffer")
def set_buffer(buf: object) -> None: ...
