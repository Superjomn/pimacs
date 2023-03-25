from typing import *

from pyimacs.lang.extension import *


class Buffer(Ext):
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

    def kill(self) -> None:
        kill_buffer_by_buffer(self._handle)

    @staticmethod
    def kill_buffer(self, x: Any) -> None:
        if type(x) is str:
            kill_buffer_by_name(x)
        elif type(x) is object:
            kill_buffer_by_buffer(x)

    def get_name(self) -> str:
        return buffer_file_name(self._handle)

    def __handle_return__(self):
        return self._handle


@register_extern("point")
def point() -> int: ...


@register_extern("point-max")
def point_max() -> int: ...


@register_extern("point-min")
def point_min() -> int: ...


@register_extern("goto_char")
def goto_char(x: int) -> None: ...


@register_extern("buffer-size")
def buffer_size() -> int: ...


@register_extern("buffer-get")
def buffer_get(name: str) -> object: ...


@register_extern("current-buffer")
def current_buffer() -> object: ...


@register_extern("set-buffer")
def set_buffer(buf: object) -> None: ...


@register_extern("buffer-file-name")
def buffer_file_name(buf: object) -> str: ...


@register_extern("generate-new-buffer")
def generate_new_buffer(name: str) -> object: ...


@register_extern("get-buffer-create")
def get_buffer_create(name: str) -> object: ...

# TODO register function name mangle


@register_extern("kill-buffer")
def kill_buffer_by_name(x: str) -> None: ...


@register_extern("kill-buffer")
def kill_buffer_by_buffer(x: object) -> None: ...
