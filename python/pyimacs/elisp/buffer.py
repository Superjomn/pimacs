from typing import *

from pyimacs.elisp.core import Guard
from pyimacs.lang.extension import *

import pyimacs


class Buffer(Ext):
    def __init__(self, buf_name: str = "", handle=None):
        self._handle = handle
        if buf_name:
            self._handle = _buffer_get(buf_name)

    @staticmethod
    def current():
        ''' Get the current buffer. '''
        return Buffer(handle=_current_buffer())

    @staticmethod
    def set_buffer(buf: "Buffer") -> None:
        _set_buffer(buf._handle)

    def kill(self) -> None:
        _kill_buffer_by_buffer(self._handle)

    @staticmethod
    def point_max(self) -> int:
        return _point_max()

    @staticmethod
    def point_min(self) -> int:
        return _point_min()

    def insert_buffer_substring(self, buf: "Buffer", start: int, end: int) -> None:
        _insert_buffer_substring(buf._handle, start, end)

    @staticmethod
    def kill_buffer(self, x: Any) -> None:
        if type(x) is str:
            _kill_buffer_by_name(x)
        elif type(x) is object:
            _kill_buffer_by_buffer(x)

    def get_content(self) -> str:
        return _buffer_content(self._handle)
        # with Guard("with-current-buffer", self._handle):
        # return _buffer_string()

    def get_name(self) -> str:
        return _buffer_file_name(self._handle)

    def insert(self, x: str) -> None:
        _insert(x)

    def __handle_return__(self):
        return self._handle


@register_extern("buffer-string")
def _buffer_string() -> str: ...


@pyimacs.aot
def _buffer_content(buf: object) -> str:
    with Guard("with-current-buffer", buf):
        return _buffer_string()


@register_extern("point")
def _point() -> int: ...


@register_extern("point-max")
def _point_max() -> int: ...


@register_extern("point-min")
def _point_min() -> int: ...


@register_extern("goto_char")
def _goto_char(x: int) -> None: ...


@register_extern("buffer-size")
def _buffer_size() -> int: ...


@register_extern("buffer-get")
def _buffer_get(name: str) -> object: ...


@register_extern("current-buffer")
def _current_buffer() -> object: ...


@register_extern("set-buffer")
def _set_buffer(buf: object) -> None: ...


@register_extern("buffer-file-name")
def _buffer_file_name(buf: object) -> str: ...


@register_extern("generate-new-buffer")
def _generate_new_buffer(name: str) -> object: ...


@register_extern("get-buffer-create")
def _get_buffer_create(name: str) -> object: ...

# TODO register function name mangle


@register_extern("kill-buffer")
def _kill_buffer_by_name(x: str) -> None: ...


@register_extern("kill-buffer")
def _kill_buffer_by_buffer(x: object) -> None: ...


@register_extern("insert-buffer-substring")
def _insert_buffer_substring(buf: object, start: int, end: int) -> None: ...


@register_extern("insert")
def _insert(buf: object, content: str) -> None: ...
