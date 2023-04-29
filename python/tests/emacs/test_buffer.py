'''
This file would be translated into a elisp script and run all the tests in emacs.
'''

from pyimacs.aot import AOTFunction
from pyimacs.elisp.buffer import Buffer
from pyimacs.elisp.core import Guard

import pyimacs


# TODO Add side-effect
@pyimacs.aot
def _test_buffer_basic() -> str:
    some_buffer = Buffer("test")
    with Guard("with-current-buffer", some_buffer):
        some_buffer.insert("hello world")
        assert some_buffer.get_content() == "hello world"


@pyimacs.aot
def main():
    return _test_buffer_basic()


def test_buffer():
    code = AOTFunction.to_lispcode()
    print(code)
    assert code.strip() == \
        '''
(defun _buffer_content (arg0)
    (let*
        ()
        (with-current-buffer arg0
            (let*
                ()
                (buffer-string)
            )
        )
    )
)


(defun _test_buffer_basic ()
    (let*
        (arg1)
        (setq arg1 (buffer-get "test"))
        (with-current-buffer arg1
            (let*
                ()
                (insert "hello world")
                (cl-assert (string= (_buffer_content arg1) "hello world") "")
            )
        )
    )
)


(defun main ()
    (let*
        ()
        (_test_buffer_basic)
    )
)
'''.strip()
