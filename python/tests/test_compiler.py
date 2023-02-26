import pyimacs.lang as pyl
from pyimacs.elisp.buffer import Buffer
from pyimacs.runtime import jit

from pyimacs import compiler


def test_empty_kernel():
    @jit
    def some_fn(a: int):
        pass

    code = compiler.compile(some_fn, signature="i -> void")
    target = '''
(defun some_fn (arg0)
    (let*
        ()
    )
)
    '''
    assert code.strip() == target.strip()


def test_naive_kernel():
    @jit
    def some_fn(a: int):
        b = (a + 1) * 23
        return b + 1

    code = compiler.compile(some_fn, signature="i -> i")
    target = '''
(defun some_fn (arg0)
    (let*
        (arg1 arg2 arg3 arg4 arg5 arg6)
        (setq arg1 1)
        (setq arg2 (+ arg0 arg1))
        (setq arg3 23)
        (setq arg4 (* arg2 arg3))
        (setq arg5 1)
        (setq arg6 (+ arg4 arg5))
        arg6
    )
)
    '''
    assert code.strip() == target.strip()


def test_kernel_with_if():
    @jit
    def some_fn(a: int):
        a = a + 1
        if True:
            return a
        else:
            # NOTE: Currently, buggy with the return statements within ifOp, we could resolve it by adding a pass to move
            # the statements outside the IfOp(wth return) to else region.
            return a + 1

    code = compiler.compile(some_fn, signature="i -> i")
    print(code)

    target = '''
(defun some_fn (arg0)
    (let*
        (arg1 arg2 arg3)
        (setq arg1 1)
        (setq arg2 (+ arg0 arg1))
        (setq arg3 -1)
        (if arg3
            (let*
                ()
                arg2
            )

            (let*
                (arg4 arg5)
                (setq arg4 1)
                (setq arg5 (+ arg2 arg4))
                arg5
            )
        )
    )
)
    '''
    assert code.strip() == target.strip()


# def test_kernel_external_call():
#     @jit
#     def some_fn():
#         buffer = Buffer("*a-buffer*")
#         name = buffer.get_name()
#         return name

#     code = compiler.compile(some_fn, signature="void -> s")
#     print(code)
