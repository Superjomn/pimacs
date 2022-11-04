# pyimacs

A python DSL for generating emacs lisp. 

## Examples

```python
@pyimacs.lisp
def get_area(width:int, height:int):
    return width * height
    
pyimacs.compile(get_area)
```

Should get a lisp program:

```elisp
(defun get_area (width height)
    (* width height))
```

There are some builtin modules in pyimacs which ecapsulate the emacs lisp core libraries, one can use them like normal Python methods.

``` python
from pyimacs import builtin as pyb
@pyimacs.lisp
def get_buffer_size():
    return pyb.max_point() - pyb.min_point()
```

The python class could be leveraged to make the program more modular.

``` python
class Rectangle:
    @pyimacs.lisp
    def __init__(self, width:int, height:int) -> None:
        self.width:int = width
        self.height:int = height
    
    @pyimacs.lisp
    def get_area(self):
        return self.wisth * self.height
        
compile(Rectangle)
```

should get the following lisp code

``` emacs-lisp
(cl-defstruct Rectangle width height)

(defun make-Rectangle (width height))
  
(defun Rectangle-get_area(self)
    (* (Rectangle-width self)
       (Rectangle-height self)))
```







