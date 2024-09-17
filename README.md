# Pimacs

Pimacs is a transpiler and programming language designed to facilitate the translation of programs written in Pimacs into Emacs lisp. This project aims to bridge the gap between the structured and object-oriented paradigms of Pimacs and macro-driven, functional programming of Emacs lisp.

## Project Structure

The project is structured as follows:

- `pimacs/`: The main package of Pimacs.
  - `ast/`: The AST of Pimacs.
  - `builtin/`: The built-in Pimacs libraries.
  - `codegen/`: The Emacs Lisp code generator.
  - `lisp/`: The AST for Emacs Lisp.
  - `sema/`: The semantic analyzer of Pimacs.
- `tests/`: The unittests for Pimacs.

## Pimacs Syntaxs
We borrowed the syntax of Python, Swift and Emacs Lisp to design the syntax of Pimacs. Pimacs is a statically-typed language, and the type of a variable is inferred from its value.

The following is a list of the syntax of Pimacs:

### Declare a variable
```python
# Declare a variable with an inferred type of int
# `var` keyword is used to declare a variable
var x = 1

# Declare a variable with an explicit type of int
var x: Int
x = 1

# Declare a constant with an inferred type of int
let x = 1
# A constant is a variable that cannot be reassigned
```

### Reference symbols from Emacs Lisp
As a transpiler, Pimacs allows you to reference symbols from Emacs Lisp. A symbol can be referenced from Emacs Lisp by using the `%` symbol, it works for both functions and variables. The symbol will be translated to the corresponding Emacs Lisp code.

The syntax is as follows:
```python
# Will be translated to `(message "Hello, world!")`
%message("Hello, world!")

# The return type of all the embedded Emacs Lisp functions is `Lisp`,
# a type similar to `any` in other programming languages,
# it can be used in Pimacs program.
var my-buffer-name = %buffer-name()
```

### Conditional statements

If statements in Pimacs are similar to those in Python. The syntax is as follows:
```python
if x > 0:
    print("x is positive")
elif x < 0:
    print("x is negative")
else:
    print("x is zero")
```

### Declare a function
```python
# Note, all the arguments and return types of a function must be explicitly declared
def foo(x: Int, y: Int) -> Int:
    return x + y

# Declare a templated function
def foo[T](x: T, y: T) -> T:
    return x
```

### Declare a class
```python
# Pimacs class will be translated to cl-defstruct in Emacs Lisp
class Point:
    var x: Int
    var y: Int

    # The constructor is similar to Python's.
    def __init__(self, x: Int, y: Int):
        self.x = x
        self.y = y

    def move(self, dx: Int, dy: Int):
        self.x += dx
        self.y += dy

# Declare an instance.
var p = Point(1, 2)
```

Separate Constructor to power OOP wrapper for existing Emacs Lisp functions.
```python
class hash-table[K, V]:

    def __getitem__(self, key: K) -> V:
        return %gethash(key, self)

    def __setitem__(self, key: K, value: V):
        %puthash(key, value, self)

    def __contains__(self, key: K) -> bool:
        return %not(%eq(%gethash(key, self, %'no-value), %'no-value))

def hash-table[K, V]() -> hash-table[K, V]:
    return %make-hash-table(%:test, %'equal)
```

### Import a module

The `import` keyword is similar to Python's, and it is used to import modules in Pimacs. The syntax is as follows:

```python
# There is a a-module.pim in the same directory as the current file
import a-module

a-module.foo()
```

Import multiple symbols at once:

```python
from a-module import foo, gloo

foo()
gloo()
```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

### Project status
This project is still in the early stages of development. The following features are planned for the first beta release:

1. Core builtin Pimacs modules, such as `hashtable`, `plist`, `set` and so on.
2. `any` data type.
3. Documentation and examples.

### Running the tests
```sh
pip install -r requirements.txt
export PYTHONPATH=$PWD
cd tests
pytest -vv
```
