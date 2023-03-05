# Lisp Dialect Design

There are some elisp-specific data types and operations.

## Data type

- String
- Object, the Object is similar to `any` type

Currently, the `list` type is not introducted, we use the `Object` type instead if necessary.

## Operations
For tuple, we don't use the official `TupleType` since we prefer a more dynamic way to define functions.

- `make_tuple(Variaic<Any> args) -> Object`
- `tuple_get(o:Object, nth:int) -> Object`
