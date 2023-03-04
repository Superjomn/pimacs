# AOT Compilation Design
This doc mainly focus on the user API of `@aot` decorator.

The python code should be something like

```python
@aot
def some_func(a:int) -> int: ...
```

The `@aot` should help to translate this function into a elisp function, and insert into a elisp module in disk, that should be the generic AOT compilation works.

Something need to mesion is that, the `@aot` may be also possible to work on classes, e.g.

## data struct
The `struct` is essential for OOP-style elisp programming.

### The related elisp functions

```python
@register_extern("cl-defstruct")
def def_struct(fields:List[str]) -> None

@register_extern("pyimacs-makestruct")
def make_struct(struct:str, args: pyimacs.list) -> object

@register_extern("pyimacs-instance")
def is_instance(o:object, struct:str) -> bool

@register_extern("pyimacs-get-field")
def get_field(o:object, field:str) -> object
```

The elisp functions with name prefix with `pyimacs-` are pyimacs-builtin elisp functions.

### Struct API

This should be an extension.

```python
class Struct(Ext):
    counter:int = 0
    names = set()

    def __init__(self, fields: List[str], name_hint=""):
        self.name = name_hint if name_hint else "struct"
        if self.name in Struct.names:
            self.name += "-%d" % Struct.counter
            Struct.counter += 1

        def_struct(fields) # this need a lisp op
        self.fields = fields

    def create(self, **kwargs) -> object:
        kvs = zip(kwargs.keys(), kwargs.values())
        handle = make_struct(self.name, kvs)
        return Field(handle, self.fields)


@dataclass
class Field(Ext):
    handle: object
    fields: List[str]

    def __getattr__(self, field:str):
        ''' Catch the field access '''
        assert field in self.fields
        return get_field(self.handle, field)
```

With the code above, we could use the `Struct` natively in python code

```python
Person = Struct(["name", "sex"])
jojo = Person.create(name="JoJo", sex="male")
jojo.name
```


### AOT on class
```python
@aot
class SomePerson:
    name: str
    age: int

    def get_name(self) -> str: ...
```
The `@aot` should help parse the definition class, and does the following things

1. Get all the fields and generate some elisp code to define a struct for this class using `cl-defstruct`, so that this class could be combined with other data structures.
2. aot compile all the other methods, and replace the `self` argument to the struct generated in the first step

The `@aot` might alter the python code to make a better user interface, the code after the decoration might be

```python
# cl-defstruct a struct named SomePerson_Struct
SomePerson_Struct = Struct(["name", "age"])

class SomePerson(Ext):
    name: str
    age: int

    def __init__(self, name, age):
        self.data = SomePerson_Struct.create(name=name, age=age)

    def get_info(self, other:str):
        return self.data.name + str(self.data.age) + other
```
