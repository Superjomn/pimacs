# This file contains code snippets for testing the parser

class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value


snippets = AttrDict()

snippets.var_case = '''
var a :Int
var b :Int = 1
var c = 1

var d :Float
var e :Float = 1.0
var f = 1.0
'''

snippets.func_case = '''
def hello-0 (name: Str) -> nil:
    var a = "Hello " + name
    print("hello %s", a)

def fib(n: Int) -> Int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
'''

snippets.decorator_case = '''
@interactive
def hello(name:Str) -> nil:
    print("Hello %s", name)
'''

snippets.decorator_case1 = '''
@some-decorator(100, 200)
@interactive("P:")
def hello(name:Str) -> nil:
    print("Hello %s", name)
'''


snippets.func_with_docstring_case = '''
def hello(name:Str) -> nil:
    "Some docs"
    return
'''

snippets.class_case = '''
class Person:
    var name: Str
    var age: Int

    def __init__(self, name: Str, age: Int) -> nil:
        self.name = name
        self.age = age

    def get-name(self) -> Str:
        return self.name

    def get-age(self) -> Int:
        return self.age
'''

snippets.lisp_symbol_case = '''
var a :Lisp = %org-mode
'''
