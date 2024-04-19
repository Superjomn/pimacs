# This file contains code snippets for testing the parser

var_case = '''
var a :Int
var b :Int = 1
var c = 1

var d :Float
var e :Float = 1.0
var f = 1.0
'''

func_case = '''
def hello-0 (name: Str) -> nil:
    var a = "Hello " + name
    print("hello %s", a)

def fib(n: Int) -> Int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
'''

decorator_case = '''
@interactive
def hello(name:Str) -> nil:
    print("Hello %s", name)
'''

decorator_case1 = '''
@some-decorator(100, 200)
@interactive("P:")
def hello(name:Str) -> nil:
    print("Hello %s", name)
'''

class_case = '''
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

lisp_symbol_case = '''
var a :Lisp = %org-mode
'''
