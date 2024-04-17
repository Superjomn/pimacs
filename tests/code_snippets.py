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
