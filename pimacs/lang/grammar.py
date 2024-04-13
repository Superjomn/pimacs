import os

from lark import Lark
from lark.indenter import PythonIndenter

dsl_grammar = open(os.path.join(os.path.dirname(__file__), 'grammar.g')).read()

parser = Lark(dsl_grammar, parser='lalr', postlex=PythonIndenter())
