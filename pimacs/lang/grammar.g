?start: file_input

single_input: _NEWLINE | statement | _NEWLINE
file_input: (_NEWLINE | statement)*

?statement: var_decl
          | let_decl
          | func_def
          | class_def
          | decorated
          | if_stmt
          | while_loop
          | for_loop
          | elisp_call
          | func_call
          | return_stmt
          | assign_stmt
          | guard_stmt

var_decl: VAR NAME [":" type] ["=" expr]
let_decl: LET NAME [":" type] ["=" expr]

if_stmt: IF expr ":" block elif_block* else_block?
elif_block: ELIF expr ":" block
else_block: ELSE ":" block

assign_stmt: dotted_name "=" expr

while_loop: "while" expr ":" block

for_loop: "for" NAME "in" expr ":" block

?decorated: decorator+ (class_def | func_def | var_decl | let_decl)
decorator: "@" (dotted_name | func_call | TEMPLATE type_placeholder_list) _NEWLINE
?dotted_name: NAME ("." NAME)*
?lisp_name: "%" LISP_NAME

// templated_func_def: [template] func_def
// template: "@template" type_placeholder_list _NEWLINE
type_placeholder_list: "[" type_placeholders "]"
type_placeholders: NAME ("," NAME)*

// function related
func_def: "def" NAME "(" [func_args] ")" ["->" type] ":" block
func_args: func_arg ("," func_arg)*
func_arg: NAME [":" type] ["=" expr]

// class related
class_def: "class" NAME ":" class_body
class_body: _NEWLINE _INDENT [doc_string] (_NEWLINE | statement)+ _DEDENT

// guard
guard_stmt: "guard" func_call ":" block

doc_string: STRING

// elisp mixing
elisp_call: "%" "(" elisp_expr ")"
          | "with" elisp_special ":" block
elisp_special: "temp_buffer" | "provide" | "defcustom" | "defvar"
elisp_expr: /.+/

block: _NEWLINE _INDENT [doc_string] (_NEWLINE | statement)+ _DEDENT

expr: atom
    | expr "+" expr       -> add
    | expr "-" expr       -> sub
    | expr "*" expr       -> mul
    | expr "/" expr       -> div
    | expr "==" expr      -> eq
    | expr "!=" expr      -> ne
    | expr ">" expr       -> gt
    | expr ">=" expr      -> ge
    | expr "<" expr       -> lt
    | expr "<=" expr      -> le
    | "(" expr ")"
    | expr "if" expr "else" expr -> select_expr
    | NOT expr            -> not_cond

atom: NUMBER                   -> number
    | STRING                   -> string
    | true | false             -> bool
    | dotted_name              -> variable
    | lisp_name                -> lisp_symbol
    | nil
    | dict
    | list
    | func_call

true: "true"
false: "false"
nil: "nil"

dict: "{"  [pair_list]  "}"
list: "["  [expr_list]  "]"

expr_list: expr (_NEWLINE | "," [_NEWLINE] expr)* ["," [_NEWLINE]]


func_call: dotted_name type_spec? "(" [call_params] ")"
call_param: expr                -> value_param
          | call_param_name "=" expr       -> key_value_param
call_params: call_param ("," call_param)*
call_param_name: NAME

return_stmt: RETURN [expr]

pair: expr ":" expr
pair_list: pair (_NEWLINE | "," [_NEWLINE] pair)* ["," [_NEWLINE]]


type: type_base [QUESTION]
    | variadic_type

type_base: PRIMITIVE_TYPE
         | custom_type
         | complex_type
         | list_type
         | dict_type
         | set_type

variadic_type: type ELLIPSIS
basic_type: PRIMITIVE_TYPE | custom_type
complex_type: NAME "[" type_list "]"
list_type: "[" type "]"
dict_type: "{" type ":" type "}"
set_type: "{" type "}"
type_spec: "[" type_list "]"

custom_type: NAME  // Allows for user-defined types, including generics

type_list: type ("," type)*

PRIMITIVE_TYPE: "Int" | "Float" | "Str" | "Bool" | "Dict" | "List" | "Set"

NAME: /%?[a-zA-Z_\-\/\+\-][a-zA-Z0-9_\-\/\+\-]*/
LISP_NAME: (":" | "-" | "_" | LETTER | "!" | "?" | "*" | "+" | "/" | "<" | "=" | ">" | "&" | "%" | "$" | "#" | "@") ("-" | "_" | LETTER | DIGIT | "!" | "?" | "*" | "+" | "/" | "<" | "=" | ">" | "&" | "%" | "$" | "#" | "@" | ".")*
NUMBER: /-?\d+(\.\d+)?/
STRING: /"(?:\\.|[^"\\])*"/
RETURN: "return"
LET: "let"
VAR: "var"
NOT: "not"
IF: "if"
ELIF: "elif"
ELSE: "else"
TEMPLATE: "template"
QUESTION: "?"
ELLIPSIS: "..."

%import common.WS_INLINE
%import common.LETTER
%import common.DIGIT

%ignore COMMENT  // Tells the parser to ignore anything matched by the COMMENT rule

COMMENT: /#[^\n]*/
_NEWLINE: (/\r?\n[\t ]*/ | COMMENT)+

//%ignore NEWLINE

%declare _INDENT _DEDENT
%ignore WS_INLINE

%ignore /[\t \f]+/  // WS
%ignore /\\[\t \f]*\r?\n/   // LINE_CONT
%ignore COMMENT
