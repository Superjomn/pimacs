?start: file_input

single_input: _NEWLINE | statement | _NEWLINE
file_input: (_NEWLINE | statement)*

?statement: var_decl
            | func_def
            | class_def
            | decorated
            | if_stmt
            | while_loop
            | for_loop
            | elisp_call
            | func_call
            | "pass" -> pass
            | return_stmt

var_decl: "var" NAME [":" type] ["=" expr]
let_decl: "let" NAME [":" type] ["=" expr]

if_stmt: "if" expr ":" block elif_block* else_block?
elif_block: "elif" expr ":" block
else_block: "else" ":" block

assign_stmt: NAME "=" expr

while_loop: "while" expr ":" block

for_loop: "for" NAME "in" expr ":" block

?decorated: decorator+ (class_def | func_def)
decorator: "@" ((dotted_name ["(" [call_params] ")"]) | "template" type_placeholder_list) _NEWLINE
dotted_name: NAME ("." NAME)*

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
class_body: _INDENT statement+ _DEDENT

// elisp mixing
elisp_call: "%" "(" elisp_expr ")"
          | "with" elisp_special ":" block
elisp_special: "temp_buffer" | "provide" | "defcustom" | "defvar"
elisp_expr: /.+/

block: _NEWLINE _INDENT [STRING] (_NEWLINE | statement)+ _DEDENT

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

atom: NUMBER                   -> number
    | STRING                   -> string
    | true | false             -> bool
    | nil
    | NAME                     -> variable
    | dict
    | list
    | func_call

true: "true"
false: "false"
nil: "nil"

dict: "{"  [pair_list]  "}"
list: "["  [expr_list]  "]"

expr_list: expr (_NEWLINE | "," [_NEWLINE] expr)* ["," [_NEWLINE]]


func_call: func_call_name "(" [call_params] ")"
call_param: expr                -> value_param
          | call_param_name "=" expr       -> key_value_param
call_params: call_param ("," call_param)*
call_param_name: NAME
func_call_name: NAME

return_stmt: "return" [expr]


pair: expr ":" expr
pair_list: pair (_NEWLINE | "," [_NEWLINE] pair)* ["," [_NEWLINE]]


type: PRIMITIVE_TYPE
     | custom_type
     | complex_type
variadic_type: type "..."
basic_type: PRIMITIVE_TYPE | custom_type
complex_type: NAME "[" type_list "]"

custom_type: NAME  // Allows for user-defined types, including generics

type_list: type ("," type)*

PRIMITIVE_TYPE: "Int" | "Float" | "Str" | "Bool" | "Dict" | "List" | "Set"

NAME: /%?[a-zA-Z_\-\/\+\-][a-zA-Z0-9_\-\/\+\-]*/
NUMBER: /-?\d+(\.\d+)?/
STRING: /"(?:\\.|[^"\\])*"/

%import common.WS_INLINE

%ignore COMMENT  // Tells the parser to ignore anything matched by the COMMENT rule

COMMENT: /#[^\n]*/
_NEWLINE: (/\r?\n[\t ]*/ | COMMENT)+

//%ignore NEWLINE

%declare _INDENT _DEDENT
%ignore WS_INLINE

%ignore /[\t \f]+/  // WS
%ignore /\\[\t \f]*\r?\n/   // LINE_CONT
%ignore COMMENT
