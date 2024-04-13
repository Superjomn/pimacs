?start: file_input

single_input: _NEWLINE | statement | _NEWLINE
file_input: (_NEWLINE | statement)*

?statement: var_decl
            | func_def
            | class_def
            | if_stmt
            | while_loop
            | for_loop
            | elisp_call
            | func_call
            | "pass" -> pass
            | return_stmt

var_decl: "var" NAME "=" expr
        | "var" NAME ":" TYPE "=" expr

if_stmt: "if" expr ":" block elif_block* else_block?
elif_block: "elif" expr ":" block
else_block: "else" ":" block

while_loop: "while" expr ":" block

for_loop: "for" NAME "in" expr ":" block

// function related
func_def: "def" NAME "(" [func_params] ")" ["->" TYPE] ":" block
func_params: func_param ("," func_param)*
func_param: NAME [":" TYPE] ["=" expr]

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
    | expr "-" expr       -> subtract
    | expr "*" expr       -> multiply
    | expr "/" expr       -> divide
    | expr "==" expr      -> eq
    | expr "!=" expr      -> neq
    | "(" expr ")"

atom: NUMBER                   -> number
    | STRING                   -> string
    | "true" | "false"         -> bool
    | "nil"                    -> nil
    | NAME                     -> variable
    | "[" [expr ("," expr)*] "]" -> list
    | "{" [pair ("," pair)*] "}" -> dict
    | func_call


func_call: NAME "(" args* ")"
arg: expr | NAME "=" expr
args: arg ("," arg)*

return_stmt: "return" [expr]


pair: expr ":" expr

TYPE: "Int" | "Float" | "Str" | "Bool" | "Dict" | "List" | "Set" | "nil" | NAME
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
