#ifndef PYIMACS_LISP_TYPES
#define PYIMACS_LISP_TYPES

include "pyimacs/Dialect/Lisp/IR/LispDialect.td"

class LispTypeDef<string name, string _mnemonic>
        : TypeDef<Lisp_Dialect, name> {
    let mnemonic = _mnemonic;
}

def String : LispTypeDef<"String", "string">;
def Int : AnyTypeOf<[I32, I64], "Int", "int">;
def Float : AnyTypeOf<[F32, F64], "Float", "float">;
def Symbol : LispTypeDef<"Symbol", "symbol">;
def Object : LispTypeDef<"Object", "object">;
def Value : AnyTypeOf<[String, Int, Float, Symbol, Object], "value">;

#endif  // LispTypes
