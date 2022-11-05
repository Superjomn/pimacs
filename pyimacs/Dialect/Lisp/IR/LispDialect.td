#ifndef PYIMACS_LISP_DIALECT
#define PYIMACS_LISP_DIALECT

include "mlir/IR/OpBase.td"

def Lisp_Dialect : Dialect {
    let name = "lisp";

    let cppNamespace = "::mlir::pyimacs";

    let summary = "The PyIMacs LISP dialect.";

    let description = [{ }];

    let dependentDialects = [ ];

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

include "pyimacs/Dialect/Lisp/IR/LispTypes.td"

#endif  // Lisp_Dialect
