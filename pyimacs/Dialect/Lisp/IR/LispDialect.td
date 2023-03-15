#ifndef PYIMACS_LISP_DIALECT
#define PYIMACS_LISP_DIALECT

include "mlir/IR/OpBase.td"

def Lisp_Dialect : Dialect {
    let name = "lisp";

    let cppNamespace = "::mlir::pyimacs";

    let summary = "The PyIMacs LISP dialect.";

    let description = [{ }];

    let dependentDialects = [
        "arith::ArithDialect",
        "math::MathDialect",
        "scf::SCFDialect",
        "LLVM::LLVMDialect",
        "func::FuncDialect",
    ];

    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let useDefaultTypePrinterParser = 1;
}

include "pyimacs/Dialect/Lisp/IR/LispTypes.td"

#endif  // Lisp_Dialect
