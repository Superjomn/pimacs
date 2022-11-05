#include "pyimacs/Dialect/Lisp/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "pyimacs/Dialect/Lisp/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

#define GET_TYPEDEF_CLASSES
#include "pyimacs/Dialect/Lisp/IR/Types.cpp.inc"

using namespace mlir;
using namespace mlir::pyimacs;

void LispDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "pyimacs/Dialect/Lisp/IR/Types.cpp.inc"
      >();
}
