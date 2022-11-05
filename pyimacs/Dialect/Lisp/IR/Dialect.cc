#include "pyimacs/Dialect/Lisp/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "pyimacs/Dialect/Lisp/IR/Types.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

// include Dialect implementation
#include "pyimacs/Dialect/Lisp/IR/Dialect.cpp.inc"

namespace mlir {
namespace pyimacs {

void LispDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pyimacs/Dialect/Lisp/IR/Ops.cpp.inc"
      >();

  registerTypes();
}

} // namespace pyimacs
} // namespace mlir

#define GET_OP_CLASSES
#include "pyimacs/Dialect/Lisp/IR/Ops.cpp.inc"
