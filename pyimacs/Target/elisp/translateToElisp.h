#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pyimacs {

// Translate the given operation to elisp code.
LogicalResult translateToElisp(Operation *op, llvm::raw_ostream &os);
LogicalResult translateToElisp(ModuleOp *op, llvm::raw_ostream &os);

} // namespace pyimacs
} // namespace mlir
