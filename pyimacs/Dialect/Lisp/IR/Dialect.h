#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "pyimacs/Dialect/Lisp/IR/Types.h"

#include "pyimacs/Dialect/Lisp/IR/Dialect.h.inc"
#define GET_OP_CLASSES
#include "pyimacs/Dialect/Lisp/IR/Ops.h.inc"
