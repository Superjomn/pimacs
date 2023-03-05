#pragma once

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "pyimacs/Dialect/Lisp/IR/Types.h"

#include "pyimacs/Dialect/Lisp/IR/Dialect.h.inc"
#define GET_OP_CLASSES
#include "pyimacs/Dialect/Lisp/IR/Ops.h.inc"
