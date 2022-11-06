#include "pyimacs/Target/elisp/translateToElisp.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "pyimacs/Dialect/Lisp/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace pyimacs {
using namespace llvm;

struct ElispEmitter {

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Return the existing or a new label for a Block.
  StringRef getOrCreateName(Block &block);

  struct Scope {
    Scope(ElispEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    ElispEmitter &emitter;
  };

  LogicalResult emitOperands(Operation &op);

  bool hasValueInScope(Value val) const { return valueMapper.count(val); }

  raw_indented_ostream &ostream() { return os; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};

/// Return the existing or a new name for a Value.
StringRef ElispEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef ElispEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

LogicalResult ElispEmitter::emitOperands(mlir::Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand not in scope";
    os << getOrCreateName(result);
    return success();
  };

  auto operands = op.getOperands();
  for (int i = 0; !operands.empty() && i < op.getOperands().size() - 1; i++) {
    if (emitOperandName(operands[i]).failed())
      return failure();
    os << " ";
  }
  if (!operands.empty())
    if (emitOperandName(operands.back()).failed())
      return failure();

  return success();
}

// Print ops

static LogicalResult printOperation(ElispEmitter &emitter, scf::IfOp op) {
  auto &os = emitter.ostream();

  return success();
}

static LogicalResult printOperation(ElispEmitter &emitter, scf::ForOp op) {
  return success();
}
static LogicalResult printOperation(ElispEmitter &emitter, pyimacs::CallOp op) {
  auto &os = emitter.ostream();
  os << "(";
  os << op.callee();
  if (!op.getOperands().empty())
    os << " ";
  if (failed(emitter.emitOperands(*op.getOperation())))
    return failure();
  os << ")";
  return success();
}

} // namespace pyimacs
} // namespace mlir
