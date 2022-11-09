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

template <typename T>
LogicalResult interleaveWithError(ArrayRef<T> arr, StringRef delim,
                                  raw_indented_ostream &os,
                                  std::function<LogicalResult(const T &)> fn) {
  for (int i = 0; !arr.empty() && i < arr.size() - 1; ++i) {
    if (failed(fn(arr[i])))
      return failure();
    os << delim;
  }
  if (!arr.empty())
    return fn(arr.back());
  return success();
}

struct ElispEmitter {
  explicit ElispEmitter(raw_ostream &os) : os(os) {
    valueInScopeCount.push(0);
    labelInScopeCount.push(0);
  }

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

  LogicalResult emitVariableDeclaration(OpResult result);
  LogicalResult emitVariableAssignment(OpResult result);
  LogicalResult emitOperands(Operation &op);
  LogicalResult emitOperation(Operation &op);
  LogicalResult emitAttribute(Attribute attr, Location loc);

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

LogicalResult ElispEmitter::emitVariableDeclaration(mlir::OpResult result) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  os << getOrCreateName(result);
  return success();
}

LogicalResult ElispEmitter::emitVariableAssignment(mlir::OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << "(setq " << getOrCreateName(result) << " ";
  return success();
}

LogicalResult ElispEmitter::emitAttribute(mlir::Attribute attr, Location loc) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "t";
      else
        os << "nil";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      os << strValue;
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  if (auto fattr = attr.dyn_cast<FloatAttr>()) {
    printFloat(fattr.getValue());
    return success();
  }

  if (auto iattr = attr.dyn_cast<IntegerAttr>()) {
    printInt(iattr.getValue(), false);
    return success();
  }

  return emitError(loc, "cannot emit attribute of type ") << attr.getType();
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
  if (failed(emitter.emitVariableAssignment(op->getResult(0))))
    return failure();

  os << "(";
  os << op.callee();
  if (!op.getOperands().empty())
    os << " ";
  if (failed(emitter.emitOperands(*op.getOperation())))
    return failure();
  os << ")";

  os << ")"; // end assign

  return success();
}

static LogicalResult printOperation(ElispEmitter &emitter, FuncOp funcOp) {
  ElispEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  os << "(defun " << funcOp.getName() << " ";
  // arg list
  os << "("; // args begin
  if (failed(interleaveWithError<BlockArgument>(
          funcOp.getArguments(), " ", os,
          [&](const BlockArgument &arg) -> LogicalResult {
            os << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ")\n"; // args end

  os.indent();

  // blocks
  // Declare all variables that hold op results including those from nested
  // regions.
  os << "(let* (";
  WalkResult result =
      funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        for (OpResult result : op->getResults()) {
          if (failed(emitter.emitVariableDeclaration(result)))
            return WalkResult(
                op->emitError("unable to declare result variable for op"));
          os << " ";
        }
        return WalkResult::advance();
      });

  if (result.wasInterrupted())
    return failure();
  Region::BlockListType &blocks = funcOp.getBlocks();
  assert(blocks.size() == 1UL);

  os << ")\n"; // end of let arg list
  os.indent();
  // let main block

  // emit blocks
  auto &block = blocks.front();

  for (Operation &op : block.getOperations()) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }

  os.unindent();
  os << ")"; // end of let main block
  os.unindent();
  os << ")\n"; // end of defun
  return success();
}

static LogicalResult printOperation(ElispEmitter &emitter, mlir::ReturnOp op) {
  assert(op->getNumOperands() <= 1);
  auto &os = emitter.ostream();
  if (op->getNumOperands() == 1) {
    if (!emitter.hasValueInScope(op->getOperand(0)))
      return op->emitError("Returns something not defined.");
    os << "\n" << emitter.getOrCreateName(op->getOperand(0));
  } else {
    // do nothing.
  }
  return success();
}

static LogicalResult printOperation(ElispEmitter &emitter,
                                    arith::ConstantOp op) {
  auto &os = emitter.ostream();
  if (!emitter.hasValueInScope(op.getResult())) {
    return op->emitError("Constant variable not defined.");
  }

  if (failed(emitter.emitVariableAssignment(op->getResult(0))))
    return failure();
  if (failed(emitter.emitAttribute(op.getValue(), op.getLoc())))
    return failure();
  os << ")\n";
  return success();
}

static LogicalResult printOperation(ElispEmitter &emitter, ModuleOp moduleOp) {
  ElispEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  return success();
}

LogicalResult ElispEmitter::emitOperation(mlir::Operation &op) {
  llvm::outs() << "op: " << op << "\n";
  LogicalResult result =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<CallOp, arith::ConstantOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<mlir::ReturnOp, mlir::ModuleOp, mlir::FuncOp>(
              [&](auto op) { return printOperation(*this, op); });

  return result;
}

LogicalResult translateToElisp(Operation *op, llvm::raw_ostream &os) {
  ElispEmitter emitter(os);
  return emitter.emitOperation(*op);
}

} // namespace pyimacs
} // namespace mlir
