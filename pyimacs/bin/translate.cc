#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "pyimacs/Dialect/Lisp/IR/Dialect.h"
#include "pyimacs/Target/elisp/translateToElisp.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <iostream>

namespace mlir {
namespace pyimacs {

OwningOpRef<ModuleOp> loadMLIRModule(llvm::StringRef inputFilename,
                                     MLIRContext &context) {
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  mlir::DialectRegistry registry;
  registry
      .insert<LispDialect, mlir::math::MathDialect, arith::ArithmeticDialect,
              StandardOpsDialect, scf::SCFDialect>();

  context.appendDialectRegistry(registry);

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer)
      -> OwningOpRef<ModuleOp> {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

    context.loadAllAvailableDialects();
    context.allowUnregisteredDialects();

    OwningOpRef<ModuleOp> module(parseSourceFile(sourceMgr, &context));
    if (!module) {
      llvm::errs() << "Parse MLIR file failed.";
      return nullptr;
    }

    return module;
  };

  auto module = processBuffer(std::move(input));
  if (!module) {
    return nullptr;
  }

  return module;
}

LogicalResult pyimacsTranslateMain(int argc, char **argv,
                                   llvm::StringRef toolName) {
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::InitLLVM y(argc, argv);

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);

  mlir::MLIRContext context;
  auto module = loadMLIRModule(inputFilename, context);
  if (!module) {
    return failure();
  }

  /*
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
   */

  std::string buf;
  llvm::raw_string_ostream os(buf);
  if (failed(translateToElisp(*module, os))) {
    llvm::errs() << "failed in translation";
    return failure();
  }
  os.flush();
  llvm::outs() << buf << "\n";

  return success();
}

} // namespace pyimacs
} // namespace mlir

int main(int argc, char **argv) {
  return failed(mlir::pyimacs::pyimacsTranslateMain(
      argc, argv, "PyIMacs Translate Testing Tool."));
}
