#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"
#include "pyimacs/Dialect/Lisp/IR/Dialect.h.inc"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::pyimacs::LispDialect, mlir::math::MathDialect,
                  mlir::arith::ArithmeticDialect, mlir::StandardOpsDialect,
                  mlir::scf::SCFDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "PyIMacs optimizer driver\n", registry));
}
