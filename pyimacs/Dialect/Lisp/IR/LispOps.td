#ifndef PYIMACS_LISP_OPS
#define PYIMACS_LISP_OPS

include "pyimacs/Dialect/Lisp/IR/LispDialect.td"
include "pyimacs/Dialect/Lisp/IR/LispAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // NoSideEffect
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td" // NoSideEffect

class Lisp_Op<string mnemonic, list<Trait> traits=[]> :
    Op<Lisp_Dialect, mnemonic, traits> {
}

def Lisp_ListOp : Lisp_Op<"list"> {
  let summary = "list op";
  let description = [{ }];
  let arguments = (ins StrAttr:$symbol);
  let results = (outs Value:$output);
  let regions = (region SizedRegion<1>:$body);
}

def Lisp_ReturnOp : Lisp_Op<"ret", [HasParent<"ListOp">, Terminator, SameOperandsAndResultType]> {
    let summary = "return operation";
    let description = [{ }];
    let arguments = (ins Value:$input);
    let results = (outs Value:$output);

    //let parser = [{ return mlir::pyimacs::parseReturnOp(parser, result); }];
    //let printer = [{ return mlir::pyimacs::printReturnOp(p, *this); }];
}

#endif
