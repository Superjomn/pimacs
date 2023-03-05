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

// Tuple is not an elisp concept, left here temporarily.
def Lisp_MakeTupleOp : Lisp_Op<"make_tuple"> {
    let summary = "make_tuple op";
    let description = [{ Make a tuple. }];
    let arguments = (ins Variadic<Value>:$args);
    let results = (outs Object:$res);
}

def Lisp_MakeSymbolOp : Lisp_Op<"make_symbol"> {
    let summary = "make_symbol op";
    let description = [{ Make a symbol. }];
    let arguments = (ins String:$name, BoolAttr:$is_keyword);
    let results = (outs Object:$res);
}

// def Lisp_ListOp : Lisp_Op<"list"> {
//   let summary = "list op";
//   let description = [{ }];
//   let arguments = (ins StrAttr:$symbol);
//   let results = (outs Value:$output);
//   let regions = (region SizedRegion<1>:$body);
// }

def Lisp_CallOp : Lisp_Op<"call"> {
    let summary = "call op";
    let arguments = (ins StrAttr:$callee, Variadic<Value>:$inputs);
    let results = (outs Value:$output);
}

// def Lisp_ConstantOp : Lisp_Op<"constant"> {
//     let summary = "constant op";
//     let description = [{
//         Create a constant.
//     }];
//     let arguments = (ins AnyAttr:$value);
//     let results = (outs Value:$output);
// }

def Lisp_GetNullOp: Lisp_Op<"get_null"> {
  let summary = "get null op";
  let description = [{
    Get NULL op.
  }];
  let arguments = (ins);
  let results = (outs Value:$output);
}

//def List_MakeSymbolOp : Lisp_Op<"make_symbol"> {
    ///let summary = "Create a symbol";
    ///let arguments = (ins StrAttr:$name);
    ///let results = (outs Symbol:$symbol);
//}

// def Lisp_ReturnOp : Lisp_Op<"ret", [HasParent<"ListOp">, Terminator, SameOperandsAndResultType]> {
//     let summary = "return operation";
//     let description = [{ }];
//     let arguments = (ins Value:$input);
//     let results = (outs Value:$output);

//     //let parser = [{ return mlir::pyimacs::parseReturnOp(parser, result); }];
//     //let printer = [{ return mlir::pyimacs::printReturnOp(p, *this); }];
// }

#endif
