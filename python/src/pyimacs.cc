#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "pyimacs/Dialect/Lisp/IR/Dialect.h"
#include "pyimacs/Dialect/Lisp/IR/Types.h"
//#include "pyimacs/Target/elisp/translateToElisp.h"

#include <Python.h>
#include <cctype>
#include <fstream>
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "pyimacs/Dialect/Lisp/IR/Dialect.h"

namespace py = pybind11;
namespace pyimacs {
using ret = py::return_value_policy;
using namespace pybind11::literals;
// Borrowed much code from OpenAI/triton triton.cc

template <typename T> std::string toStr(const T &v) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << v;
  os.flush();
  return str;
}

void initMLIR(py::module &m) {
  py::class_<mlir::MLIRContext>(m, "MLIRContext")
      .def(py::init<>())
      .def("load_pyimacs", [](mlir::MLIRContext &self) {
        self.getOrLoadDialect<mlir::pyimacs::LispDialect>();
        self.loadAllAvailableDialects();
      });

  py::class_<mlir::Type>(m, "Type")
      .def("is_integer", &mlir::Type::isInteger)
      .def("is_int",
           [](mlir::Type &self) {
             return self.isInteger(32) || self.isInteger(64);
           })
      .def(
          "is_float",
          [](mlir::Type &self) -> bool { return self.isF16() || self.isF32(); })
      .def("is_string",
           [](mlir::Type &self) {
             return self.isa<mlir::pyimacs::StringType>();
           })
      .def("is_object",
           [](mlir::Type &self) -> bool {
             return self.isa<mlir::pyimacs::ObjectType>();
           })
      .def("__str__", &toStr<mlir::Type>);

  py::class_<mlir::Value>(m, "Value")
      .def("set_attr",
           [](mlir::Value &self, const std::string &name,
              mlir::Attribute &attr) -> void {
             if (mlir::Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               /* issue an warning */
             }
           })
      .def("replace_all_uses_with",
           [](mlir::Value &self, mlir::Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })

      .def("get_type", &mlir::Value::getType)
      .def("__hash__",
           [](mlir::Value &self) -> size_t {
             return std::hash<std::string>{}(toStr(self));
           })
      .def("__str__",
           [](mlir::Value &self) -> std::string { return toStr(self); })
      .def("__repr__", [](mlir::Value &self) -> std::string {
        return "<Value " +
               std::to_string(std::hash<std::string>{}(toStr(self))) + ">";
      });

  py::class_<mlir::BlockArgument, mlir::Value>(m, "BlockArgument");

  py::class_<mlir::Region>(m, "Region")
      .def(py::init<>())
      .def("get_parent_region", &mlir::Region::getParentRegion, ret::reference)
      .def("size", [](mlir::Region &self) { return self.getBlocks().size(); })
      .def(
          "blocks",
          [](mlir::Region &self, int idx) -> mlir::Block * {
            mlir::Block *ret{};
            for (auto &b : self.getBlocks()) {
              ret = &b;
              if (idx-- == 0)
                break;
            }
            return ret;
          },
          ret::reference)
      .def("empty", &mlir::Region::empty);

  py::class_<mlir::Block>(m, "Block")
      .def("arg",
           [](mlir::Block &self, int index) -> mlir::BlockArgument {
             return self.getArgument(index);
           })
      .def("get_num_arguments", &mlir::Block::getNumArguments)
      .def("dump", &mlir::Block::dump)
      .def("move_before", &mlir::Block::moveBefore)
      .def("insert_before", &mlir::Block::insertBefore)
      .def("get_parent", &mlir::Block::getParent, ret::reference)
      .def("merge_block_before",
           [](mlir::Block &self, mlir::Block &dst) {
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](mlir::Block &self, mlir::Value &v, mlir::Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](mlir::OpOperand &operand) {
               mlir::Operation *user = operand.getOwner();
               mlir::Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock =
                     currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("operations",
           [](mlir::Block &self) -> std::vector<const mlir::Operation *> {
             std::vector<const mlir::Operation *> ret;
             for (auto &op : self.getOperations()) {
               ret.push_back(&op);
             }
             return ret;
           });

  py::class_<mlir::Attribute>(m, "Attribute")
      .def("is_int",
           [](mlir::Attribute &self) { return self.isa<mlir::IntegerAttr>(); })
      .def("is_float",
           [](mlir::Attribute &self) { return self.isa<mlir::FloatAttr>(); })
      .def("is_string",
           [](mlir::Attribute &self) { return self.isa<mlir::StringAttr>(); })
      .def("is_bool",
           [](mlir::Attribute &self) { return self.isa<mlir::BoolAttr>(); })
      .def("to_int",
           [](mlir::Attribute &self) -> int64_t {
             return self.cast<mlir::IntegerAttr>().getInt();
           })
      .def("to_float",
           [](mlir::Attribute &self) -> float {
             return self.cast<mlir::FloatAttr>().getValueAsDouble();
           })
      .def("to_bool",
           [](mlir::Attribute &self) -> bool {
             return self.cast<mlir::BoolAttr>().getValue();
           })
      .def("to_string", [](mlir::Attribute &self) -> std::string {
        return self.cast<mlir::StringAttr>().getValue().str();
      });

  py::class_<mlir::IntegerAttr, mlir::Attribute>(m, "IntegerAttr");
  py::class_<mlir::BoolAttr, mlir::Attribute>(m, "BoolAttr");

  // Ops
  py::class_<mlir::OpState>(m, "OpState")
      .def("set_attr",
           [](mlir::OpState &self, std::string &name,
              mlir::Attribute &attr) -> void { self->setAttr(name, attr); })
      .def(
          "get_num_results",
          [](mlir::OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](mlir::OpState &self, unsigned idx) -> mlir::Value {
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](mlir::OpState &self, unsigned idx) -> mlir::Region & {
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](mlir::scf::ForOp &self, unsigned idx) -> mlir::Block * {
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](mlir::OpState &self) { self->dump(); })
      .def("__str__",
           [](mlir::OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self->print(os);
             return str;
           })
      .def("append_operand",
           [](mlir::OpState &self, mlir::Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](mlir::OpState &self) -> bool {
        return mlir::succeeded(mlir::verify(self.getOperation()));
      });

  // py::class_<mlir::OpOperand>(m, "OpOperand")
  //     .def("__len__",
  //          [](mlir::OpOperand &self) -> int { return self.getOperandNumber();
  //          })
  //     .def("__getitem__", [](mlir::Operation &self, int i) -> mlir::Value {
  //       return self.getOperand(i);
  //     });

  py::class_<mlir::OpResult, mlir::Value>(m, "OpResult")
      .def("result_number",
           [](mlir::OpResult &self) { return self.getResultNumber(); });

  py::class_<mlir::OpOperand>(m, "OpOperand")
      .def("operand_number",
           [](mlir::OpOperand &self) { self.getOperandNumber(); })
      .def("get",
           [](mlir::OpOperand &self) -> mlir::Value { return self.get(); })
      .def("__len__", [](mlir::OpOperand &self) -> int {
        return self.getOperandNumber();
      });

  py::class_<mlir::Operation, std::unique_ptr<mlir::Operation, py::nodelete>>(
      m, "Operation")
      .def("operands",
           [](mlir::Operation &self) -> std::vector<mlir::Value> {
             return std::vector<mlir::Value>(self.getOperands().begin(),
                                             self.getOperands().end());
           })
      .def("name",
           [](mlir::Operation &self) -> std::string {
             return self.getName().getStringRef().str();
           })
      .def("num_operands",
           [](mlir::Operation &self) { return self.getNumOperands(); })
      .def(
          "get_operand",
          [](mlir::Operation &self, int id) -> mlir::OpOperand * {
            return &self.getOpOperand(id);
          },
          ret::reference)
      .def("get_num_results",
           [](mlir::Operation &self) -> int { return self.getNumResults(); })
      .def(
          "get_result",
          [](mlir::Operation &self, int idx) -> mlir::Value {
            return self.getResult(idx);
          },
          ret::reference)
      .def("get_attr",
           [](mlir::Operation &self, const std::string &name)
               -> mlir::Attribute { return self.getAttr(name); })
      .def("to_value",
           [](mlir::Operation &self) { return mlir::Value(self.getResult(0)); })
      .def("to_if_op",
           [](mlir::Operation &self) {
             return llvm::cast<mlir::scf::IfOp>(self);
           })
      .def("to_call_op",
           [](mlir::Operation &self) {
             return llvm::cast<mlir::func::CallOp>(self);
           })
      .def("to_llvm_call_op",
           [](mlir::Operation &self) {
             return llvm::cast<mlir::LLVM::CallOp>(self);
           })

      .def("__str__", [](mlir::Operation &self) -> std::string {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        os << self;
        os.flush();
        return buf;
      });

  // scf Ops
  py::class_<mlir::scf::ForOp, mlir::OpState>(m, "ForOp")
      .def("get_induction_var", &mlir::scf::ForOp::getInductionVar);

  py::class_<mlir::func::CallOp, mlir::OpState>(m, "CallOp")
      .def("get_callee",
           [](mlir::func::CallOp &self) -> std::string {
             return self.getCallee().str();
           })
      .def("operands",
           [](mlir::func::CallOp &self) -> std::vector<mlir::Value> {
             return std::vector<mlir::Value>(self.getOperands().begin(),
                                             self.getOperands().end());
           });

  py::class_<mlir::LLVM::CallOp, mlir::OpState>(m, "LLVMCallOp")
      .def("get_callee",
           [](mlir::LLVM::CallOp &self) -> std::string {
             llvm::StringRef callee = self.getCallee().value();
             return callee.str();
           })
      .def("operands",
           [](mlir::LLVM::CallOp &self) -> std::vector<mlir::Value> {
             return std::vector<mlir::Value>(self.getOperands().begin(),
                                             self.getOperands().end());
           });

  py::class_<mlir::scf::IfOp, mlir::OpState>(m, "IfOp")
      .def("get_then_block", &mlir::scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &mlir::scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &mlir::scf::IfOp::thenYield)
      .def("get_else_yield", &mlir::scf::IfOp::elseYield);
  py::class_<mlir::scf::YieldOp, mlir::OpState>(m, "YieldOp");
  py::class_<mlir::scf::WhileOp, mlir::OpState>(m, "WhileOp")
      .def("get_before", &mlir::scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &mlir::scf::WhileOp::getAfter, ret::reference);
  py::class_<mlir::scf::ConditionOp, mlir::OpState>(m, "CondtionOp");

  py::class_<mlir::pyimacs::GuardOp, mlir::OpState>(m, "GuardOp")
      .def(
          "get_body",
          [](mlir::pyimacs::GuardOp &self) -> mlir::Block & {
            return self.getBody().front();
          },
          ret::reference);

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<mlir::ModuleOp, mlir::OpState>(m, "Module", py::dynamic_attr())
      .def("dump", &mlir::ModuleOp::dump)
      .def("__str__",
           [](mlir::ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("push_back",
           [](mlir::ModuleOp &self, mlir::func::FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("push_back",
           [](mlir::ModuleOp &self, mlir::LLVM::LLVMFuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("has_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function__",
           [](mlir::ModuleOp &self,
              std::string &funcName) -> mlir::func::FuncOp {
             return self.lookupSymbol<mlir::func::FuncOp>(funcName);
           })
      .def("get_llvm_function",
           [](mlir::ModuleOp &self,
              std::string &funcName) -> mlir::LLVM::LLVMFuncOp {
             return self.lookupSymbol<mlir::LLVM::LLVMFuncOp>(funcName);
           })

      .def("get_function_names",
           [](mlir::ModuleOp &self) -> std::vector<std::string> {
             std::vector<std::string> ret;
             self.walk([&](mlir::Operation *op) {
               if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
                 ret.push_back(func->getName().stripDialect().str());
               }
               if (auto func = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
                 ret.push_back(func.getName().str());
               }
             });
             return ret;
           })
      .def("get_body", [](mlir::ModuleOp &self, int blockId) {
        return self.getBody(blockId);
      });

  /*
  m.def(
      "parse_mlir_module",
      [](const std::string &inputFilename, mlir::MLIRContext &context) {
        // initialize registry
        mlir::DialectRegistry registry;
        registry.insert<mlir::pyimacs::LispDialect, mlir::math::MathDialect,
                        mlir::arith::ArithmeticDialect,
  mlir::scf::SCFDialect>(); context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        // parse module
        mlir::OwningOpRef<mlir::ModuleOp> module(
            mlir::parseSourceFile(inputFilename, &context));
        // locations are incompatible with ptx < 7.5 !
        module->walk([](mlir::Operation *op) {
          op->setLoc(mlir::UnknownLoc::get(op->getContext()));
        });
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");

        return module->clone();
      },
      ret::take_ownership);
      */

  py::class_<mlir::func::FuncOp, mlir::OpState>(m, "Function")
      .def("args",
           [](mlir::func::FuncOp &self, unsigned idx) -> mlir::Value {
             return self.getArgument(idx);
           })
      .def("get_name",
           [](mlir::func::FuncOp &self) -> std::string {
             return self->getName().stripDialect().str();
           })
      .def("num_args",
           [](mlir::func::FuncOp &self) -> int {
             return self.getNumArguments();
           })
      .def(
          "add_entry_block",
          [](mlir::func::FuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          },
          ret::reference)
      .def("remove", [](mlir::func::FuncOp &self) { self->remove(); })
      .def(
          "body",
          [](mlir::func::FuncOp &self) -> mlir::Region * {
            return &self.getBody();
          },
          ret::reference)
      .def(
          "set_arg_attr",
          [](mlir::func::FuncOp &self, int arg_no, const std::string &name,
             int val) {
            // set arg attributes "name" to value "val"
            auto attrTy = mlir::IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, mlir::IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      .def("reset_type", &mlir::func::FuncOp::setType);

  py::class_<mlir::LLVM::LLVMFuncOp, mlir::OpState>(m, "LLVMFunction")
      .def("args",
           [](mlir::LLVM::LLVMFuncOp &self, unsigned idx) -> mlir::Value {
             return self.getArgument(idx);
           })
      .def("get_name",
           [](mlir::LLVM::LLVMFuncOp &self) -> std::string {
             return self.getName().str();
           })
      .def("num_args",
           [](mlir::LLVM::LLVMFuncOp &self) -> int {
             return self.getNumArguments();
           })
      .def(
          "add_entry_block",
          [](mlir::LLVM::LLVMFuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          },
          ret::reference)
      .def("remove", [](mlir::LLVM::LLVMFuncOp &self) { self->remove(); })
      .def(
          "body",
          [](mlir::LLVM::LLVMFuncOp &self) -> mlir::Region * {
            return &self.getBody();
          },
          ret::reference)
      .def(
          "set_arg_attr",
          [](mlir::LLVM::LLVMFuncOp &self, int arg_no, const std::string &name,
             int val) {
            // set arg attributes "name" to value "val"
            auto attrTy = mlir::IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, mlir::IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      .def("reset_type", &mlir::LLVM::LLVMFuncOp::setType);

  py::class_<mlir::OpBuilder::InsertPoint>(m, "InsertPoint");
}

void initBuilder(py::module &m) {

  py::class_<mlir::OpBuilder>(m, "Builder", py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      // // getters
      .def_property_readonly("context", &mlir::OpBuilder::getContext,
                             ret::reference)
      .def("create_module",
           [](mlir::OpBuilder &self) -> mlir::ModuleOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::ModuleOp>(loc);
           })
      .def("ret",
           [](mlir::OpBuilder &self, std::vector<mlir::Value> &vals) -> void {
             auto loc = self.getUnknownLoc();
             self.create<mlir::func::ReturnOp>(loc, vals);
           })
      .def(
          "call",
          [](mlir::OpBuilder &self, mlir::func::FuncOp &func,
             std::vector<mlir::Value> &args) -> mlir::Operation * {
            auto loc = self.getUnknownLoc();
            return self.create<mlir::func::CallOp>(loc, func, args);
          },
          ret::reference)
      .def(
          "llvm_call",
          [](mlir::OpBuilder &self, mlir::LLVM::LLVMFuncOp &func,
             std::vector<mlir::Value> &args) -> mlir::Operation * {
            auto loc = self.getUnknownLoc();
            return self.create<mlir::LLVM::CallOp>(loc, func, args);
          },
          ret::reference)
      .def("extern_call",
           [](mlir::OpBuilder &self, mlir::Type retType, std::string callee,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             auto loc = self.getUnknownLoc();
             auto calleeAttr =
                 mlir::StringAttr::get(loc.getContext(), callee.data());
             return self.create<mlir::pyimacs::CallOp>(loc, retType, calleeAttr,
                                                       args);
           })
      .def("make_tuple",
           [](mlir::OpBuilder &self,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             auto loc = self.getUnknownLoc();
             auto ObjectTy = mlir::pyimacs::ObjectType::get(self.getContext());
             return self.create<mlir::pyimacs::MakeTupleOp>(loc, ObjectTy,
                                                            args);
           })
      .def("make_symbol",
           [](mlir::OpBuilder &self, mlir::Value name,
              mlir::Value is_keyword) -> mlir::OpState {
             auto loc = self.getUnknownLoc();
             auto value = llvm::dyn_cast<mlir::arith::ConstantOp>(
                 is_keyword.getDefiningOp());
             assert(value);
             auto boolAttr = value.getValue().cast<mlir::BoolAttr>();
             return self.create<mlir::pyimacs::MakeSymbolOp>(loc, name,
                                                             boolAttr);
           })
      .def("guard",
           [](mlir::OpBuilder &self, const std::string &name) {
             auto loc = self.getUnknownLoc();
             auto value = mlir::StringAttr::get(self.getContext(), name.data());
             auto op = self.create<mlir::pyimacs::GuardOp>(loc, self, value);
             return op;
           })

      .def("make_symbol",
           [](mlir::OpBuilder &self, mlir::Value name,
              bool is_keyword) -> mlir::OpState {
             auto loc = self.getUnknownLoc();
             auto boolAttr = mlir::BoolAttr::get(self.getContext(), is_keyword);
             return self.create<mlir::pyimacs::MakeSymbolOp>(loc, name,
                                                             boolAttr);
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](mlir::OpBuilder &self, mlir::Block &block) -> void {
             self.setInsertionPointToStart(&block);
           })
      .def("set_insertion_point_to_end",
           [](mlir::OpBuilder &self, mlir::Block &block) {
             self.setInsertionPointToEnd(&block);
           })
      .def(
          "get_insertion_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            return self.getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point", &mlir::OpBuilder::saveInsertionPoint)
      .def("restore_insertion_point", &mlir::OpBuilder::restoreInsertionPoint)
      // .def("set_insert_point", [](ir::builder *self,
      // std::pair<ir::basic_block*, ir::instruction*> pt) {
      //   ir::basic_block *bb = pt.first;
      //   ir::instruction *instr = pt.second;
      //   if (instr) {
      //     if (bb != instr->get_parent())
      //       throw std::runtime_error("invalid insertion point, instr not in
      //       bb");
      //     self->set_insert_point(instr);
      //   } else {
      //     assert(bb);
      //     self->set_insert_point(bb);
      //   }
      // })
      // Attr
      .def("get_bool_attr", &mlir::OpBuilder::getBoolAttr)
      .def("get_int32_attr", &mlir::OpBuilder::getI32IntegerAttr)
      // Use arith.ConstantOp to create constants
      // // Constants
      .def("get_int1",
           [](mlir::OpBuilder &self, bool v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI1Type()));
           })
      .def("get_int32",
           [](mlir::OpBuilder &self, int64_t v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 loc, v, self.getI32Type()));
           })
      .def("get_string",
           [](mlir::OpBuilder &self, std::string v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return mlir::Value(self.create<mlir::arith::ConstantOp>(
                 loc, mlir::StringAttr::get(self.getContext(), v.data()),
                 mlir::pyimacs::StringType::get(self.getContext())));
           })

      // .def("get_uint32", &ir::builder::get_int32, ret::reference)
      // .def("get_int64", [](ir::builder *self, int64_t v) { return
      // self->get_int64((uint64_t)v); }, ret::reference) .def("get_uint64",
      // &ir::builder::get_int64, ret::reference) .def("get_float16",
      // &ir::builder::get_float16, ret::reference)
      .def("get_float32",
           [](mlir::OpBuilder &self, float v) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::arith::ConstantOp>(
                 loc, self.getF32FloatAttr(v));
           })
      .def("get_null_value",
           [](mlir::OpBuilder &self, mlir::Type type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             if (auto floatTy = type.dyn_cast<mlir::FloatType>())
               return self.create<mlir::arith::ConstantFloatOp>(
                   loc, mlir::APFloat(floatTy.getFloatSemantics(), 0), floatTy);
             else if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(loc, 0, intTy);
             else
               throw std::runtime_error("Not implemented");
           })
      .def("get_null_as_int",
           [](mlir::OpBuilder &self) -> mlir::Value {
             return self.create<mlir::pyimacs::GetNullOp>(
                 self.getUnknownLoc(), self.getIntegerType(32));
           })
      .def("get_null_as_float",
           [](mlir::OpBuilder &self) -> mlir::Value {
             return self.create<mlir::pyimacs::GetNullOp>(self.getUnknownLoc(),
                                                          self.getF32Type());
           })
      .def("get_null_as_string",
           [](mlir::OpBuilder &self) -> mlir::Value {
             return self.create<mlir::pyimacs::GetNullOp>(
                 self.getUnknownLoc(),
                 mlir::pyimacs::StringType::get(self.getContext()));
           })
      .def("get_null_as_object",
           [](mlir::OpBuilder &self) -> mlir::Value {
             return self.create<mlir::pyimacs::GetNullOp>(
                 self.getUnknownLoc(),
                 mlir::pyimacs::ObjectType::get(self.getContext()));
           })
      .def("get_all_ones_value",
           [](mlir::OpBuilder &self, mlir::Type type) -> mlir::Value {
             auto loc = self.getUnknownLoc();
             uint64_t val = 0xFFFFFFFFFFFFFFFF;
             if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(loc, val, intTy);
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return mlir::LLVM::LLVMVoidType::get(self.getContext());
           })
      .def("get_int1_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getI1Type();
           }) // or ret::copy?
      .def("get_int8_ty",
           [](mlir::OpBuilder &self) -> mlir::Type { return self.getI8Type(); })
      .def("get_int16_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return self.getType<mlir::IntegerType>(16);
           })
      .def(
          "get_int32_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI32Type(); })
      .def(
          "get_int64_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getI64Type(); })
      .def(
          "get_float_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF32Type(); })
      .def(
          "get_double_ty",
          [](mlir::OpBuilder &self) -> mlir::Type { return self.getF64Type(); })
      .def("get_string_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return mlir::pyimacs::StringType::get(self.getContext());
           })
      .def("get_object_ty",
           [](mlir::OpBuilder &self) -> mlir::Type {
             return mlir::pyimacs::ObjectType::get(self.getContext());
           })
      .def("get_block_ty",
           [](mlir::OpBuilder &self, mlir::Type &elementType,
              std::vector<int64_t> &shape) -> mlir::Type {
             return mlir::RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty__",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> inTypes,
              std::vector<mlir::Type> outTypes) -> mlir::Type {
             return self.getFunctionType(inTypes, outTypes);
           })
      .def(
          "get_llvm_function_ty",
          [](mlir::OpBuilder &self, std::vector<mlir::Type> inTypes,
             std::vector<mlir::Type> outTypes,
             bool isVarArg = true) -> mlir::Type {
            assert(outTypes.size() <= 1);
            mlir::Type outTy =
                outTypes.empty()
                    ? mlir::LLVM::LLVMVoidType::get(self.getContext())
                    : outTypes[0];
            return mlir::LLVM::LLVMFunctionType::get(outTy, inTypes, isVarArg);
          },
          "get a FunctionType in LLVM dialect.", py::arg("in_types"),
          py::arg("out_types"), py::arg("is_var_arg") = false)

      // Ops
      .def("get_or_insert_function__",
           [](mlir::OpBuilder &self, mlir::ModuleOp &module,
              const std::string &funcName, mlir::Type &funcType,
              const std::string &visibility) -> mlir::func::FuncOp {
             if (mlir::Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<mlir::func::FuncOp>(funcOperation);

             auto loc = self.getUnknownLoc();
             mlir::OpBuilder::InsertionGuard guard(self);
             self.setInsertionPoint(module);
             if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
               mlir::ArrayRef<mlir::NamedAttribute> attrs = {
                   mlir::NamedAttribute(self.getStringAttr("sym_visibility"),
                                        self.getStringAttr(visibility))};
               return self.create<mlir::func::FuncOp>(loc, funcName, funcTy,
                                                      attrs);
             }
             throw std::runtime_error("invalid function type");
           })
      .def("get_or_insert_llvm_function",
           [](mlir::OpBuilder &self, mlir::ModuleOp &module,
              const std::string &funcName, mlir::Type &funcType,
              const std::string &visibility) -> mlir::LLVM::LLVMFuncOp {
             if (mlir::Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(funcOperation);

             auto loc = self.getUnknownLoc();
             mlir::OpBuilder::InsertionGuard guard(self);
             self.setInsertionPoint(module);
             if (auto funcTy =
                     funcType.dyn_cast<mlir::LLVM::LLVMFunctionType>()) {
               mlir::ArrayRef<mlir::NamedAttribute> attrs = {
                   mlir::NamedAttribute(self.getStringAttr("sym_visibility"),
                                        self.getStringAttr(visibility))};
               return self.create<mlir::LLVM::LLVMFuncOp>(loc, funcName,
                                                          funcTy);
             }
             throw std::runtime_error("invalid function type");
           })
      .def(
          "create_block",
          [](mlir::OpBuilder &self) -> mlir::Block * {
            mlir::Region *parent = self.getBlock()->getParent();
            return self.createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](mlir::OpBuilder &self, mlir::Region &parent,
             std::vector<mlir::Type> &argTypes) -> mlir::Block * {
            auto argLoc = self.getUnknownLoc();
            llvm::SmallVector<mlir::Location, 8> argLocs(argTypes.size(),
                                                         argLoc);
            return self.createBlock(&parent, {}, argTypes, argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](mlir::OpBuilder &self)
              -> mlir::Block * { return new mlir::Block(); },
          ret::reference)
      // Structured control flow
      .def("create_for_op",
           [](mlir::OpBuilder &self, mlir::Value &lb, mlir::Value &ub,
              mlir::Value &step,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::ForOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ForOp>(loc, lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              mlir::Value &condition, bool withElse) -> mlir::scf::IfOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::IfOp>(loc, retTypes, condition,
                                                 withElse);
           })
      .def("create_yield_op",
           [](mlir::OpBuilder &self,
              std::vector<mlir::Value> &yields) -> mlir::scf::YieldOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::YieldOp>(loc, yields);
           })
      .def("create_while_op",
           [](mlir::OpBuilder &self, std::vector<mlir::Type> &retTypes,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::WhileOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::WhileOp>(loc, retTypes, initArgs);
           })
      .def("create_condition_op",
           [](mlir::OpBuilder &self, mlir::Value &cond,
              std::vector<mlir::Value> &args) -> mlir::scf::ConditionOp {
             auto loc = self.getUnknownLoc();
             return self.create<mlir::scf::ConditionOp>(loc, cond, args);
           })
#define ADD_BINARY_OP(api__, ir_op__)                                          \
  .def("create_" #api__,                                                       \
       [](mlir::OpBuilder &self, mlir::Value &a,                               \
          mlir::Value &b) -> mlir::Value {                                     \
         auto loc = self.getUnknownLoc();                                      \
         return self.create<mlir::arith::ir_op__##Op>(loc, a, b);              \
       })

      // float type
      ADD_BINARY_OP(fmul, MulF) ADD_BINARY_OP(fdiv, DivF)
          ADD_BINARY_OP(frem, RemF) ADD_BINARY_OP(fadd, AddF)
              ADD_BINARY_OP(fsub, SubF)

      // int type
      ADD_BINARY_OP(mul, MulI) ADD_BINARY_OP(div, DivUI)
          ADD_BINARY_OP(add, AddI) ADD_BINARY_OP(sub, SubI)

#define ADD_CMP_OP(api__, op__, enum__)                                        \
  .def("create_" #api__,                                                       \
       [](mlir::OpBuilder &self, mlir::Value &lhs,                             \
          mlir::Value &rhs) -> mlir::Value {                                   \
         auto loc = self.getUnknownLoc();                                      \
         return self.create<mlir::arith::op__##Op>(                            \
             loc, mlir::arith::CmpIPredicate::enum__, lhs, rhs);               \
       })

      // int type
      ADD_CMP_OP(icmpSLE, CmpI, sle) ADD_CMP_OP(icmpSLT, CmpI, slt)
          ADD_CMP_OP(icmpSGE, CmpI, sge) ADD_CMP_OP(icmpSGT, CmpI, sgt)
              ADD_CMP_OP(icmpULE, CmpI, ule) ADD_CMP_OP(icmpULT, CmpI, ult)
                  ADD_CMP_OP(icmpUGE, CmpI, uge) ADD_CMP_OP(icmpUGT, CmpI, ugt)
                      ADD_CMP_OP(icmpEQ, CmpI, eq) ADD_CMP_OP(icmpNE, CmpI, ne)

#undef ADD_CMP_OP
#define ADD_CMP_OP(api__, op__, enum__)                                        \
  .def("create_" #api__,                                                       \
       [](mlir::OpBuilder &self, mlir::Value &lhs,                             \
          mlir::Value &rhs) -> mlir::Value {                                   \
         auto loc = self.getUnknownLoc();                                      \
         return self.create<mlir::arith::op__##Op>(                            \
             loc, mlir::arith::CmpFPredicate::enum__, lhs, rhs);               \
       })

      // float type
      ADD_CMP_OP(fcmpOLT, CmpF, OLT) ADD_CMP_OP(fcmpOGT, CmpF, OGT)
          ADD_CMP_OP(fcmpOLE, CmpF, OLE) ADD_CMP_OP(fcmpOGE, CmpF, OGE)
              ADD_CMP_OP(fcmpOEQ, CmpF, OEQ) ADD_CMP_OP(fcmpONE, CmpF, ONE);
}

void initTarget(py::module &m) {
  // m.def("to_lisp_code", [](mlir::ModuleOp &op) -> std::string {
  //   std::string buf;
  //   llvm::raw_string_ostream os(buf);
  //   mlir::pyimacs::translateToElisp(&op, os);
  //   os.flush();
  //   return buf;
  // });
}

} // namespace pyimacs

void initPyimas(py::module &m) {
  py::module pyimacs = m.def_submodule("pyimacs");
  auto ir = pyimacs.def_submodule("ir");
  pyimacs::initMLIR(ir);
  pyimacs::initBuilder(ir);

  auto target = pyimacs.def_submodule("target");
  pyimacs::initTarget(target);
}

PYBIND11_MODULE(libpyimacs, m) {
  m.doc() = "Python bindings to the C++ PyIMacs API.";
  initPyimas(m);
}
