get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_llvm_executable(pyimacs-opt opt.cc PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(pyimacs-opt)

message(WARNING "dialect_libs: ${dialect_libs}")
target_link_libraries(pyimacs-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  LispIR
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)
