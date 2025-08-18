# ToyML dialect CMakeLists.txt

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ToyML/ToyMLOps.td)
mlir_tablegen(ToyMLOps.h.inc -gen-op-decls)
mlir_tablegen(ToyMLOps.cpp.inc -gen-op-defs)
mlir_tablegen(ToyMLDialect.h.inc -gen-dialect-decls -dialect=toyml)
mlir_tablegen(ToyMLDialect.cpp.inc -gen-dialect-defs -dialect=toyml)
add_public_tablegen_target(ToyMLOpsIncGen)
add_mlir_doc(ToyMLDialect ToyMLDialect ToyML/ -gen-dialect-doc)
add_mlir_doc(ToyMLOps ToyMLOps ToyML/ -gen-op-doc)

add_mlir_dialect_library(MLIRToyML
  ToyMLDialect.cpp

  DEPENDS
  ToyMLOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
)
