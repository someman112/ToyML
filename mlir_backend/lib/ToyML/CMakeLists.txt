# ToyML dialect CMakeLists.txt

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(LLVM_TARGET_DEFINITIONS ${INCLUDE_DIR}/ToyML/ToyMLOps.td)

# Configure tablegen outputs to the include directory where the headers will find them
set(MLIR_TABLEGEN_OUTPUT_DIR ${INCLUDE_DIR}/ToyML)

# Generate TableGen outputs
mlir_tablegen(ToyMLOps.h.inc -gen-op-decls -I${INCLUDE_DIR})
mlir_tablegen(ToyMLOps.cpp.inc -gen-op-defs -I${INCLUDE_DIR})
mlir_tablegen(ToyMLDialect.h.inc -gen-dialect-decls -dialect=toyml -I${INCLUDE_DIR})
mlir_tablegen(ToyMLDialect.cpp.inc -gen-dialect-defs -dialect=toyml -I${INCLUDE_DIR})
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
