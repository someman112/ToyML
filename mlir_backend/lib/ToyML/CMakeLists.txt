# ToyML dialect CMakeLists.txt

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# We're using manually written operations instead of TableGen-generated ones
# set(LLVM_TARGET_DEFINITIONS ${INCLUDE_DIR}/ToyML/ToyMLOps.td)

# Set the output directory for TableGen files directly
set(MLIR_TABLEGEN_OUTPUT_DIR "${BUILD_DIR}")

# We're using manually written operations instead of TableGen-generated ones
# mlir_tablegen(ToyMLOps.h.inc -gen-op-decls -I${INCLUDE_DIR})
# mlir_tablegen(ToyMLOps.cpp.inc -gen-op-defs -I${INCLUDE_DIR})

# Generate TableGen outputs for Dialect
set(LLVM_TARGET_DEFINITIONS ${INCLUDE_DIR}/ToyML/ToyMLDialect.td)
mlir_tablegen(ToyMLDialect.h.inc -gen-dialect-decls -dialect=toyml -I${INCLUDE_DIR})
mlir_tablegen(ToyMLDialect.cpp.inc -gen-dialect-defs -dialect=toyml -I${INCLUDE_DIR})

# We're using manually written types instead of TableGen-generated ones
# set(LLVM_TARGET_DEFINITIONS ${INCLUDE_DIR}/ToyML/ToyMLTypes.td)
# mlir_tablegen(ToyMLTypes.h.inc -gen-typedef-decls -I${INCLUDE_DIR})
# mlir_tablegen(ToyMLTypes.cpp.inc -gen-typedef-defs -I${INCLUDE_DIR})

# Add custom command to copy the generated files
add_custom_command(
  OUTPUT ${BUILD_DIR}/inc_copied
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_DIR}/inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BUILD_DIR}/ToyMLDialect.h.inc
    ${BUILD_DIR}/ToyMLDialect.cpp.inc
    ${BUILD_DIR}/inc/
  COMMAND ${CMAKE_COMMAND} -E touch ${BUILD_DIR}/inc_copied
  DEPENDS
    ${BUILD_DIR}/ToyMLDialect.h.inc
    ${BUILD_DIR}/ToyMLDialect.cpp.inc
  COMMENT "Copying TableGen output files to include directory"
)

# Create custom targets for TableGen files
# Not using TableGen for operations
# add_public_tablegen_target(ToyMLOpsIncGen)
add_public_tablegen_target(ToyMLDialectIncGen)
# Not using TableGen for types
# add_public_tablegen_target(ToyMLTypesIncGen)

# Create custom target for copying include files
add_custom_target(ToyMLIncCopy DEPENDS ${BUILD_DIR}/inc_copied)

# Add all targets to master dependencies
add_dependencies(toyml-tablegen-targets 
  # ToyMLOpsIncGen 
  ToyMLDialectIncGen 
  # ToyMLTypesIncGen 
  ToyMLIncCopy)

add_mlir_doc(ToyMLDialect ToyMLDialect ToyML/ -gen-dialect-doc)
# Not using TableGen for operations
# add_mlir_doc(ToyMLOps ToyMLOps ToyML/ -gen-op-doc)

# Build the dialect library with all source files
add_mlir_dialect_library(MLIRToyML
  ToyMLDialect.cpp
  ToyMLOps.cpp
  ToyMLTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ToyML

  DEPENDS
  # ToyMLOpsIncGen
  MLIRDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
)

# Add include directories
target_include_directories(MLIRToyML
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${MLIR_TABLEGEN_OUTPUT_DIR}
)
