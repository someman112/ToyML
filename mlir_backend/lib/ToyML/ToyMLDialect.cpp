//===- ToyMLDialect.cpp - ToyML dialect implementation -----------*- C++ -*-===//
//
// This file implements the ToyML dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "ToyML/ToyMLDialect.h"

using namespace mlir;
using namespace mlir::toyml;

//===----------------------------------------------------------------------===//
// ToyML dialect.
//===----------------------------------------------------------------------===//

// Include the generated dialect implementation
#include "../../build/lib/ToyML/ToyMLDialect.cpp.inc"

void ToyMLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "../../build/lib/ToyML/ToyMLOps.cpp.inc"
  >();
}
