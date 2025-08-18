//===- ToyMLDialect.cpp - ToyML dialect implementation -----------*- C++ -*-===//
//
// This file implements the ToyML dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "ToyML/ToyMLDialect.h"
#include "ToyML/ToyMLOps.h"

using namespace mlir;
using namespace mlir::toyml;

//===----------------------------------------------------------------------===//
// ToyML dialect.
//===----------------------------------------------------------------------===//

// Include the generated dialect implementation
#include "ToyML/ToyMLDialect.cpp.inc"

void ToyMLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyML/ToyMLOps.cpp.inc"
  >();
}

ToyMLDialect::ToyMLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<ToyMLDialect>()) {
  initialize();
}
