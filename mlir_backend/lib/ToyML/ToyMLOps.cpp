//===- ToyMLOps.cpp - ToyML dialect operations implementation -*- C++ -*-===//
//
// This file implements the operations for the ToyML dialect.
//
//===----------------------------------------------------------------------===//

#include "ToyML/ToyMLOps.h"
#include "ToyML/ToyMLDialect.h"
#include "ToyML/ToyMLTypes.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace toyml;

// We're disabling TableGen for operations temporarily
// to get a simpler build working

// #define GET_OP_CLASSES
// #include "../../build/lib/ToyML/ToyMLOps.cpp.inc"