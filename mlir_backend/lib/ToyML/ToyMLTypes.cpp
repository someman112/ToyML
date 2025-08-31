//===- ToyMLTypes.cpp - ToyML dialect types implementation ----*- C++ -*-===//
//
// This file implements the types for the ToyML dialect.
//
//===----------------------------------------------------------------------===//

#include "ToyML/ToyMLTypes.h"
#include "ToyML/ToyMLDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace toyml;

// We're not using TableGen-generated classes for types
// #define GET_TYPEDEF_CLASSES
// #include "../../build/lib/ToyML/ToyMLTypes.cpp.inc"

// DatasetType, ModelType, and MetricType implementations are inline in the header