//===- ToyMLOps.h - ToyML dialect operations -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ToyML dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOYML_TOYMLOPS_H
#define MLIR_TOYML_TOYMLOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "ToyML/ToyMLDialect.h"
#include "ToyML/ToyMLTypes.h"

namespace mlir {
namespace toyml {

// Forward declarations of operations
class DatasetOp;
class SplitOp;
class TrainOp;
class EvaluateOp;
class PrintOp;
class PipelineOp;
class PipelineEndOp;

// Disable TableGen-generated operation classes
// #define GET_OP_CLASSES
// #include "../../build/lib/ToyML/ToyMLOps.h.inc"

} // namespace toyml
} // namespace mlir

#endif // MLIR_TOYML_TOYMLOPS_H
