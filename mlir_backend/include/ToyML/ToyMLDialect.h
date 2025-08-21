//===- ToyMLDialect.h - ToyML dialect declaration -------------*- C++ -*-===//
//
// This file declares the ToyML dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef TOYML_DIALECT_H
#define TOYML_DIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace toyml {

// Forward declare the generated dialect declaration
class ToyMLDialect;

} // namespace toyml
} // namespace mlir

// Include the generated dialect declarations
#include "../../build/lib/ToyML/ToyMLDialect.h.inc"

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "../../build/lib/ToyML/ToyMLOps.h.inc"

#endif // TOYML_DIALECT_H
