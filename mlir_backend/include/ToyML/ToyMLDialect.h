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

// Include the generated dialect declarations
#include "ToyML/ToyMLDialect.h.inc"

namespace mlir {
namespace toyml {

class ToyMLDialect : public Dialect {
public:
  explicit ToyMLDialect(MLIRContext *context);
  
  static StringRef getDialectNamespace() { return "toyml"; }
  
  // Override the dialect hooks
  void initialize();
};

} // namespace toyml
} // namespace mlir

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "ToyML/ToyMLOps.h.inc"

#endif // TOYML_DIALECT_H
