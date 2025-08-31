//===- ToyMLDialect.td - ToyML dialect definition -----------*- tablegen -*-===//
//
// This file defines the ToyML dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef TOYML_DIALECT
#define TOYML_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ToyML dialect definition
//===----------------------------------------------------------------------===//

def ToyML_Dialect : Dialect {
  let name = "toyml";
  let summary = "A dialect for ToyML machine learning workflow operations";
  let description = [{
    The ToyML dialect provides operations for describing machine learning
    workflows, including dataset management, model training, and evaluation.
  }];
  
  let cppNamespace = "::mlir::toyml";
  
  let useDefaultAttributePrinterParser = 1;
  
  // Add custom type parsing and printing
  let extraClassDeclaration = [{
    // Type parsing and printing methods for ToyML dialect types
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
  }];
}

// Base class for all ToyML operations
class ToyML_Op<string mnemonic, list<Trait> traits = []> :
    Op<ToyML_Dialect, mnemonic, traits>;

#endif // TOYML_DIALECT
