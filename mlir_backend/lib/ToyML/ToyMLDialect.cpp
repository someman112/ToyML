//===- ToyMLDialect.cpp - ToyML dialect implementation --------*- C++ -*-===//
//
// This file implements the ToyML dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "ToyML/ToyMLDialect.h"
#include "ToyML/ToyMLOps.h"
#include "ToyML/ToyMLTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace toyml;

//===----------------------------------------------------------------------===//
// ToyML dialect
//===----------------------------------------------------------------------===//

#include "../../build/lib/ToyML/ToyMLDialect.cpp.inc"

void ToyMLDialect::initialize() {
  // We're temporarily not registering any operations
  // We'll implement the operations manually later
  
  // Manually register types
  addTypes<DatasetType, ModelType, MetricType>();
}

// Add implementations for parseAttribute and printAttribute
// These are required by the Dialect class but we don't have any custom attributes yet
Attribute ToyMLDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // Return null attribute as we don't have any custom attributes yet
  return Attribute();
}

void ToyMLDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // Do nothing as we don't have any custom attributes yet
}

//===----------------------------------------------------------------------===//
// ToyML dialect type parsing and printing
//===----------------------------------------------------------------------===//

Type ToyMLDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "dataset")
    return DatasetType::get(getContext());
  if (keyword == "model")
    return ModelType::get(getContext());
  if (keyword == "metric")
    return MetricType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown ToyML type: ") << keyword;
  return Type();
}

void ToyMLDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
      .Case<DatasetType>([&](auto) { printer << "dataset"; })
      .Case<ModelType>([&](auto) { printer << "model"; })
      .Case<MetricType>([&](auto) { printer << "metric"; })
      .Default([](Type) { llvm_unreachable("unexpected ToyML type"); });
}