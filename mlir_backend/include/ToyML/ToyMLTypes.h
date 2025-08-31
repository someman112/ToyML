//===- ToyMLTypes.h - ToyML dialect types ----------------------*- C++ -*-===//
//
// This file defines the types for the ToyML dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOYML_TYPES_H
#define TOYML_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"
#include "ToyML/ToyMLDialect.h"

namespace mlir {
namespace toyml {

namespace detail {
// Define storage classes for the ToyML dialect types if needed
} // namespace detail

// Dataset Type
class DatasetType : public Type::TypeBase<DatasetType, Type, TypeStorage> {
public:
  using Base::Base;
  
  static DatasetType get(MLIRContext *context) {
    return Base::get(context);
  }
};

// Model Type
class ModelType : public Type::TypeBase<ModelType, Type, TypeStorage> {
public:
  using Base::Base;
  
  static ModelType get(MLIRContext *context) {
    return Base::get(context);
  }
};

// Metric Type (previously SummaryType)
class MetricType : public Type::TypeBase<MetricType, Type, TypeStorage> {
public:
  using Base::Base;
  
  static MetricType get(MLIRContext *context) {
    return Base::get(context);
  }
};

// We're not using TableGen-generated classes for types 
// #define GET_TYPEDEF_CLASSES
// #include "../../build/lib/ToyML/ToyMLTypes.h.inc"

} // namespace toyml
} // namespace mlir

#endif // TOYML_TYPES_H
