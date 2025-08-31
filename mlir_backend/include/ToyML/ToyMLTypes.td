//===- ToyMLTypes.td - ToyML dialect types -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// Defines the types for the ToyML dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOYML_TYPES
#define TOYML_TYPES

include "mlir/IR/AttrTypeBase.td"
include "ToyML/ToyMLDialect.td"

//===----------------------------------------------------------------------===//
// ToyML Type Definitions
//===----------------------------------------------------------------------===//

class ToyML_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ToyML_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Dataset Type
def ToyML_DatasetType : ToyML_Type<"Dataset", "dataset"> {
  let summary = "A dataset type";
  let description = [{
    A type representing a machine learning dataset.
  }];
  
  let assemblyFormat = "";
}

// Model Type
def ToyML_ModelType : ToyML_Type<"Model", "model"> {
  let summary = "A model type";
  let description = [{
    A type representing a machine learning model.
  }];
  
  let assemblyFormat = "";
}

// Metric Type
def ToyML_MetricType : ToyML_Type<"Metric", "metric"> {
  let summary = "A metric type";
  let description = [{
    A type representing metrics from model evaluation.
  }];
  
  let assemblyFormat = "";
}

#endif // TOYML_TYPES
