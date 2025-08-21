//===- ToyMLOps.td - ToyML dialect operations -----------*- tablegen -*-===//
//
// This file defines the ToyML operations in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef TOYML_OPS
#define TOYML_OPS

include "ToyML/ToyMLDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dataset Operations
//===----------------------------------------------------------------------===//

def DatasetOp : ToyML_Op<"dataset", [Pure]> {
  let summary = "Define and load a dataset from a source path";
  let description = [{
    This operation defines and loads a dataset, with an optional name/alias.
    
    Examples:
    ```mlir
    // With a name/alias
    %0 = toyml.dataset "data.csv" as "mnist"
    
    // Without a name/alias
    %1 = toyml.dataset "data.csv"
    ```
  }];
  
  let arguments = (ins
    StrAttr:$path,
    OptionalAttr<StrAttr>:$name
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$path (`as` $name^)? attr-dict `:` type($result)";
}

def SplitOp : ToyML_Op<"split", []> {
  let summary = "Split a dataset into multiple parts";
  let description = [{
    This operation splits a source dataset into multiple target datasets,
    potentially with parameters like split ratio.
    
    Example:
    ```mlir
    %train, %test = toyml.split %dataset into ["train", "test"] {ratio = 0.8}
    ```
  }];
  
  let arguments = (ins
    AnyType:$source,
    ArrayAttr:$targets,
    OptionalAttr<DictionaryAttr>:$params
  );
  
  let results = (outs Variadic<AnyType>:$results);
  
  let assemblyFormat = "$source `into` $targets (`with` $params^)? attr-dict `:` type($source) `->` type($results)";
}

def TrainOp : ToyML_Op<"train", []> {
  let summary = "Train a model on a dataset";
  let description = [{
    This operation trains a model on a source dataset with an optional model type
    and parameters.
    
    Examples:
    ```mlir
    // With default model type
    %model = toyml.train %dataset {epochs = 10, lr = 0.01}
    
    // With specific model type
    %model = toyml.train %dataset using "linear_regression" {epochs = 10}
    ```
  }];
  
  let arguments = (ins
    AnyType:$source,
    OptionalAttr<StrAttr>:$model_type,
    OptionalAttr<DictionaryAttr>:$params
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$source (`using` $model_type^)? (`with` $params^)? attr-dict `:` type($source) `->` type($result)";
}

def EvaluateOp : ToyML_Op<"evaluate", []> {
  let summary = "Evaluate a model on a dataset";
  let description = [{
    This operation evaluates a model on a test dataset.
    
    Example:
    ```mlir
    %metrics = toyml.evaluate %model on %test_data
    ```
  }];
  
  let arguments = (ins
    AnyType:$model,
    AnyType:$dataset
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$model `on` $dataset attr-dict `:` type($model) `,` type($dataset) `->` type($result)";
}

def PrintOp : ToyML_Op<"print", [Pure]> {
  let summary = "Print an expression";
  let description = [{
    This operation prints a value or expression result.
    
    Example:
    ```mlir
    toyml.print %value : f32
    ```
  }];
  
  let arguments = (ins AnyType:$input);
  
  // No results
  let results = (outs);
  
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PipelineOp : ToyML_Op<"pipeline", [
    SingleBlockImplicitTerminator<"PipelineEndOp">
  ]> {
  let summary = "Define a pipeline of operations";
  let description = [{
    This operation defines a named pipeline containing a sequence of operations.
    
    Example:
    ```mlir
    toyml.pipeline "Prep" {
      // operations
    }
    ```
  }];
  
  let arguments = (ins StrAttr:$name);
  
  // Region containing the pipeline operations
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = "$name $body attr-dict";
}

def PipelineEndOp : ToyML_Op<"pipeline_end", [
    Pure, HasParent<"PipelineOp">, Terminator
  ]> {
  let summary = "Terminator for pipeline regions";
  
  let arguments = (ins);
  let results = (outs);
  
  let assemblyFormat = "attr-dict";
}

#endif // TOYML_OPS
