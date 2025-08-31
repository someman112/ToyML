# ToyML Compiler Project

a compiler for a DSL that is designed for simpler machine learning workflows.

## Project Overview

ToyML is a dsl to simplify machine learning tasks such as loading datasets, training/testing models etc. The intention is for the compiler to translate source code into MLIR so it can then be further optimized and lowered to various backends.

## Project Structure

- `lexer_parser/`: 
  - `dsl.lark`:  Grammar definition (using lark)
  - `parser.py`: Parser implementation (using Lark)
  - `parser_tests.py`:

- `Transformer/`:
  - `transformer.py`: convert parse trees --> AST nodes
  - `transformer_tests.py`: 

- `mlir_backend/`: 
  - `mlir_gen.py`: Python code that emits MLIR
  - `mlir_gen_tests.py`:
  - `include/ToyML/`: MLIR dialect header files
  - `lib/ToyML/`: MLIR dialect implementation files

- `examples/`: Example ToyML programs
- `samples/`: Sample ToyML programs demonstrating language features

## Current Status

The project currently has:
- A working parser/lexer
- AST transformer to convert parse trees to AST nodes
- Basic MLIR generation from AST nodes
- Partial implementation of the MLIR dialect

## Next Steps

1. **Complete the MLIR dialect implementation: (will take most time)**
   - Implement the operation classes (DatasetOp, SplitOp, TrainOp, EvaluateOp, PrintOp, PipelineOp, PipelineEndOp) in `ToyMLOps.cpp`
   - Define operation classes with proper traits, attributes,verification logic,etc.
   - Register operations with the dialect in `ToyMLDialect.cpp`

2. **Test MLIR generation:**
   - write tests for the end-to-end workflow from ToyML source code to MLIR

3. **Implement MLIR lowering passes:**
   - conversion passes to lower ToyML dialect to standard MLIR dialects

4. **Add runtime support:**
   -  i.e runtime functions that will be called by the generated code

## Building and Running

### Prerequisites
- LLVM/MLIR  libraries
- Python with necessary lark packages 

### Building
```bash
# Navigate to mlir_backend directory
cd mlir_backend

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release
```

### Running
```bash
# Parse and transform ToyML to AST
python toyml_compiler.py <input_file.toyml>
```

## Example ToyML Code

```
dataset "data/mnist.csv" as train_data;
model = train(train_data, "RandomForest", {trees: 100});
print(model);
```

## Development Notes

- Dalect is started with basic types (DatasetType, ModelType, MetricType)
- Basic ops declarations are written but need implementation
- MLIR generation (`mlir_gen.py`) assumes implemented ops will be available
- Need to fully implement operation verification and custom assembly formats

## References

- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [LLVM Project](https://llvm.org/)
- [Lark Parser](https://github.com/lark-parser/lark)
