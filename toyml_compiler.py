#!/usr/bin/env python3

import sys
import os
import subprocess
from pathlib import Path
import argparse

# Add parent directory to path so we can import from lexer_parser and transformer
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lexer_parser.parser import parse_file
from Transformer.transformer import transform_ast
from mlir_backend.mlir_gen import generate_mlir

def compile_toyml(input_file, output_file=None, emit_mlir=False, optimize=False):
    """
    Compile a ToyML DSL file to MLIR or execute it.
    
    Args:
        input_file: Path to the input ToyML file
        output_file: Path to the output file (optional)
        emit_mlir: Whether to emit MLIR instead of executing
        optimize: Whether to run optimizations
    """
    print(f"Compiling {input_file}...")
    
    # Parse the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    ast = parse_file(content)
    print("Parsed AST successfully.")
    
    # Transform the AST
    transformed_ast = transform_ast(ast)
    print("Transformed AST successfully.")
    
    # Generate MLIR
    mlir_output = generate_mlir(transformed_ast)
    print("Generated MLIR successfully.")
    
    # Determine output location
    if output_file:
        out_path = output_file
    else:
        out_path = Path(input_file).with_suffix('.mlir')
    
    # Write MLIR to file
    with open(out_path, 'w') as f:
        f.write(mlir_output)
    print(f"Wrote MLIR to {out_path}")
    
    if emit_mlir:
        return
    
    # Build paths to our compiler executable
    mlir_backend_dir = Path(__file__).resolve().parent
    compiler_path = mlir_backend_dir / 'build' / 'bin' / 'toyml-compiler'
    if os.name == 'nt':  # Windows
        compiler_path = mlir_backend_dir / 'build' / 'bin' / 'Release' / 'toyml-compiler.exe'
    
    # Execute the compiler on the generated MLIR
    cmd = [str(compiler_path)]
    if optimize:
        cmd.append("--mlir-print-ir-after-all")
    cmd.append(str(out_path))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running compiler: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser(description='ToyML Compiler')
    parser.add_argument('input', help='Input ToyML file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--emit-mlir', action='store_true', help='Emit MLIR only')
    parser.add_argument('--optimize', action='store_true', help='Run optimizations')
    
    args = parser.parse_args()
    compile_toyml(args.input, args.output, args.emit_mlir, args.optimize)

if __name__ == "__main__":
    main()
