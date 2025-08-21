import sys
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

# Add parent directory to path so we can import transformer
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from Transformer.transformer import (
    DatasetNode, LoadNode, SplitNode, TrainNode, 
    EvaluateNode, PrintNode, PipelineNode
)

class MLIREmitter:
    def __init__(self):
        self.buffer = []
        self.indent_level = 0
        self.var_counter = 0
        self.symbol_table = {}  # Maps variable names to SSA values
    
    def emit(self, line: str):
        """Add a line to the MLIR output with proper indentation."""
        self.buffer.append("  " * self.indent_level + line)
    
    def fresh_var(self) -> str:
        """Generate a fresh SSA variable."""
        var = f"%{self.var_counter}"
        self.var_counter += 1
        return var
    
    def emit_module_header(self):
        """Emit the MLIR module header."""
        self.emit("module {")
        self.indent_level += 1
        self.emit("func.func @main() {")
        self.indent_level += 1
    
    def emit_module_footer(self):
        """Emit the MLIR module footer."""
        self.emit("return")
        self.indent_level -= 1
        self.emit("}")
        self.indent_level -= 1
        self.emit("}")
    
    def visit(self, node):
        """Visit an AST node and emit corresponding MLIR."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        """Default handler for node types without specific visitors."""
        raise NotImplementedError(f"No visitor for {type(node).__name__}")
    
    def visit_DatasetNode(self, node: DatasetNode) -> str:
        """Emit MLIR for a dataset node."""
        var = self.fresh_var()
        self.emit(f'{var} = toyml.dataset "{node.path}"' + 
                 (f' as "{node.name}"' if node.name else '') + 
                 f' : !toyml.dataset')
        if node.name:
            self.symbol_table[node.name] = var
        return var
    
    def visit_LoadNode(self, node: LoadNode) -> str:
        """Emit MLIR for a load node."""
        var = self.fresh_var()
        # Using dataset op for load, as they are similar in function
        self.emit(f'{var} = toyml.dataset "{node.path}"' + 
                 (f' as "{node.alias}"' if node.alias else '') + 
                 f' : !toyml.dataset')
        if node.alias:
            self.symbol_table[node.alias] = var
        return var
    
    def visit_SplitNode(self, node: SplitNode) -> List[str]:
        """Emit MLIR for a split node."""
        source_var = self.symbol_table.get(node.source, node.source)
        if not source_var.startswith('%'):
            source_var = f"%{source_var}"  # Ensure it's an SSA variable
            
        target_vars = [self.fresh_var() for _ in node.targets]
        
        # Prepare params as a dictionary string
        param_str = ", ".join(f'{k} = {self._format_value(v)}' for k, v in node.params.items())
        
        # Format targets as an array
        targets_str = '[' + ', '.join(f'"{t}"' for t in node.targets) + ']'
        
        self.emit(f'{", ".join(target_vars)} = toyml.split {source_var} into {targets_str}' +
                 (f' with {{{param_str}}}' if node.params else '') + 
                 f' : !toyml.dataset -> {", ".join("!toyml.dataset" for _ in target_vars)}')
        
        # Update symbol table with target variables
        for i, target in enumerate(node.targets):
            self.symbol_table[target] = target_vars[i]
        
        return target_vars
    
    def visit_TrainNode(self, node: TrainNode) -> str:
        """Emit MLIR for a train node."""
        var = self.fresh_var()
        source_var = self.symbol_table.get(node.source, node.source)
        if not source_var.startswith('%'):
            source_var = f"%{source_var}"  # Ensure it's an SSA variable
        
        # Prepare model type and parameters
        model_type_str = f' using "{node.model_type}"' if node.model_type else ''
        
        # Prepare params as a dictionary string
        param_str = ", ".join(f'{k} = {self._format_value(v)}' for k, v in node.params.items())
        
        self.emit(f'{var} = toyml.train {source_var}{model_type_str}' +
                 (f' with {{{param_str}}}' if node.params else '') + 
                 f' : !toyml.dataset -> !toyml.model')
        
        # Default model name for reference by other nodes
        self.symbol_table['model'] = var
        
        return var
    
    def visit_EvaluateNode(self, node: EvaluateNode) -> str:
        """Emit MLIR for an evaluate node."""
        var = self.fresh_var()
        model_var = self.symbol_table.get(node.model, node.model)
        if not model_var.startswith('%'):
            model_var = f"%{model_var}"  # Ensure it's an SSA variable
            
        on_var = self.symbol_table.get(node.on, node.on)
        if not on_var.startswith('%'):
            on_var = f"%{on_var}"  # Ensure it's an SSA variable
        
        self.emit(f'{var} = toyml.evaluate {model_var} on {on_var} : !toyml.model, !toyml.dataset -> f32')
        
        # Store as metrics for reference by other nodes
        self.symbol_table['metrics'] = var
        
        return var
    
    def visit_PrintNode(self, node: PrintNode) -> None:
        """Emit MLIR for a print node."""
        # Handle either a direct variable or a function call expression
        if isinstance(node.expr, dict) and 'call' in node.expr:
            # This is a function call like summarize(model)
            fn_name = node.expr['call']
            args = node.expr['args']
            
            # Convert each arg to its SSA variable
            arg_vars = []
            for arg in args:
                arg_var = self.symbol_table.get(arg, arg)
                if not arg_var.startswith('%'):
                    arg_var = f"%{arg_var}"  # Ensure it's an SSA variable
                arg_vars.append(arg_var)
            
            # Create a call to the function
            result_var = self.fresh_var()
            arg_types = ", ".join(["!toyml.model" for _ in arg_vars])  # Assuming model type for simplicity
            self.emit(f'{result_var} = call @{fn_name}({", ".join(arg_vars)}) : ({arg_types}) -> !toyml.summary')
            
            # Print the result
            self.emit(f'toyml.print {result_var} : !toyml.summary')
        else:
            # This is a direct variable reference
            expr_var = self.symbol_table.get(node.expr, node.expr)
            if not expr_var.startswith('%'):
                expr_var = f"%{expr_var}"  # Ensure it's an SSA variable
            
            # Infer type from variable name (simple heuristic)
            type_str = "!toyml.model"
            if "metrics" in expr_var or expr_var in self.symbol_table and self.symbol_table[expr_var] == "metrics":
                type_str = "f32"
            
            self.emit(f'toyml.print {expr_var} : {type_str}')
    
    def visit_PipelineNode(self, node: PipelineNode) -> None:
        """Emit MLIR for a pipeline node."""
        # Save current state for symbol table
        old_symbols = self.symbol_table.copy()
        
        # Emit pipeline operation with region
        self.emit(f'toyml.pipeline "{node.name}" {{')
        self.indent_level += 1
        
        # Visit all statements in the pipeline body
        for stmt in node.body:
            self.visit(stmt)
        
        # Add the pipeline end operation
        self.emit(f'toyml.pipeline_end')
        
        # Restore indentation and close the region
        self.indent_level -= 1
        self.emit('}')
        
        # Merge symbols defined in pipeline with outer scope
        # Note: In a real implementation, you might want more careful scoping
        self.symbol_table.update(old_symbols)
    
    def _format_value(self, value: Any) -> str:
        """Format a Python value for MLIR."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value)
    
    def emit_mlir(self, nodes: List[Any]) -> str:
        """Emit MLIR for a list of AST nodes."""
        self.emit_module_header()
        
        for node in nodes:
            self.visit(node)
        
        self.emit_module_footer()
        return "\n".join(self.buffer)

def generate_mlir(ast: List[Any]) -> str:
    """Generate MLIR from an AST."""
    emitter = MLIREmitter()
    return emitter.emit_mlir(ast)
