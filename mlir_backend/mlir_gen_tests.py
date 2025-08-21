import sys
import os
from pathlib import Path
import unittest

# Add parent directory to path so we can import transformer
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(str(parent_dir))

from mlir_gen import MLIREmitter, generate_mlir
from Transformer.transformer import (
    DatasetNode, LoadNode, SplitNode, TrainNode, 
    EvaluateNode, PrintNode, PipelineNode
)

# Import parser components to demonstrate end-to-end testing
from lark import Lark
from lark.indenter import Indenter
from Transformer.transformer import ASTBuilder

class DSLOutOfTheBoxIndenter(Indenter):
    NL_type = "_NL"
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4

class MLIRGenTests(unittest.TestCase):
    
    def setUp(self):
        # Setup parser for end-to-end tests
        grammar_path = os.path.join(parent_dir, "lexer_parser", "dsl.lark")
        self.parser = Lark.open(
            grammar_path,
            parser="lalr",
            postlex=DSLOutOfTheBoxIndenter(),
            start="start",
        )
    
    def test_dataset_node(self):
        """Test MLIR generation for a DatasetNode."""
        node = DatasetNode(name="mnist", path="mnist.csv")
        emitter = MLIREmitter()
        mlir = emitter.emit_mlir([node])
        
        # Check if the generated MLIR contains the dataset definition
        self.assertIn('= toyml.dataset "mnist.csv" as "mnist" : !toyml.dataset', mlir)
    
    def test_load_node(self):
        """Test MLIR generation for a LoadNode."""
        node = LoadNode(path="data.csv", alias="data")
        emitter = MLIREmitter()
        mlir = emitter.emit_mlir([node])
        
        # Check if the generated MLIR contains the load operation
        self.assertIn('= toyml.dataset "data.csv" as "data" : !toyml.dataset', mlir)
    
    def test_split_node(self):
        """Test MLIR generation for a SplitNode."""
        dataset = DatasetNode(name="data", path="data.csv")
        split = SplitNode(source="data", targets=["train", "test"], params={"ratio": 0.8})
        
        emitter = MLIREmitter()
        mlir = emitter.emit_mlir([dataset, split])
        
        # Check if the generated MLIR contains the split operation with parameters
        self.assertIn('= toyml.split %0 into ["train", "test"] with {ratio = 0.8}', mlir)
    
    def test_train_node(self):
        """Test MLIR generation for a TrainNode."""
        dataset = DatasetNode(name="train", path="train.csv")
        train = TrainNode(source="train", model_type="linear_regression", params={"epochs": 5, "lr": 0.01})
        
        emitter = MLIREmitter()
        mlir = emitter.emit_mlir([dataset, train])
        
        # Check if the generated MLIR contains the train operation
        self.assertIn('= toyml.train %0 using "linear_regression" with {epochs = 5, lr = 0.01}', mlir)
    
    def test_evaluate_node(self):
        """Test MLIR generation for an EvaluateNode."""
        dataset = DatasetNode(name="test", path="test.csv")
        train = TrainNode(source="train", model_type=None, params={})
        evaluate = EvaluateNode(model="model", on="test")
        
        # We need to set up the symbol table manually for this test
        emitter = MLIREmitter()
        emitter.symbol_table["model"] = "%1"  # Simulate model from TrainNode
        mlir = emitter.emit_mlir([dataset, train, evaluate])
        
        # Check if the generated MLIR contains the evaluate operation
        self.assertIn('= toyml.evaluate %1 on %0 : !toyml.model, !toyml.dataset -> f32', mlir)
    
    def test_print_node_direct(self):
        """Test MLIR generation for a PrintNode with direct variable."""
        dataset = DatasetNode(name="data", path="data.csv")
        print_node = PrintNode(expr="data")
        
        emitter = MLIREmitter()
        mlir = emitter.emit_mlir([dataset, print_node])
        
        # Check if the generated MLIR contains the print operation
        self.assertIn('toyml.print %0 : !toyml.model', mlir)
    
    def test_print_node_function_call(self):
        """Test MLIR generation for a PrintNode with function call."""
        dataset = DatasetNode(name="data", path="data.csv")
        train = TrainNode(source="data", model_type=None, params={})
        print_node = PrintNode(expr={"call": "summarize", "args": ["model"]})
        
        # Setup symbol table
        emitter = MLIREmitter()
        emitter.symbol_table["model"] = "%1"  # Simulate model from TrainNode
        mlir = emitter.emit_mlir([dataset, train, print_node])
        
        # Check for function call and print
        self.assertIn('call @summarize(%1) : (!toyml.model) -> !toyml.summary', mlir)
        self.assertIn('toyml.print %2 : !toyml.summary', mlir)
    
    def test_pipeline_node(self):
        """Test MLIR generation for a PipelineNode."""
        pipeline = PipelineNode(
            name="Prep", 
            body=[
                SplitNode(source="data", targets=["train", "test"], params={"ratio": 0.8})
            ]
        )
        
        # Setup for pipeline test
        emitter = MLIREmitter()
        emitter.symbol_table["data"] = "%0"  # Simulate dataset
        mlir = emitter.emit_mlir([pipeline])
        
        # Check pipeline structure
        self.assertIn('toyml.pipeline "Prep" {', mlir)
        self.assertIn('= toyml.split %0 into ["train", "test"] with {ratio = 0.8}', mlir)
        self.assertIn('toyml.pipeline_end', mlir)
    
    def test_end_to_end(self):
        """Test end-to-end workflow: parse DSL source to AST then to MLIR."""
        dsl_source = '''\
Dataset mnist from "mnist.csv"

Pipeline Preprocess:
    Split mnist into train, test with ratio=0.8

Train train with epochs=10
Evaluate model on test
Print summarize(model)
'''
        # Parse to AST
        tree = self.parser.parse(dsl_source)
        ast = ASTBuilder().transform(tree)
        
        # The transform returns a Tree with nodes as children, extract them
        nodes = ast.children if hasattr(ast, 'children') else [ast]
        
        # Generate MLIR
        mlir = generate_mlir(nodes)
        
        # Basic structure checks
        self.assertIn('module {', mlir)
        self.assertIn('func.func @main() {', mlir)
        self.assertIn('return', mlir)
        
        # Check key operations
        self.assertIn('= toyml.dataset "mnist.csv" as "mnist"', mlir)
        self.assertIn('toyml.pipeline "Preprocess"', mlir)
        self.assertIn('= toyml.split', mlir)
        self.assertIn('= toyml.train', mlir)
        self.assertIn('= toyml.evaluate', mlir)
        self.assertIn('call @summarize', mlir)
        self.assertIn('toyml.print', mlir)
    
    def test_symbol_table_tracking(self):
        """Test that symbol table correctly tracks variables across operations."""
        # Create a workflow with variable dependencies
        ast = [
            DatasetNode(name="data", path="data.csv"),
            SplitNode(source="data", targets=["train", "val"], params={"ratio": 0.7}),
            TrainNode(source="train", model_type=None, params={"epochs": 10}),
            EvaluateNode(model="model", on="val")
        ]
        
        # Generate MLIR
        mlir = generate_mlir(ast)
        
        # Check that variables are referenced correctly
        self.assertIn('= toyml.dataset "data.csv" as "data"', mlir)
        # The split should reference the dataset variable
        self.assertIn('= toyml.split %0 into ["train", "val"]', mlir)
        # The train should reference the first output of the split (train dataset)
        self.assertIn('= toyml.train %1 with {epochs = 10}', mlir)
        # The evaluate should reference the model from training and val dataset
        self.assertIn('= toyml.evaluate %3 on %2', mlir)

if __name__ == "__main__":
    unittest.main()
