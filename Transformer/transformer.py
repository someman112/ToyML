from lark import Transformer, v_args
from dataclasses import dataclass
from typing import List, Dict, Union

# --- AST Node Defn ---
@dataclass
class DatasetNode:
    name: str
    path: str

@dataclass
class LoadNode:
    path: str
    alias: Union[str, None] = None

@dataclass
class SplitNode:
    source: str
    targets: List[str]
    params: Dict[str, Union[int, float, str]]

@dataclass
class TrainNode:
    source: str
    model_type: Union[str, None]
    params: Dict[str, Union[int, float, str]]

@dataclass
class EvaluateNode:
    model: str
    on: str

@dataclass
class PrintNode:
    expr: Union[str, Dict]

@dataclass
class PipelineNode:
    name: str
    body: List

# --- Transformer ---
@v_args(inline=True)   # children as args
class ASTBuilder(Transformer):
    def dataset_stmt(self, name, path):
        return DatasetNode(name, path)
    def load_cmd(self, path, alias=None):
        return LoadNode(path, alias)
    def split_cmd(self, source, *rest):
        params = {}
        targets = []
        for item in rest:
            if isinstance(item, dict):
                params = item
            else:
                targets.append(item)
        return SplitNode(source, targets, params)
    def train_cmd(self, source, *rest):
        model_type = None
        params = {}
        
        for item in rest:
            if isinstance(item, dict):
                params = item
            else:
                model_type = item
                
        return TrainNode(source, model_type, params or {})
    def eval_cmd(self, model, on):
        return EvaluateNode(model, on)
    def print_stmt(self, expr):
        return PrintNode(expr)
    def pipeline_stmt(self, name, *stmts):
        return PipelineNode(name, list(stmts))
    def param_list(self, *params):
        return dict(params)
    def param(self, key, val):
        return (key, val)
    def expr(self, *args):
        if len(args) == 1:
            return args[0]
        fn, arglist = args
        return {"call": fn, "args": arglist}
    def arg_list(self, *args):
        return list(args)
    def arg(self, *vals):
        return vals[0] if len(vals) == 1 else (vals[0], vals[1])
    def IDENT(self, tk):  return str(tk)
    def STRING(self, tk): return str(tk)[1:-1]
    def NUMBER(self, tk):
        s = str(tk)
        return float(s) if '.' in s else int(s)
