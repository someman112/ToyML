from transformer import ASTBuilder

from lark import Lark
from lark.indenter import Indenter

class DSLOutOfTheBoxIndenter(Indenter):
    NL_type        = "_NL"
    OPEN_PAREN_types  = []
    CLOSE_PAREN_types = []
    INDENT_type    = "_INDENT"
    DEDENT_type    = "_DEDENT"
    tab_len        = 4


import os

# Get the absolute path to the lexer_parser directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grammar_path = os.path.join(parent_dir, "lexer_parser", "dsl.lark")

parser = Lark.open(
    grammar_path,
    parser="lalr",
    postlex=DSLOutOfTheBoxIndenter(),
    start="start",
)

dsl_source = '''\
Dataset raw from "data.csv"
Load "data.csv" as backup

Pipeline Prep:
    Split raw into train, test with ratio=0.8
    Load "test.csv" as t2

Train train with epochs=10
Train train using linear_regression with epochs=5, lr=0.01
Evaluate model on test
Print summarize(model)
'''

tree = parser.parse(dsl_source)
ast = ASTBuilder().transform(tree)
print(ast)
