from lark import Lark
from lark.indenter import Indenter

class DSLOutOfTheBoxIndenter(Indenter):
    NL_type        = "_NL"
    OPEN_PAREN_types  = []
    CLOSE_PAREN_types = []
    INDENT_type    = "_INDENT"
    DEDENT_type    = "_DEDENT"
    tab_len        = 4


parser = Lark.open(
    "dsl.lark",
    parser="lalr",
    postlex=DSLOutOfTheBoxIndenter(),
    start="start",
)

text = '''\
Dataset raw from "data.csv"
Load "data.csv" as backup

Pipeline Prep:
    Split raw into train, test with ratio=0.8
    Load "test.csv" as t2

Train train with epochs=10
Evaluate model on test
Print summarize(model)
'''

tree = parser.parse(text)
print(tree.pretty())

