# tests/test_dsl_parser.py

import pytest
from parser import parser   # adjust this import to match your project

# ——————————————————————————————
# Valid DSL snippets (should parse without error)
# ——————————————————————————————
valid_scripts = [
    # Simple dataset declaration
    'Dataset raw from "data.csv"\n',
    # Simple load command
    'Load "file.csv" as data_copy\n',
    # Split with comma-separated list and optional with clause
    'Split raw into train, test with ratio=0.8\n',
    # Train without params
    'Train model\n',
    # Train with params
    'Train model with epochs=5, lr=0.001\n',
    # Evaluate command
    'Evaluate model on test_set\n',
    # Print statement
    'Print summarize(model)\n',
    # Comment and import
    '# This is a comment\nimport utils\n',
    # Pipeline with two statements
    (
        'Dataset raw from "data.csv"\n'
        'Pipeline Prep:\n'
        '    Load "data.csv" as temp\n'
        '    Split temp into a, b with ratio=0.5\n'
    ),
    # Script with blank lines and trailing newlines
    (
        '\n'
        'Dataset raw from "data.csv"\n'
        '\n'
        'Load "file.csv"\n'
        '\n'
    ),
]

# ——————————————————————————————
# Invalid DSL snippets (should raise a parsing error)
# ——————————————————————————————
invalid_scripts = [
    # Missing quotes around filename
    'Load data.csv as raw\n',
    # Missing IDENT in Train
    'Train\n',
    # Missing "on" keyword
    'Evaluate model test\n',
    # Malformed split (no targets)
    'Split raw into with ratio=0.8\n',
    # Pipeline without indent
    (
        'Pipeline X:\n'
        'Load "a.csv" as x\n'
    ),
    # Extra comma in list
    'Split raw into a, b, with ratio=0.8\n',
]

@pytest.mark.parametrize("dsl", valid_scripts)
def test_valid_scripts(dsl):
    """Valid DSL snippets should parse without errors."""
    parser.parse(dsl)

@pytest.mark.parametrize("dsl", invalid_scripts)
def test_invalid_scripts(dsl):
    """Invalid DSL snippets should raise a parsing exception."""
    with pytest.raises(Exception):
        parser.parse(dsl)
