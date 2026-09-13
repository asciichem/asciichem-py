"""asciichem - Python implementation of the AsciiChem contract.

parse("H_2O").to_text()      canonical AsciiChem text (round-trip contract)
parse("H_2O").to_model_json()   asciichem-model v1 wire JSON
"""
from .model import AsciiChemError, ParseError
from .parser import parse_text
from .text import render
from .wire import from_model_json, to_model_json, to_wire

__version__ = "0.1.0"
__all__ = ["parse_text", "from_model_json", "to_model_json", "to_wire",
           "AsciiChemError", "ParseError", "__version__"]


def _render(node):
    return render(node)


# Convenience: attach to_text()/to_model_json() to nodes dynamically is
# un-Pythonic; expose a parse() that returns the Formula plus helpers.
def parse(source: str):
    return parse_text(source)
