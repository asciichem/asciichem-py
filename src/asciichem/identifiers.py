"""Offline format validation for chemical identifier conventions
(CAS RN, InChI, InChIKey, SMILES sanity) - the Python mirror of
`AsciiChem::Identifiers`.

Validation is purely lexical/checksum based. It never touches the
network and never decides whether an identifier is *assigned* to a
real substance - that is resolution (TODO.v2 07 Layer 2), not
linting. Conventions without a validator (e.g. `iupac`, `cid`,
`chebi`) are simply not checkable offline: `diagnostic()` answers
None for them, like the reference.

Extension point: adding a convention is one validator class plus one
REGISTRY entry - nothing else changes."""
from __future__ import annotations

import re

from . import periodic_table


class Cas:
    """CAS Registry Number format and check-digit validation.

    A CAS RN is DDDDDDD-DD-D: 2-7 digits, hyphen, 2 digits, hyphen,
    one check digit. The check digit equals the weighted sum of the
    preceding digits (weights 1..n counted from the right) modulo 10.
    Reference for the published algorithm:
    github.com/simonengelke/CAS_Validation (MIT) - reimplemented,
    not copied."""

    FORMAT = re.compile(r"^\d{2,7}-\d{2}-\d$")

    @staticmethod
    def diagnostic(value) -> str | None:
        if not isinstance(value, str):
            return f"CAS RN must be a String, got {type(value).__name__}"
        if not Cas.FORMAT.match(value):
            return (f"CAS RN {value!r} is not in DDDDDDD-DD-D form "
                    "(2–7 digits, 2 digits, 1 check digit)")
        expected = Cas._weighted_sum(value) % 10
        if Cas._check_digit(value) == expected:
            return None
        return (f"CAS RN {value!r}: check digit {Cas._check_digit(value)} "
                f"does not match weighted sum ({expected})")

    @staticmethod
    def valid(value: str) -> bool:
        return Cas.diagnostic(value) is None

    @staticmethod
    def _digits(value: str) -> list[int]:
        return [int(c) for c in value.replace("-", "")]

    @staticmethod
    def _check_digit(value: str) -> int:
        return Cas._digits(value)[-1]

    @staticmethod
    def _weighted_sum(value: str) -> int:
        # Sum of each digit before the check digit, weighted by its
        # position counted from the right (the neighbour of the check
        # digit has weight 1).
        return sum(d * (i + 1) for i, d in
                   enumerate(reversed(Cas._digits(value)[:-1])))


class Inchi:
    """IUPAC International Chemical Identifier (InChI) lexical
    validation: versioned prefix plus a well-formed formula layer
    whose element symbols exist in the periodic table. Layers after
    the formula layer are not validated - full InChI semantics is
    the InChI software's job (TODO.v2 10), not a linter's."""

    PREFIX = re.compile(r"^InChI=1S?/")
    ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

    @staticmethod
    def diagnostic(value) -> str | None:
        if not isinstance(value, str):
            return f"InChI must be a String, got {type(value).__name__}"
        layer = Inchi.formula_layer(value)
        if layer is None:
            return (f"InChI {value!r} must start with "
                    "\"InChI=1/\" or \"InChI=1S/\"")
        counts, reason = Inchi._parse_formula_layer(layer)
        if counts is not None:
            return None
        return f"InChI {value!r}: {reason}"

    @staticmethod
    def valid(value: str) -> bool:
        return Inchi.diagnostic(value) is None

    @staticmethod
    def formula_layer(value) -> str | None:
        """The formula layer: the segment between the prefix and the
        first subsequent "/" (e.g. "C9H8O4" in
        "InChI=1S/C9H8O4/c1-6(10)..."), or None when the value lacks
        the versioned prefix. Single source of truth for offline
        InChI/molecule composition checks."""
        if not isinstance(value, str):
            return None
        match = Inchi.PREFIX.match(value)
        if match is None:
            return None
        rest = value[match.end():]
        return rest.split("/", 1)[0] if rest else ""

    @staticmethod
    def formula_counts(value: str) -> dict[str, int] | None:
        """Hash of element symbol => atom count from the formula layer,
        or None when the value is not analysable."""
        layer = Inchi.formula_layer(value)
        if layer is None:
            return None
        return Inchi._parse_formula_layer(layer)[0]

    @staticmethod
    def _parse_formula_layer(layer: str) -> tuple[dict[str, int] | None, str | None]:
        if layer == "":
            return None, "formula layer is empty"
        counts: dict[str, int] = {}
        rest = layer
        while rest:
            token = Inchi.ELEMENT_TOKEN.match(rest)
            if token is None or token.start() != 0:
                return None, f"malformed formula layer near {rest!r}"
            symbol = token.group(1)
            if not periodic_table.known(symbol):
                return None, f"unknown element symbol {symbol!r} in formula layer"
            count = int(token.group(2)) if token.group(2) else 1
            counts[symbol] = counts.get(symbol, 0) + count
            rest = rest[token.end():]
        return counts, None


class Inchikey:
    """InChIKey format validation: 27 uppercase letters in three blocks
    of 14, 10, and 1, separated by hyphens (e.g.
    BSYNRYMUTXBXSQ-UHFFFAOYSA-N for aspirin). Check-character
    verification is left to the InChI engine (TODO.v2 10) - format
    alone catches the common corruptions (case, block lengths,
    misplaced hyphens)."""

    FORMAT = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

    @staticmethod
    def diagnostic(value) -> str | None:
        if not isinstance(value, str):
            return f"InChIKey must be a String, got {type(value).__name__}"
        if Inchikey.FORMAT.match(value):
            return None
        return (f"InChIKey {value!r} must be 27 uppercase letters in "
                "XXXXXXXXXXXXXX-XXXXXXXXXX-X form (blocks of 14, 10, and 1)")

    @staticmethod
    def valid(value: str) -> bool:
        return Inchikey.diagnostic(value) is None


class Smiles:
    """SMILES sanity validation. This is deliberately *not* a SMILES
    parser (structure interchange has its own module) - it catches
    corruption: whitespace, unbalanced brackets and parentheses,
    unpaired ring-bond digits, and element tokens that are not in the
    periodic table.

    Lowercase aromatic atoms (c1ccccc1) are valid SMILES and pass;
    aromaticity only constrains future *ingestion* (Kekule-first),
    not identifier validity."""

    ORGANIC_SUBSET = ["B", "C", "N", "O", "P", "S", "F", "I", "Cl", "Br", "H"]
    AROMATIC_SUBSET = ["b", "c", "n", "o", "p", "s", "f", "cl", "br"]
    BRACKET_SYMBOL = re.compile(r"^\d*([A-Z][a-z]{0,2}|\*)")

    @staticmethod
    def diagnostic(value) -> str | None:
        if not isinstance(value, str):
            return f"SMILES must be a String, got {type(value).__name__}"
        if value == "":
            return "SMILES must not be empty"
        if re.search(r"\s", value):
            return f"SMILES {value!r} must not contain whitespace"
        reason = Smiles._structural_reasons(value)
        return reason or Smiles._element_reasons(value)

    @staticmethod
    def valid(value: str) -> bool:
        return Smiles.diagnostic(value) is None

    @staticmethod
    def element_set(value) -> list[str] | None:
        """Sorted unique element symbols appearing in the SMILES
        (bracket atoms and the organic/aromatic subsets, aromatic
        normalised to element symbols), or None when the value is not
        analysable. Single source of truth for offline SMILES/molecule
        element-set checks."""
        if not isinstance(value, str):
            return None
        if Smiles._structural_reasons(value) is not None:
            return None
        symbols: list[str] = []
        index = 0
        while index < len(value):
            index, analysable = Smiles._advance(value, index, symbols)
            if not analysable:
                return None
        return sorted(set(symbols))

    @staticmethod
    def _structural_reasons(value: str) -> str | None:
        if not Smiles._balanced(value, "[", "]"):
            return f"SMILES {value!r}: square brackets are not balanced"
        if not Smiles._balanced(value, "(", ")"):
            return f"SMILES {value!r}: parentheses are not balanced"
        # Ring digits only exist outside brackets - bracket interiors
        # hold isotope/H-count digits, which must not pair up.
        bare = re.sub(r"\[[^\]]*\]", "", value)
        digits = sorted({c for c in bare if c.isdigit()})
        for digit in digits:
            if bare.count(digit) % 2 == 1:
                return (f"SMILES {value!r}: ring-bond digit {digit} appears an "
                        "odd number of times (ring closures must pair)")
        return None

    @staticmethod
    def _balanced(value: str, opener: str, closer: str) -> bool:
        depth = 0
        for char in value:
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
            if depth < 0:
                return False
        return depth == 0

    @staticmethod
    def _element_reasons(value: str) -> str | None:
        if Smiles.element_set(value) is not None:
            return None
        return f"SMILES {value!r}: contains a malformed or unknown element token"

    @staticmethod
    def _advance(value: str, index: int, symbols: list[str]) -> tuple[int, bool]:
        char = value[index]
        if char == "[":
            return Smiles._consume_bracket_atom(value, index, symbols)
        if char.isalpha():
            return Smiles._consume_organic_atom(value, index, symbols)
        return index + 1, True

    @staticmethod
    def _consume_bracket_atom(value: str, index: int,
                              symbols: list[str]) -> tuple[int, bool]:
        close = value.find("]", index)
        if close == -1:
            return 0, False
        symbol = Smiles._bracket_atom_symbol(value[index + 1:close])
        if symbol is None:
            return 0, False
        symbols.append(symbol)
        return close + 1, True

    @staticmethod
    def _consume_organic_atom(value: str, index: int,
                              symbols: list[str]) -> tuple[int, bool]:
        token = Smiles._two_letter_token(value, index, value[index])
        if token is None:
            return 0, False
        symbols.append(token[0])
        return index + token[1], True

    @staticmethod
    def _two_letter_token(value: str, index: int, char: str) -> tuple[str, int] | None:
        two = value[index:index + 2]
        if two in Smiles.ORGANIC_SUBSET:
            return two, 2
        if two in Smiles.AROMATIC_SUBSET:
            return two.capitalize(), 2
        if char in Smiles.ORGANIC_SUBSET:
            return char, 1
        if char in Smiles.AROMATIC_SUBSET:
            return char.upper(), 1
        return None

    @staticmethod
    def _bracket_atom_symbol(content: str) -> str | None:
        match = Smiles.BRACKET_SYMBOL.match(content)
        if match is None:
            return None
        symbol = match.group(1)
        if symbol == "*":
            return None
        return symbol if periodic_table.known(symbol) else None


# The convention registry: validator classes by convention name. The
# linter consults known()/diagnostic(); adding a convention is one
# class plus one entry here (OCP).
REGISTRY: dict[str, type] = {
    "cas": Cas,
    "inchi": Inchi,
    "inchikey": Inchikey,
    "smiles": Smiles,
}


def _normalize(convention) -> str:
    return str(convention)


def validator(convention) -> type | None:
    return REGISTRY.get(_normalize(convention))


def known(convention) -> bool:
    """Whether a validator is registered for the convention. Unknown
    conventions are not errors - they are simply not checkable
    offline (e.g. `iupac`, `cid`, `chebi`)."""
    return _normalize(convention) in REGISTRY


def valid(convention, value) -> bool:
    found = validator(convention)
    return found is not None and found.valid(value)


def diagnostic(convention, value) -> str | None:
    """None when the value is valid (or the convention is unknown);
    a human-readable reason otherwise."""
    found = validator(convention)
    return found.diagnostic(value) if found else None
