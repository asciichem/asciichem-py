"""Semantic model - the Python mirror of AsciiChem::Model. Plain
dataclasses; the Text formatter and wire form are views over the same
tree (model-driven, one tree for every formatter)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Union

Node = Union["Atom", "Bond", "Group", "Molecule", "Reaction", "ReactionCascade",
             "ElectronConfiguration", "EmbeddedMath", "Text",
             "Crystal", "Spectrum", "Calculation", "ZMatrix", "Mechanism"]

BOND_ASCII = {"single": "-", "double": "=", "triple": "#", "quadruple": "##",
              "wedge": ">-", "hash": "-<", "dative": "~>", "wavy": "~~",
              "aromatic": ":"}
BOND_ENTITY = {"single": "-", "double": "=", "triple": "≡", "quadruple": "≣",
               "wedge": "↑", "hash": "↓", "dative": "→", "wavy": "∼",
               "aromatic": ":"}
ARROW_ASCII = {"forward": "->", "reverse": "<-", "equilibrium": "<=>", "resonance": "<->"}
ARROW_ENTITY = {"forward": "→", "reverse": "←", "equilibrium": "⇌", "resonance": "↔"}
BRACKETS = {"paren": "()", "square": "[]", "brace": "{}"}
STEREO_LETTER = {"R": "R", "S": "S", "E": "E", "Z": "Z", "alpha": "alpha", "beta": "beta"}


class AsciiChemError(Exception):
    pass


class ParseError(AsciiChemError):
    pass


@dataclass
class Atom:
    element: str
    isotope: Optional[str] = None
    subscript: Optional[str] = None
    superscript: Optional[str] = None
    charge: Optional[str] = None
    oxidation_state: Optional[str] = None
    lone_pairs: Optional[int] = None
    radical_electrons: Optional[int] = None
    ring_closures: Optional[str] = None
    aromatic: Optional[bool] = None
    hydrogens: Optional[int] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    z2: Optional[float] = None
    atom_parity: Optional[str] = None
    chirality: Optional[str] = None
    spin_multiplicity: Optional[str] = None
    atom_title: Optional[str] = None
    x_fract: Optional[float] = None
    y_fract: Optional[float] = None
    z_fract: Optional[float] = None

    def diagnostic_label(self) -> str:
        """Human-readable label for linter diagnostics ("Atom(C)")."""
        return f"Atom({self.element})"


@dataclass
class Bond:
    kind: str = "single"
    # SMILES directional single bond (/ up, \ down) flanking a
    # stereogenic double bond — orthogonal to the kind.
    direction: Optional[str] = None

    @property
    def ascii(self) -> str:
        return BOND_ASCII[self.kind]

    @property
    def entity(self) -> str:
        return BOND_ENTITY[self.kind]


@dataclass
class Group:
    nodes: list
    multiplicity: Optional[str] = None
    bracket: str = "paren"

    @property
    def open_char(self) -> str:
        return BRACKETS[self.bracket][0]

    @property
    def close_char(self) -> str:
        return BRACKETS[self.bracket][1]


@dataclass
class Identifier:
    value: str
    convention: str


@dataclass
class Name:
    content: str


@dataclass
class Molecule:
    nodes: list
    coefficient: Optional[str] = None
    stereo: Optional[str] = None
    names: list = field(default_factory=list)
    identifiers: list = field(default_factory=list)
    title: Optional[str] = None
    formulas: list = field(default_factory=list)
    properties: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    metadata: list = field(default_factory=list)

    def stoichiometric_coefficient(self) -> int:
        """The integer stoichiometric coefficient: the authored leading
        number, or 1 when absent or not positive. Single source for
        every consumer (reaction stoichiometry, the balance check,
        composition) - coefficients are model data, not per-caller
        parsing."""
        own = int(self.coefficient) if self.coefficient else None
        return own if own and own > 0 else 1

    def element_counts(self, with_coefficient: bool = True) -> dict[str, int]:
        """Element symbol => total atom count, recursing through groups
        and nested molecules with subscripts and multiplicities
        applied. Pass with_coefficient=False for the composition of a
        single molecule instance - identifiers describe the substance,
        not the stoichiometric multiplier."""
        multiplier = self.stoichiometric_coefficient() if with_coefficient else 1
        tally: dict[str, int] = {}
        for node in self.nodes:
            _tally_element(node, tally, multiplier)
        return tally

    def hill_formula(self) -> str:
        """Hill-system canonical formula: C first, then H, then others
        alphabetically. Single source of truth for "what is the
        canonical formula string of this molecule"
        (`H_2SO_4` -> "H2O4S" - no carbon, alpha sort)."""
        counts = self.element_counts()
        ordered: list[str] = []
        for priority in ("C", "H"):
            if priority in counts:
                ordered.append(priority)
        ordered.extend(e for e in sorted(counts) if e not in ("C", "H"))
        return "".join(f"{e}{counts[e]}" if counts[e] != 1 else e
                       for e in ordered)


def _leading_integer(raw: Optional[str]) -> int:
    """The leading integer of a digit string (Ruby's String#to_i
    semantics: leading digits, 0 when none)."""
    digits = ""
    for char in str(raw or ""):
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else 0


def _tally_element(node, tally: dict[str, int], multiplier: int) -> None:
    if isinstance(node, Atom):
        sub = _leading_integer(node.subscript) if node.subscript else 0
        count = (sub if sub > 0 else 1) * multiplier
        tally[node.element] = tally.get(node.element, 0) + count
    elif isinstance(node, Group):
        mult = _leading_integer(node.multiplicity) if node.multiplicity else 0
        for inner in node.nodes:
            _tally_element(inner, tally, multiplier * (mult if mult > 0 else 1))
    elif isinstance(node, Molecule):
        inner_coeff = node.stoichiometric_coefficient()
        for inner in node.nodes:
            _tally_element(inner, tally, multiplier * inner_coeff)


CONDITION_ROLE_PREFIXES = {
    "cat": "catalyst", "catalyst": "catalyst",
    "solv": "solvent", "solvent": "solvent",
    "reag": "reagent", "reagent": "reagent",
}


@dataclass
class ConditionEntry:
    role: Optional[str]
    value: str
    side: str


def _entries_for(side: str, text: Optional[str]) -> list[ConditionEntry]:
    if text is None or not text.strip():
        return []
    out: list[ConditionEntry] = []
    for chunk in text.split(";"):
        trimmed = chunk.strip()
        if not trimmed:
            continue
        prefix, colon, rest = trimmed.partition(":")
        role = CONDITION_ROLE_PREFIXES.get(prefix.strip().lower()) if colon else None
        out.append(ConditionEntry(role=role,
                                  value=rest.strip() if role else trimmed,
                                  side=side))
    return out


@dataclass
class ReactionConditions:
    above: Optional[str] = None
    below: Optional[str] = None

    @property
    def entries(self) -> list[ConditionEntry]:
        """Typed `role: value` entries derived from the bracket text
        (which stays verbatim - the round-trip source). Semicolons
        separate entries; unknown role prefixes are free text (role
        None), never an error."""
        return (_entries_for("above", self.above)
                + _entries_for("below", self.below))


@dataclass
class Reaction:
    reactants: list
    products: list
    arrow: str = "forward"
    conditions: Optional[ReactionConditions] = None

    def diagnostic_label(self) -> str:
        """Human-readable label for linter diagnostics (cascade steps
        are bare `Reaction`s, so they carry the label too)."""
        return "Reaction"

    @property
    def arrow_ascii(self) -> str:
        return ARROW_ASCII[self.arrow]

    @property
    def arrow_entity(self) -> str:
        return ARROW_ENTITY[self.arrow]


@dataclass
class ReactionCascade:
    steps: list


@dataclass
class ElectronConfiguration:
    orbitals: list  # list[(orbital, occupancy)]
    term_symbol: Optional[tuple] = None  # (multiplicity, letter, j)


@dataclass
class EmbeddedMath:
    source: str


@dataclass
class Text:
    content: str


@dataclass
class Crystal:
    name: Optional[str] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    spacegroup: Optional[str] = None
    atoms: list = field(default_factory=list)

    # The six unit-cell parameters in canonical iteration order, split
    # lengths vs angles - the linter iterates these without
    # re-deriving subsets (the reference's LENGTH/ANGLE_FIELDS).
    CELL_PARAMS = ("a", "b", "c", "alpha", "beta", "gamma")
    LENGTH_FIELDS = ("a", "b", "c")
    ANGLE_FIELDS = ("alpha", "beta", "gamma")

    def diagnostic_label(self) -> str:
        return f"Crystal({self.name or 'unnamed'})"


@dataclass
class SpectrumPeak:
    position: Optional[str] = None
    intensity: Optional[str] = None
    multiplicity: Optional[str] = None
    assignment: Optional[str] = None


@dataclass
class Spectrum:
    technique: Optional[str] = None
    params: dict = field(default_factory=dict)
    peaks: list = field(default_factory=list)

    def diagnostic_label(self) -> str:
        return f"Spectrum({self.technique or 'unknown'})"


@dataclass
class CalculationProperty:
    title: str
    value: str
    units: Optional[str] = None


@dataclass
class Calculation:
    method: Optional[str] = None
    basis: Optional[str] = None
    properties: list = field(default_factory=list)

    def diagnostic_label(self) -> str:
        return f"Calculation({self.method}/{self.basis})"


@dataclass
class ZRow:
    atom: str
    ref1: Optional[str] = None
    distance: Optional[str] = None
    ref2: Optional[str] = None
    angle: Optional[str] = None
    ref3: Optional[str] = None
    dihedral: Optional[str] = None


@dataclass
class ZMatrix:
    rows: list = field(default_factory=list)

    def diagnostic_label(self) -> str:
        return f"ZMatrix({len(self.rows)} rows)"


@dataclass
class MechanismFlow:
    source: str
    target: str


@dataclass
class MechanismStep:
    label: str
    reaction: str
    flows: list = field(default_factory=list)


@dataclass
class Mechanism:
    steps: list = field(default_factory=list)
    spectators: list = field(default_factory=list)

    def diagnostic_label(self) -> str:
        return f"Mechanism({len(self.steps)} steps)"


@dataclass
class Formula:
    nodes: list


def diagnostic_label(node) -> str:
    """Human-readable label for linter diagnostics — the Python mirror
    of `Model::Node#diagnostic_label`. Nodes with identifying fields
    override (`Atom` identifies itself by element, `Crystal` by name,
    `Reaction` stays bare); the default derives from the class name
    (snake-case words, capitalised: `ReactionCascade` ->
    "Reaction Cascade"). Keeping the override on the model keeps the
    linter OCP-clean - adding a new model class means overriding the
    method on that class, not editing a linter case statement."""
    override = getattr(node, "diagnostic_label", None)
    if callable(override):
        return override()
    return " ".join(re.findall(r"[A-Z][a-z]*", type(node).__name__))
