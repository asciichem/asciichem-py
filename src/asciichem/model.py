"""Semantic model - the Python mirror of AsciiChem::Model. Plain
dataclasses; the Text formatter and wire form are views over the same
tree (model-driven, one tree for every formatter)."""
from __future__ import annotations

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
    spin_multiplicity: Optional[str] = None
    atom_title: Optional[str] = None
    x_fract: Optional[float] = None
    y_fract: Optional[float] = None
    z_fract: Optional[float] = None


@dataclass
class Bond:
    kind: str = "single"

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


@dataclass
class ReactionConditions:
    above: Optional[str] = None
    below: Optional[str] = None


@dataclass
class Reaction:
    reactants: list
    products: list
    arrow: str = "forward"
    conditions: Optional[ReactionConditions] = None

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


@dataclass
class MechanismStep:
    label: str
    reaction: str


@dataclass
class Mechanism:
    steps: list = field(default_factory=list)
    spectators: list = field(default_factory=list)


@dataclass
class Formula:
    nodes: list
