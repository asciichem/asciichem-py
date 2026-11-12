"""Periodic table as a single source of truth for element data - the
Python mirror of AsciiChem::PeriodicTable. Consumed by the linter
checks (element validation, isotope sanity, valence) and the offline
identifier validators (InChI formula layer, SMILES element tokens).

Adding a new field (e.g. covalent radius for layout, common oxidation
states for redox checks) means adding one column to `Element` and
populating it across `ELEMENTS` - no other code changes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    symbol: str
    atomic_number: int
    max_valence: int


# IUPAC 2021 standard atomic weights (isotope-averaged) for the most
# common elements. Absent symbols are "unknown" to atomic_mass().
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008, "He": 4.0026,
    "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867,
    "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971,
    "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Ag": 107.87, "Cd": 112.41,
    "Sn": 118.71, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "Au": 196.97, "Hg": 200.59,
    "Pb": 207.2, "Bi": 208.98, "U": 238.03,
}

# Subset covering the elements chemistry most commonly deals with.
# Adding more elements is one line per element; the linter picks them
# up automatically.
ELEMENTS: dict[str, Element] = {
    # Period 1
    "H": Element("H", 1, 1),
    "He": Element("He", 2, 0),
    # Period 2
    "Li": Element("Li", 3, 1),
    "Be": Element("Be", 4, 2),
    "B": Element("B", 5, 3),
    "C": Element("C", 6, 4),
    "N": Element("N", 7, 3),
    "O": Element("O", 8, 2),
    "F": Element("F", 9, 1),
    "Ne": Element("Ne", 10, 0),
    # Period 3
    "Na": Element("Na", 11, 1),
    "Mg": Element("Mg", 12, 2),
    "Al": Element("Al", 13, 3),
    "Si": Element("Si", 14, 4),
    "P": Element("P", 15, 5),
    "S": Element("S", 16, 6),
    "Cl": Element("Cl", 17, 1),
    "Ar": Element("Ar", 18, 0),
    # Period 4 (common)
    "K": Element("K", 19, 1),
    "Ca": Element("Ca", 20, 2),
    "Sc": Element("Sc", 21, 3),
    "Ti": Element("Ti", 22, 4),
    "V": Element("V", 23, 5),
    "Cr": Element("Cr", 24, 6),
    "Mn": Element("Mn", 25, 7),
    "Fe": Element("Fe", 26, 6),
    "Co": Element("Co", 27, 4),
    "Ni": Element("Ni", 28, 4),
    "Cu": Element("Cu", 29, 4),
    "Zn": Element("Zn", 30, 2),
    "Ga": Element("Ga", 31, 3),
    "Ge": Element("Ge", 32, 4),
    "As": Element("As", 33, 3),
    "Se": Element("Se", 34, 2),
    "Br": Element("Br", 35, 1),
    "Kr": Element("Kr", 36, 0),
    # Period 5 (common)
    "Rb": Element("Rb", 37, 1),
    "Sr": Element("Sr", 38, 2),
    "Y": Element("Y", 39, 3),
    "Zr": Element("Zr", 40, 4),
    "Nb": Element("Nb", 41, 5),
    "Mo": Element("Mo", 42, 6),
    "Tc": Element("Tc", 43, 6),
    "Ru": Element("Ru", 44, 6),
    "Rh": Element("Rh", 45, 6),
    "Pd": Element("Pd", 46, 4),
    "Ag": Element("Ag", 47, 4),
    "Cd": Element("Cd", 48, 2),
    "In": Element("In", 49, 3),
    "Sn": Element("Sn", 50, 4),
    "Sb": Element("Sb", 51, 3),
    "Te": Element("Te", 52, 2),
    "I": Element("I", 53, 1),
    "Xe": Element("Xe", 54, 0),
    # Period 6 (common)
    "Cs": Element("Cs", 55, 1),
    "Ba": Element("Ba", 56, 2),
    "W": Element("W", 74, 6),
    "Pt": Element("Pt", 78, 4),
    "Au": Element("Au", 79, 6),
    "Hg": Element("Hg", 80, 2),
    "Tl": Element("Tl", 81, 3),
    "Pb": Element("Pb", 82, 4),
    "Bi": Element("Bi", 83, 3),
    # Period 7 (common)
    "U": Element("U", 92, 6),
}


def element(symbol: str) -> Element | None:
    return ELEMENTS.get(symbol)


def known(symbol: str) -> bool:
    return symbol in ELEMENTS


def atomic_number(symbol: str) -> int | None:
    found = element(symbol)
    return found.atomic_number if found else None


def max_valence(symbol: str) -> int | None:
    found = element(symbol)
    return found.max_valence if found else None


def atomic_mass(symbol: str) -> float | None:
    return ATOMIC_MASSES.get(symbol)


def symbols() -> list[str]:
    return list(ELEMENTS)
