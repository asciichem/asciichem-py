"""Perception pass (TODO.v2/13 item 2) - the Python mirror of
AsciiChem::Perception. Chemistry the syntax leaves implicit, derived
once and consumed everywhere; semantics are conformance-tested through
the shared corpus (fixtures/perception.json), never re-derived. None
in the corpus means perception declines - this implementation must
decline identically."""
from __future__ import annotations

from .model import Atom
from .structure import Edge, build_graph

DEFAULT_VALENCES: dict[str, list[int]] = {
    "H": [1], "B": [3], "C": [4], "N": [3], "O": [2],
    "F": [1], "Al": [3], "Si": [4], "P": [3, 5],
    "S": [2, 4, 6], "Cl": [1], "Br": [1], "I": [1],
}

BOND_ORDER: dict[str, int] = {
    "single": 1, "double": 2, "triple": 3, "quadruple": 4,
    "aromatic": 1, "wedge": 1, "hash": 1, "dative": 1, "wavy": 1,
}

AROMATIC_H_ORDER = 1.5


def neighbours(atom_index: int, edges: list[Edge]) -> list[int]:
    acc: list[int] = []
    for edge in edges:
        if edge.from_ == atom_index:
            acc.append(edge.to)
        if edge.to == atom_index:
            acc.append(edge.from_)
    return acc


def bond_order_sum(atom_index: int, edges: list[Edge]) -> int:
    """Valence-feasibility sum (aromatic orders as 1)."""
    total = 0
    for edge in edges:
        if edge.from_ != atom_index and edge.to != atom_index:
            continue
        total += BOND_ORDER.get(edge.kind, 1)
    return total


def hydrogen_valence_sum(atom_index: int, edges: list[Edge]) -> float:
    """Hydrogen-counting sum: aromatic bonds order as 1.5 (Hueckel),
    so both spellings of benzene sum to 3.0 at every carbon."""
    total = 0.0
    for edge in edges:
        if edge.from_ != atom_index and edge.to != atom_index:
            continue
        total += (AROMATIC_H_ORDER if edge.kind == "aromatic"
                  else BOND_ORDER.get(edge.kind, 1))
    return total


def valence_deficits(element: str, valence_sum: float) -> list[float]:
    """Whole-number, non-negative gaps between each default valence and
    the hydrogen-valence sum; empty when the element is unknown or no
    valence fits integrally."""
    valences = DEFAULT_VALENCES.get(element)
    if valences is None:
        return []
    return [d for v in valences
            if (d := v - valence_sum) >= 0 and d == int(d)]


def implicit_hydrogens(atom: Atom, valence_sum: float) -> int | None:
    """Implicit hydrogens: the smallest fitting default valence minus
    the hydrogen-valence sum.

    - Atoms that STATE hydrogens ([NH4+], [CH4]) return 0 - never add
      to what the source said.
    - Charged atoms return None: charge algebra is deferred.
    - Unknown elements, non-integral or over-valent sums return None -
      never invent.
    """
    if atom.hydrogens is not None:
        return 0
    if atom.charge is not None:
        return None
    deficits = valence_deficits(atom.element, valence_sum)
    if not deficits:
        return None
    return int(min(deficits))


def _is_sp2(atom_index: int, atoms: list[Atom], edges: list[Edge]) -> bool:
    if atoms[atom_index].aromatic is True:
        return True
    return any(
        (edge.from_ == atom_index or edge.to == atom_index)
        and edge.kind == "double"
        for edge in edges
    )


def aromatic_rings(atoms: list[Atom], edges: list[Edge]) -> list[list[int]]:
    """Six-membered cycles where every atom is sp2 (participates in a
    double bond) or aromatic-flagged - the fused-system rule (per-ring
    alternation can never see both naphthalene rings). Full Hueckel
    pi-electron counting is a later slice."""
    cycles = [c for c in _ring_cycles(edges, len(atoms)) if len(c) == 6]
    return [c for c in cycles if all(_is_sp2(i, atoms, edges) for i in c)]


def bond_kind(a: int, b: int, edges: list[Edge]) -> str | None:
    for edge in edges:
        if (edge.from_, edge.to) in ((a, b), (b, a)):
            return edge.kind
    return None


def _ring_cycles(edges: list[Edge], atom_count: int) -> list[list[int]]:
    found: list[list[int]] = []
    for start in range(atom_count):
        _dfs_cycles(edges, start, start, [start], 6, found)
    seen: set[tuple[int, int]] = set()
    out: list[list[int]] = []
    for cycle in found:
        key = (min(cycle), max(cycle))
        if key not in seen:
            seen.add(key)
            out.append(cycle)
    return out


def _dfs_cycles(edges: list[Edge], start: int, current: int, path: list[int],
                max_size: int, found: list[list[int]]) -> None:
    for nb in neighbours(current, edges):
        if len(path) >= 2 and nb == path[-2]:
            continue
        if nb == start:
            if len(path) >= 3:
                found.append(list(path))
        elif nb not in path and len(path) < max_size:
            _dfs_cycles(edges, start, nb, path + [nb], max_size, found)
