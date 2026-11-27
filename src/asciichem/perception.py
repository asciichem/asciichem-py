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


def hydrogen_valence_sum(atom_index: int, edges: list[Edge],
                         ring: list[int] | None = None,
                         atoms: list[Atom] | None = None) -> float:
    """Hydrogen-counting sum: aromatic bonds order as 1.5 (Hueckel),
    so both spellings of benzene sum to 3.0 at every carbon.

    When a detected aromatic five-membered ring is supplied, its
    lone-pair donation donor counts its ring bonds as the kekulized
    singles the donation model implies (pyrrole N: two singles + one
    H = valence 3 -> one implicit hydrogen; furan O: two singles =
    valence 2 -> none) -- the blanket 1.5 sum over-values a donor
    (3.0 against valence 2 or 3) and would decline exactly the
    heteroatoms the donation rule exists to answer."""
    donor = (ring is not None and atoms is not None
             and _donation_donor(atoms[atom_index], atoms, atom_index, edges, ring))
    total = 0.0
    for edge in edges:
        if edge.from_ != atom_index and edge.to != atom_index:
            continue
        if edge.kind == "aromatic":
            total += 1.0 if donor else AROMATIC_H_ORDER
        else:
            total += BOND_ORDER.get(edge.kind, 1)
    return total


def _donation_donor(atom: Atom, atoms: list[Atom], atom_index: int,
                    edges: list[Edge], ring: list[int]) -> bool:
    """The donation donor of an aromatic five-membered ring: a
    lone-pair-donating element with no in-system double bond of its
    own (its 2pi come from the lone pair), uncharged, with no stated
    hydrogens (a stated count is the source's word -- never
    re-derived), and **unique among nitrogens** -- the donation is
    one lone pair per ring, so a plain `n` beside another donor
    element (imidazole's `n` beside an `[nH]`, oxazole's `n` beside
    the `o`) is pyridine-type: it keeps the blanket Hueckel sum and
    answers zero. O and S are the preferred donors and do not defer
    (oxazole's `o` stays the donor beside its `n`)."""
    if len(ring) != 5:
        return False
    if atom_index not in ring:
        return False
    if atom.charge is not None or atom.hydrogens is not None:
        return False
    if atom.element not in LONE_PAIR_DONORS:
        return False
    if _in_system_double(atom_index, edges, set(ring)):
        return False
    if atom.element == "N":
        for other_index in ring:
            if other_index == atom_index:
                continue
            if atoms[other_index].element in LONE_PAIR_DONORS:
                return False
    return True


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


# pi-electron-donating heteroatoms: a ring atom with no double bond of
# its own contributes its lone pair (pyrrole N, furan O, thiophene S) -
# the 2 electrons that make those five-membered rings 4n+2.
LONE_PAIR_DONORS = ("N", "O", "S", "P", "Se", "As")


def aromatic_rings(atoms: list[Atom], edges: list[Edge]) -> list[list[int]]:
    """Aromatic rings by Hueckel counting: 5- and 6-membered cycles
    whose pi contributions total 6 or 10 (4n+2 with n >= 1; n = 0 is a
    charged three-membered ring, which perception declines). Per atom:
    an in-system double bond contributes 1; a lone-pair donor with no
    double contributes 2; a carbon with only single bonds or an
    exocyclic double contributes 0. Charged atoms make the count
    undecidable and the ring is declined; aromatic-flagged rings are
    taken as stated.

    Sees fused systems (naphthalene's shared bond is single, but each
    atom's double bond lies in the ring system) and keeps quinones out
    (their carbons' only doubles point at exocyclic oxygens)."""
    cycles = _ring_cycles(edges, len(atoms))
    cyclic_atoms = {atom for cycle in cycles for atom in cycle}
    return [c for c in cycles if _aromatic_cycle(c, atoms, edges, cyclic_atoms)]


def _aromatic_cycle(cycle, atoms, edges, cyclic_atoms) -> bool:
    if any(atoms[i].aromatic is True for i in cycle):
        return True
    if len(cycle) not in (5, 6):
        return False
    electrons = [_ring_pi_electrons(atoms[i], i, edges, cyclic_atoms)
                 for i in cycle]
    if any(e is None for e in electrons):
        return False
    return sum(electrons) in (6, 10)


def _ring_pi_electrons(atom, atom_index, edges, cyclic_atoms):
    if atom.charge is not None:
        return None
    if _in_system_double(atom_index, edges, cyclic_atoms):
        return 1
    if atom.element in LONE_PAIR_DONORS:
        return 2
    return 0


def _in_system_double(atom_index, edges, cyclic_atoms) -> bool:
    for edge in edges:
        if edge.from_ != atom_index and edge.to != atom_index:
            continue
        if edge.kind != "double":
            continue
        partner = edge.to if edge.from_ == atom_index else edge.from_
        if partner in cyclic_atoms:
            return True
    return False


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
