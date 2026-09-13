"""Shared structure plumbing for the interchange formats - the Python
mirror of AsciiChem::Structure: a graph walk (pending-bond semantics
plus ring-closure edges, identity by object id because dataclass eq
makes benzene's carbons all equal) and the adjacency linearizer
(bond tokens for consecutive edges, interval-reused ring digits for
the rest - the model's own ring mechanism, so every existing
renderer works unchanged)."""
from __future__ import annotations

from .model import Atom, Bond, Group, Molecule, Node, ParseError

MAX_DIGIT = 9


class Edge:
    __slots__ = ("from_", "to", "kind")

    def __init__(self, from_: int, to: int, kind: str):
        self.from_ = from_
        self.to = to
        self.kind = kind


def build_graph(molecule: Molecule):
    atoms: list[Atom] = []
    edges: list[Edge] = []
    index_by_id: dict[int, int] = {}
    pending: Bond | None = None
    last: int | None = None

    def walk(nodes):
        nonlocal pending, last
        for node in nodes:
            if isinstance(node, Atom):
                index = len(atoms)
                index_by_id[id(node)] = index
                atoms.append(node)
                if pending is not None and last is not None:
                    edges.append(Edge(last, index, pending.kind))
                last = index
                pending = None
            elif isinstance(node, Bond):
                pending = node
            elif isinstance(node, (Group, Molecule)):
                walk(node.nodes)

    walk(molecule.nodes)

    for a, b in ring_bond_pairs(atoms):
        kind = ("aromatic" if atoms[a].aromatic and atoms[b].aromatic
                else "single")
        edges.append(Edge(a, b, kind))

    return atoms, edges


def ring_bond_pairs(atoms: list[Atom]):
    open_rings: dict[str, int] = {}
    pairs = []
    for index, atom in enumerate(atoms):
        if not atom.ring_closures:
            continue
        for digit in atom.ring_closures:
            opener = open_rings.pop(digit, None)
            if opener is not None:
                pairs.append((opener, index))
            else:
                open_rings[digit] = index
    return pairs


def linearize(atoms: list[Atom], edges: list[Edge]) -> list[Node]:
    close_at = [-1] * MAX_DIGIT
    digits_for = [""] * len(atoms)
    token_before: dict[int, str] = {}

    for edge in edges:
        first, last = min(edge.from_, edge.to), max(edge.from_, edge.to)
        if last - first == 1:
            token_before[last] = edge.kind
        else:
            digit_index = next((d for d in range(MAX_DIGIT)
                                if close_at[d] <= first), None)
            if digit_index is None:
                raise ParseError(
                    f"more than {MAX_DIGIT} overlapping non-adjacent bonds - "
                    "beyond the model's ring-closure digit capacity")
            close_at[digit_index] = last
            digit = str(digit_index + 1)
            digits_for[first] += digit
            digits_for[last] += digit

    nodes: list[Node] = []
    for index, atom in enumerate(atoms):
        kind = token_before.get(index)
        if index > 0 and kind:
            nodes.append(Bond(kind))
        atom.ring_closures = digits_for[index] or None
        nodes.append(atom)
    return nodes
