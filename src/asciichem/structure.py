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
    __slots__ = ("from_", "to", "kind", "direction", "shape")

    def __init__(self, from_: int, to: int, kind: str,
                 direction: str | None = None,
                 shape: str | None = None):
        self.from_ = from_
        self.to = to
        self.kind = kind
        self.direction = direction
        # The SMILES reader's structural mark (chain/branch/ring) — the
        # linearizer partitions by it instead of index arithmetic
        # (branch atoms parsed before their continuation are
        # index-consecutive and would masquerade as chain tokens).
        # None for edges derived from an already-linearized model.
        self.shape = shape


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
                    edges.append(Edge(last, index, pending.kind, pending.direction))
                last = index
                pending = None
            elif isinstance(node, Bond):
                pending = node
            elif isinstance(node, (Group, Molecule)):
                # A group whose FIRST node is a bond is a BRANCH group:
                # the leading bond attaches the branch to the atom
                # before the group (the branch root); the continuation
                # resumes from the root — `C(-O)=O` reads C-O and C=O,
                # never O=O.
                branch_root = None
                if isinstance(node, Group) and last is not None:
                    first = node.nodes[0] if node.nodes else None
                    while isinstance(first, Molecule):
                        first = first.nodes[0] if first.nodes else None
                    branch_root = last if isinstance(first, Bond) else None
                walk(node.nodes)
                if branch_root is not None:
                    last = branch_root

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
    # Partition by the parser's own shape marks — tokens and digits
    # first (so the chain map is complete), then the branch trees. A
    # branch edge inside another branch's tree emits as a NESTED
    # group; only top-level branches render at the flat stream.
    # Producers without shape marks (molfile, CML) ride the flat rule:
    # consecutive indices are chain tokens, every other non-ring edge
    # a digit pair — the shape those producers were recorded against.
    close_at = [-1] * MAX_DIGIT
    digits_for = [""] * len(atoms)
    token_before: dict[int, Edge] = {}
    children: list[list[Edge]] = [[] for _ in atoms]
    in_branch: set[int] = set()
    branch_edges: list[Edge] = []

    for edge in edges:
        if edge.shape == "branch":
            branch_edges.append(edge)
        elif edge.shape == "chain" or (
            edge.shape is None and abs(edge.to - edge.from_) == 1
        ):
            token_before[edge.to] = edge
            children[edge.from_].append(edge)
        else:
            first, last = min(edge.from_, edge.to), max(edge.from_, edge.to)
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

    for edge in branch_edges:
        _collect_branch(edge.to, children, in_branch)
        children[edge.from_].append(edge)
    branch_at: dict[int, list[Edge]] = {}
    for edge in branch_edges:
        if edge.from_ not in in_branch:
            branch_at.setdefault(edge.from_, []).append(edge)

    def branch_group(edge: Edge) -> Group:
        # The branch's group: the leading bond, then the tree's atoms
        # in walk order with their chain tokens.
        nodes: list[Node] = [Bond(edge.kind, direction=edge.direction)]
        _emit_tree(edge.to, nodes, atoms, digits_for, children, branch_edges)
        return Group(bracket="paren", nodes=nodes)

    nodes: list[Node] = []
    for index, atom in enumerate(atoms):
        if index in in_branch:
            continue
        edge = token_before.get(index)
        if index > 0 and edge:
            nodes.append(Bond(edge.kind, direction=edge.direction))
        atom.ring_closures = digits_for[index] or None
        nodes.append(atom)
        for branch in branch_at.get(index, []):
            nodes.append(branch_group(branch))
    return nodes


def _collect_branch(start: int, children, in_branch: set[int]) -> None:
    in_branch.add(start)
    for edge in children[start]:
        _collect_branch(edge.to, children, in_branch)


def _emit_tree(atom_index: int, nodes: list[Node], atoms, digits_for,
               children, branch_edges) -> None:
    atom = atoms[atom_index]
    if digits_for[atom_index]:
        atom.ring_closures = digits_for[atom_index]
    nodes.append(atom)
    # Branch groups sit directly after their parent atom (the walkers
    # bind a group's leading bond to the atom before it — emitting a
    # continuation first would steal the binding).
    for edge in children[atom_index]:
        if edge in branch_edges:
            nodes.append(_branch_group_nested(edge, atoms, digits_for,
                                              children, branch_edges))
    for edge in children[atom_index]:
        if edge in branch_edges:
            continue
        nodes.append(Bond(edge.kind, direction=edge.direction))
        _emit_tree(edge.to, nodes, atoms, digits_for, children,
                   branch_edges)


def _branch_group_nested(edge: Edge, atoms, digits_for, children,
                         branch_edges) -> Group:
    nodes: list[Node] = [Bond(edge.kind, direction=edge.direction)]
    _emit_tree(edge.to, nodes, atoms, digits_for, children, branch_edges)
    return Group(bracket="paren", nodes=nodes)
