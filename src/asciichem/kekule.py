"""The text projection of an aromatic ring is its Kekule form - the
Python mirror of AsciiChem::Structure::Kekule (asciichem-ruby#188).

Aromatic text bonds are unparseable (the Lewis lone-pair prefix owns
':' in every position an aromatic bond could occupy), so ``c1ccccc1``
renders as ``C1=C-C=C-C=C1``. A Kekule assignment is a perfect
matching on the component's atoms (each atom exactly one double
bond), found by deterministic backtracking in Graph edge order
(include before skip) so all runtimes derive identical output.
Closure edges are never chosen: ring-closure digits are single by
grammar, so a matching that doubles one would silently lose that
bond's order in the text. Components with no perfect matching (odd
rings like furan) stay out of the projection and keep the aromatic
rendering (the spec'd text loss). The model itself is never touched -
the projection is a render-time view, exactly like perception."""
from __future__ import annotations

from .structure import Edge, build_graph


def bond_kinds(molecule) -> dict[tuple[int, int], str]:
    """The Kekule projection of ``molecule``'s aromatic chain bonds:
    ``((from, to) -> "double" | "single")`` keyed by ordered
    atom-index pair (the same key the positional text walk derives
    for the bond token it renders). Ring-closure edges carry no bond
    token (digits are single by grammar) and get no entry."""
    kinds: dict[tuple[int, int], str] = {}
    _atoms, edges = build_graph(molecule)
    aromatic = [edge for edge in edges if edge.kind == "aromatic"]
    if not aromatic:
        return kinds

    for component in _components(aromatic):
        indices = _matching(component)
        if indices is None:
            continue
        for index, edge in enumerate(component):
            first, last = min(edge.from_, edge.to), max(edge.from_, edge.to)
            if last - first != 1:
                continue
            kinds[(first, last)] = "double" if index in indices else "single"
    return kinds


def _components(aromatic: list[Edge]) -> list[list[Edge]]:
    """Connected components of the aromatic subgraph (edges sharing
    atoms). Edges leave in shift order - the Ruby port's order."""
    found: list[list[Edge]] = []
    remaining = list(aromatic)
    while remaining:
        seed = remaining.pop(0)
        component = [seed]
        atoms = [seed.from_, seed.to]
        grew = True
        while grew:
            grew = False
            kept = []
            for edge in remaining:
                if edge.from_ in atoms or edge.to in atoms:
                    component.append(edge)
                    atoms.append(edge.from_)
                    atoms.append(edge.to)
                    grew = True
                else:
                    kept.append(edge)
            remaining = kept
        found.append(component)
    return found


def _matching(component: list[Edge]) -> list[int] | None:
    """Indices of a perfect matching over the component's atoms (each
    atom incident to exactly one chosen edge), or None. Closure edges
    are unchoosable. Include before skip - the deterministic order all
    runtimes mirror."""
    atom_set = sorted({index for edge in component for index in (edge.from_, edge.to)})
    counts: dict[int, int] = {}
    chosen: list[int] = []
    found: list[int] | None = None

    def search(index: int) -> bool:
        nonlocal found
        if index == len(component):
            if all(counts.get(atom, 0) == 1 for atom in atom_set):
                found = list(chosen)
                return True
            return False
        edge = component[index]
        chain = abs(edge.from_ - edge.to) == 1
        if chain and counts.get(edge.from_, 0) != 1 and counts.get(edge.to, 0) != 1:
            counts[edge.from_] = counts.get(edge.from_, 0) + 1
            counts[edge.to] = counts.get(edge.to, 0) + 1
            chosen.append(index)
            matched = search(index + 1)
            chosen.pop()
            counts[edge.from_] = counts.get(edge.from_, 0) - 1
            counts[edge.to] = counts.get(edge.to, 0) - 1
            if matched:
                return True
        return search(index + 1)

    search(0)
    return found
