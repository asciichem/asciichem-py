"""Deterministic 2D layout for bonded molecules.

Ports the TypeScript implementation's layout.ts (itself a port of the
reference decomposition): a pure walker produces a neutral atom+bond
graph (ring-closure pairing mirrors RingBonds), then a
chemistry-shaped deterministic placer — rings as regular polygons
(fused rings anchored on shared atoms), chains as 120-degree zigzags
grown breadth-first in source order.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .model import Atom, Bond, Group, Molecule, Node

BOND_LENGTH = 56.0


@dataclass
class RingBond:
    digit: str
    from_atom: int
    to_atom: int


@dataclass
class LayoutBond:
    from_idx: int
    to_idx: int
    bond: Bond


@dataclass
class LayoutGraph:
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[LayoutBond] = field(default_factory=list)
    ring_bonds: list[RingBond] = field(default_factory=list)


@dataclass
class Point:
    x: float
    y: float


def ring_bonds(atoms: list[Atom]) -> list[RingBond]:
    """Ring-closure pairing, ported from RingBonds: walk atoms in
    source order; a repeated digit closes the ring between the opener
    and the closer. Multiple digits per atom open in parallel."""
    open_rings: dict[str, int] = {}
    rings: list[RingBond] = []
    for index, atom in enumerate(atoms):
        if not atom.ring_closures:
            continue
        for digit in atom.ring_closures:
            if digit in open_rings:
                rings.append(RingBond(digit, open_rings.pop(digit), index))
            else:
                open_rings[digit] = index
    return rings


def unclosed_ring_atoms(atoms: list[Atom]) -> list[int]:
    open_rings: dict[str, int] = {}
    for index, atom in enumerate(atoms):
        if not atom.ring_closures:
            continue
        for digit in atom.ring_closures:
            if digit in open_rings:
                del open_rings[digit]
            else:
                open_rings[digit] = index
    return list(open_rings.values())


def walk_molecule(molecule: Molecule) -> LayoutGraph:
    """Walks a molecule into a neutral atom+bond graph. Bond tokens
    connect the last atom of the previous unit to the first atom of
    the next; nested groups contribute their walk order."""
    graph = LayoutGraph()

    def walk(node: Node) -> tuple[int, int] | None:
        if isinstance(node, Atom):
            graph.atoms.append(node)
            last = len(graph.atoms) - 1
            return last, last
        if isinstance(node, Group):
            first = None
            last = None
            for child in node.nodes:
                span = walk(child)
                if span is not None:
                    if first is None:
                        first = span[0]
                    last = span[1]
            if first is None or last is None:
                return None
            return first, last
        return None

    previous = None
    pending_bond = None
    for unit in molecule.nodes:
        if isinstance(unit, Bond):
            pending_bond = unit
            continue
        span = walk(unit)
        if span is not None:
            if previous is not None and pending_bond is not None:
                graph.bonds.append(LayoutBond(previous, span[0], pending_bond))
            previous = span[1]
            pending_bond = None

    graph.ring_bonds = ring_bonds(graph.atoms)
    for ring in graph.ring_bonds:
        graph.bonds.append(LayoutBond(ring.from_atom, ring.to_atom, Bond("single")))
    return graph


def _neighbors(graph: LayoutGraph, node: int) -> list[int]:
    out: list[int] = []
    for bond in graph.bonds:
        if bond.from_idx == node:
            out.append(bond.to_idx)
        elif bond.to_idx == node:
            out.append(bond.from_idx)
    return out


def _ring_cycle(graph: LayoutGraph, closure: RingBond) -> list[int] | None:
    """Shortest path between the closure endpoints, excluding the
    closure edge itself — the path plus the closure edge is the ring
    cycle."""
    queue: deque[list[int]] = deque([[closure.from_atom]])
    visited = {closure.from_atom}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == closure.to_atom:
            return path
        for neighbor in _neighbors(graph, node):
            is_closure_edge = (node == closure.from_atom and neighbor == closure.to_atom) or (
                node == closure.to_atom and neighbor == closure.from_atom
            )
            if is_closure_edge or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append([*path, neighbor])
    return None


def place_graph(graph: LayoutGraph) -> list[Point]:
    positions: list[Point | None] = [None] * len(graph.atoms)

    cycles = []
    for closure in graph.ring_bonds:
        path = _ring_cycle(graph, closure)
        if path is not None and len(path) >= 3:
            cycles.append(path)

    # Place rings first: each ring is a regular polygon. A ring with
    # an already-placed member anchors on it (fused look); a ring in
    # a new component starts at the next free center to the right.
    free_center_x = 0.0
    for cycle in cycles:
        anchor_index = next((i for i, m in enumerate(cycle) if positions[m] is not None), -1)
        radius = _polygon_radius(len(cycle))
        if anchor_index == -1:
            center = Point(free_center_x, 0.0)
            _place_polygon(cycle, center, radius, 0.0, positions)
            free_center_x += radius * 2 + BOND_LENGTH
        else:
            anchor = cycle[anchor_index]
            anchor_pos = positions[anchor]
            assert anchor_pos is not None
            # Direction pointing away from any already-placed neighbor
            # of the anchor; the polygon center sits opposite it.
            outward = -math.inf
            for neighbor in _neighbors(graph, anchor):
                npos = positions[neighbor]
                if npos is not None:
                    outward = math.atan2(anchor_pos.y - npos.y, anchor_pos.x - npos.x)
                    break
            vertex_angle = _angle_of_member(len(cycle), anchor_index)
            center_angle = vertex_angle + math.pi if outward == -math.inf else outward
            center = Point(
                anchor_pos.x + radius * math.cos(center_angle),
                anchor_pos.y + radius * math.sin(center_angle),
            )
            # Rotate the polygon so the anchor lands on its position.
            _place_polygon(cycle, center, radius, center_angle - vertex_angle, positions)

    # Grow chains breadth-first from every placed atom (zigzag), then
    # from any still-unplaced atom (new component at origin baseline).
    for i in range(len(graph.atoms)):
        if positions[i] is None:
            continue
        _grow(i, positions, graph, {i})
    for i in range(len(graph.atoms)):
        if positions[i] is not None:
            continue
        positions[i] = Point(free_center_x, 0.0)
        free_center_x += BOND_LENGTH
        _grow(i, positions, graph, {i})

    return [p for p in positions if p is not None]


def _grow(
    start: int,
    positions: list[Point | None],
    graph: LayoutGraph,
    visited: set[int],
) -> None:
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        current_pos = positions[current]
        if current_pos is None:
            continue
        # Direction the chain is travelling: away from the origin of
        # the component (0,0) — keeps zigzags deterministic and
        # horizontal-ish.
        heading = math.atan2(-current_pos.y, current_pos.x)
        children = [
            n
            for n in _neighbors(graph, current)
            if n not in visited and positions[n] is None
        ]
        for i, child in enumerate(children):
            if child in visited:
                continue
            spread = -math.pi / 3 if i == 0 else math.pi / 3 if i == 1 else 0.0
            angle = heading + spread
            positions[child] = Point(
                current_pos.x + BOND_LENGTH * math.cos(angle),
                current_pos.y + BOND_LENGTH * math.sin(angle),
            )
            visited.add(child)
            queue.append(child)


def _polygon_radius(sides: int) -> float:
    """Circumradius for a regular polygon with the given side length."""
    return BOND_LENGTH / (2 * math.sin(math.pi / sides))


def _angle_of_member(sides: int, member_index: int) -> float:
    return math.pi * 2 * member_index / sides - math.pi / 2


def _place_polygon(
    members: list[int],
    center: Point,
    radius: float,
    shift: float,
    positions: list[Point | None],
) -> None:
    for i, atom in enumerate(members):
        angle = _angle_of_member(len(members), i) + shift
        positions[atom] = Point(
            center.x + radius * math.cos(angle),
            center.y + radius * math.sin(angle),
        )


def layout_molecule(molecule: Molecule) -> tuple[LayoutGraph, list[Point]]:
    graph = walk_molecule(molecule)
    return graph, place_graph(graph)
