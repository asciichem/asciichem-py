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

    # Rings first, in source order. Each cycle is placed free
    # (nothing placed yet), fused (two or more atoms already placed —
    # fit the polygon to them), or anchored (one placed atom — orient
    # away from the placed region).
    free_center_x = 0.0
    for cycle in cycles:
        anchored = [m for m in cycle if positions[m] is not None]
        if not anchored:
            free_center_x = _place_free_cycle(cycle, positions, free_center_x)
        elif len(anchored) >= 2:
            _place_fused_cycle(cycle, positions)
        else:
            _place_anchored_cycle(cycle, positions)

    # Grow chains breadth-first from every placed atom (zigzag), then
    # from any still-unplaced atom (new component at origin baseline).
    for i in range(len(graph.atoms)):
        if positions[i] is None:
            continue
        _grow(i, positions, graph)
    for i in range(len(graph.atoms)):
        if positions[i] is not None:
            continue
        positions[i] = Point(free_center_x, 0.0)
        free_center_x += BOND_LENGTH
        _grow(i, positions, graph)

    return [p for p in positions if p is not None]


def _place_free_cycle(
    cycle: list[int],
    positions: list[Point | None],
    free_center_x: float,
) -> float:
    radius = _polygon_radius(len(cycle))
    _place_polygon(cycle, Point(free_center_x, 0.0), radius, 0.0, positions)
    return free_center_x + radius * 2 + BOND_LENGTH


def _place_fused_cycle(cycle: list[int], positions: list[Point | None]) -> None:
    """Fused on shared atoms: derive the rigid transform (rotation +
    translation) that maps the ideal polygon's two placed members onto
    their laid-out positions, then fill only the unplaced members.
    Placed atoms never move — the first ring keeps its geometry."""
    a = next(m for m in cycle if positions[m] is not None)
    b = next(m for m in cycle if positions[m] is not None and m != a)
    ia = cycle.index(a)
    ib = cycle.index(b)
    ideal_a = _ideal_point(len(cycle), ia, 0.0)
    ideal_b = _ideal_point(len(cycle), ib, 0.0)
    pos_a, pos_b = positions[a], positions[b]
    assert pos_a is not None and pos_b is not None
    theta = math.atan2(pos_b.y - pos_a.y, pos_b.x - pos_a.x) - math.atan2(
        ideal_b.y - ideal_a.y, ideal_b.x - ideal_a.x
    )
    cos, sin = math.cos(theta), math.sin(theta)
    for i, member in enumerate(cycle):
        if positions[member] is not None:
            continue
        ideal = _ideal_point(len(cycle), i, 0.0)
        dx = ideal.x - ideal_a.x
        dy = ideal.y - ideal_a.y
        positions[member] = Point(
            pos_a.x + dx * cos - dy * sin,
            pos_a.y + dx * sin + dy * cos,
        )


def _place_anchored_cycle(cycle: list[int], positions: list[Point | None]) -> None:
    """One placed atom: centre the polygon outward, away from the
    centroid of everything already placed (a spiro centre must not
    fold the new ring back onto the old one). The anchor sits opposite
    the centre direction, so the shift must put the anchor's own
    vertex there (+π)."""
    anchor = next(m for m in cycle if positions[m] is not None)
    anchor_index = cycle.index(anchor)
    anchor_pos = positions[anchor]
    assert anchor_pos is not None
    vertex_angle = _angle_of_member(len(cycle), anchor_index)
    placed = [p for p in positions if p is not None]
    if len(placed) >= 2:
        centroid = Point(
            sum(p.x for p in placed) / len(placed),
            sum(p.y for p in placed) / len(placed),
        )
        center_angle = math.atan2(anchor_pos.y - centroid.y, anchor_pos.x - centroid.x)
    else:
        center_angle = vertex_angle
    shift = center_angle + math.pi - vertex_angle
    radius = _polygon_radius(len(cycle))
    center = Point(
        anchor_pos.x + radius * math.cos(center_angle),
        anchor_pos.y + radius * math.sin(center_angle),
    )
    for i, member in enumerate(cycle):
        if positions[member] is not None:
            continue
        ideal = _ideal_point(len(cycle), i, shift)
        positions[member] = Point(center.x + ideal.x, center.y + ideal.y)


def _grow(
    start: int,
    positions: list[Point | None],
    graph: LayoutGraph,
) -> None:
    # Zigzag growth: each generation turns the opposite way from its
    # parent, so chains alternate instead of curling. Tracked per atom
    # as [parent, incoming_angle, turn].
    start_pos = positions[start]
    assert start_pos is not None
    visited: dict[int, tuple[int | None, float, int]] = {
        start: (None, math.atan2(start_pos.y, start_pos.x), -1)
    }
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        current_pos = positions[current]
        if current_pos is None:
            continue
        incoming_angle = visited[current][1]
        children = [
            n
            for n in _neighbors(graph, current)
            if n not in visited and positions[n] is None
        ]
        for child in children:
            if child in visited:
                continue
            turn = -visited[current][2]
            angle = incoming_angle + turn * (math.pi / 3)
            positions[child] = _clear_position(current_pos, angle, positions)
            visited[child] = (current, angle, turn)
            queue.append(child)


def _clear_position(
    from_pos: Point,
    angle: float,
    positions: list[Point | None],
) -> Point:
    """Try the preferred spot, then spread out; if the preferred spot
    is occupied (another substituent, a ring atom), fall back to the
    candidate farthest from any placed atom. Deterministic: first best
    wins ties."""
    candidates = [
        Point(
            from_pos.x + BOND_LENGTH * math.cos(angle + spread),
            from_pos.y + BOND_LENGTH * math.sin(angle + spread),
        )
        for spread in (0.0, -math.pi / 3, math.pi / 3, -2 * math.pi / 3, 2 * math.pi / 3, math.pi)
    ]
    occupied = [p for p in positions if p is not None]
    for candidate in candidates:
        if all(_distance(candidate, other) >= BOND_LENGTH / 2 for other in occupied):
            return candidate
    return max(candidates, key=lambda candidate: sum(_distance(candidate, o) for o in occupied))


def _distance(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


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


def _ideal_point(sides: int, member_index: int, shift: float) -> Point:
    """The ideal polygon vertex position relative to the centre."""
    radius = _polygon_radius(sides)
    angle = _angle_of_member(sides, member_index) + shift
    return Point(radius * math.cos(angle), radius * math.sin(angle))


def layout_molecule(molecule: Molecule) -> tuple[LayoutGraph, list[Point]]:
    graph = walk_molecule(molecule)
    # Authored coordinates (the CML x2/y2 round-trip, molfile-carried)
    # take precedence over every computed layout — the depiction-hint
    # override: the renderer speaks the molecule's own geometry when it
    # carries one.
    if graph.atoms and all(a.x2 is not None and a.y2 is not None for a in graph.atoms):
        return graph, [Point(a.x2, a.y2) for a in graph.atoms]
    return graph, place_graph(graph)
