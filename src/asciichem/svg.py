"""SVG renderers — the Python port of the TypeScript implementation's
formatter/svg.ts (linear, typographic) and formatter/structural-svg.ts
(2D skeletal diagram). Bonded molecules render structurally; everything
else renders linearly, reactions and beyond-formulas nodes included.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .layout import Point, layout_molecule, unclosed_ring_atoms
from .model import (
    Atom,
    Bond,
    EmbeddedMath,
    ElectronConfiguration,
    Formula,
    Group,
    Mechanism,
    Molecule,
    Node,
    Reaction,
    ReactionCascade,
    STEREO_LETTER,
    Text,
)
from .text import render as render_text

FONT_SIZE = 15.0
LABEL_RADIUS = 11.0  # bond lines stop this far from atom centers
LINE_FONT_SIZE = 16.0
CHAR_WIDTH = LINE_FONT_SIZE * 0.62  # monospace-ish estimate
LINE_HEIGHT = LINE_FONT_SIZE * 1.6


def _fmt(n: float) -> str:
    # The JS reference interpolates numbers; whole values print
    # without a decimal and -0 prints as 0.
    rounded = math.floor(n * 10 + 0.5) / 10  # Math.round semantics
    if rounded == 0:
        return "0"
    if rounded == int(rounded):
        return str(int(rounded))
    return repr(rounded)


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# -- public surface ------------------------------------------------------


def render_structural_svg(formula) -> str:
    """2D skeletal diagram; linear typographic fallback inside."""
    children = [
        _render_molecule(n) if isinstance(n, Molecule) else render_svg(n)
        for n in formula.nodes
    ]
    boxes = [_view_box_of(c) for c in children]
    if children:
        min_x = min(b[0] for b in boxes)
        min_y = min(b[1] for b in boxes)
        max_x = max(b[0] + b[2] for b in boxes)
        max_y = max(b[1] + b[3] for b in boxes)
    else:
        min_x, min_y, max_x, max_y = 0.0, 0.0, 40.0, 40.0
    inner = "\n".join(
        f'    <svg x="{_fmt(x - min_x)}" y="{_fmt(y - min_y)}" width="{_fmt(w)}" height="{_fmt(h)}" overflow="visible">{_strip_outer(svg)}</svg>'
        for svg, (x, y, w, h) in zip(children, boxes)
    )
    width = max(max_x - min_x, 40.0)
    height = max(max_y - min_y, 40.0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(width)} {_fmt(height)}" '
        f'width="{_fmt(width)}" height="{_fmt(height)}" role="img">\n{inner}\n</svg>'
    )


def render_svg(node) -> str:
    """Linear (typographic) SVG of any node."""
    return _linear_document([_segments_for(node)])


# -- structural ----------------------------------------------------------


def _render_molecule(molecule: Molecule) -> str:
    has_bond = any(isinstance(n, Bond) for n in molecule.nodes)
    if not has_bond:
        # No authored connectivity: a formula is not a structure (the
        # isomerism point) — render typographically.
        return render_svg(molecule)
    graph, positions = layout_molecule(molecule)
    elements: list[str] = []
    drawn: set[tuple[int, int]] = set()

    for bond in graph.bonds:
        key = (bond.from_idx, bond.to_idx) if bond.from_idx < bond.to_idx else (bond.to_idx, bond.from_idx)
        if key in drawn:
            continue
        drawn.add(key)
        elements.append(_bond_element(positions[bond.from_idx], positions[bond.to_idx], bond.bond))

    for atom, pos in zip(graph.atoms, positions):
        elements.append(_atom_element(atom, pos))

    for i in unclosed_ring_atoms(graph.atoms):
        pos = positions[i]
        elements.append(
            f'<text x="{_fmt(pos.x + 14)}" y="{_fmt(pos.y - 10)}" font-size="10" '
            'fill="#b45309" font-family="monospace">?</text>'
        )

    xs = [p.x for p in positions] + [_free_label_extent(positions)]
    ys = [p.y for p in positions]
    min_x, min_y = min(xs) - 24, min(ys) - 24
    max_x, max_y = max(xs) + 24, max(ys) + 24
    return _wrap(elements, min_x, min_y, max_x, max_y)


def _bond_element(from_p: Point, to_p: Point, bond: Bond) -> str:
    dx = to_p.x - from_p.x
    dy = to_p.y - from_p.y
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    a = Point(from_p.x + ux * LABEL_RADIUS, from_p.y + uy * LABEL_RADIUS)
    b = Point(to_p.x - ux * LABEL_RADIUS, to_p.y - uy * LABEL_RADIUS)
    nx, ny = -uy, ux
    lines: list[str] = []

    if bond.kind == "single":
        lines.append(_line(a, b))
    elif bond.kind == "aromatic":
        lines.append(_line(a, b).replace('stroke-width="1.6"', 'stroke-width="1.6" stroke-dasharray="4 2.5"'))
    elif bond.kind == "double":
        lines.append(_offset_line(a, b, nx, ny, -2.6))
        lines.append(_offset_line(a, b, nx, ny, 2.6))
    elif bond.kind == "triple":
        lines.append(_line(a, b))
        lines.append(_offset_line(a, b, nx, ny, -3.4))
        lines.append(_offset_line(a, b, nx, ny, 3.4))
    elif bond.kind == "quadruple":
        for off in (-4.8, -1.6, 1.6, 4.8):
            lines.append(_offset_line(a, b, nx, ny, off))
    elif bond.kind == "wedge":
        lines.append(
            f'<polygon points="{_fmt(from_p.x)},{_fmt(from_p.y)} {_fmt(b.x + nx * 4)},{_fmt(b.y + ny * 4)} '
            f'{_fmt(b.x - nx * 4)},{_fmt(b.y - ny * 4)}" fill="currentColor"/>'
        )
    elif bond.kind == "hash":
        t = 0.25
        while t <= 0.95:
            w = 1.5 + 3.5 * t
            px = a.x + (b.x - a.x) * t
            py = a.y + (b.y - a.y) * t
            lines.append(
                f'<line x1="{_fmt(px + nx * w)}" y1="{_fmt(py + ny * w)}" '
                f'x2="{_fmt(px - nx * w)}" y2="{_fmt(py - ny * w)}" stroke="currentColor" stroke-width="1.2"/>'
            )
            t += 0.18
    elif bond.kind == "dative":
        lines.append(
            f'<line x1="{_fmt(a.x)}" y1="{_fmt(a.y)}" x2="{_fmt(b.x)}" y2="{_fmt(b.y)}" '
            'stroke="currentColor" stroke-width="1.6"/>'
        )
        lines.append(_arrow_head(b, ux, uy))
    elif bond.kind == "wavy":
        lines.append(_wavy_path(a, b))
    return "".join(lines)


def _arrow_head(b: Point, ux: float, uy: float) -> str:
    size = 6.0
    left = Point(b.x - ux * size - uy * size * 0.6, b.y - uy * size + ux * size * 0.6)
    right = Point(b.x - ux * size + uy * size * 0.6, b.y - uy * size - ux * size * 0.6)
    return (
        f'<polyline points="{_fmt(left.x)},{_fmt(left.y)} {_fmt(b.x)},{_fmt(b.y)} '
        f'{_fmt(right.x)},{_fmt(right.y)}" fill="none" stroke="currentColor" stroke-width="1.6"/>'
    )


def _wavy_path(a: Point, b: Point) -> str:
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    waves = max(2, round(length / 12))
    d = f"M {_fmt(a.x)} {_fmt(a.y)}"
    for i in range(1, waves + 1):
        t = i / waves
        sign = 1.0 if i % 2 == 0 else -1.0
        amp = sign * 3.5 * math.sin(math.pi * t)
        d += f" L {_fmt(a.x + dx * t + nx * amp)} {_fmt(a.y + dy * t + ny * amp)}"
    return f'<path d="{d}" fill="none" stroke="currentColor" stroke-width="1.6"/>'


def _line(a: Point, b: Point) -> str:
    return (
        f'<line x1="{_fmt(a.x)}" y1="{_fmt(a.y)}" x2="{_fmt(b.x)}" y2="{_fmt(b.y)}" '
        'stroke="currentColor" stroke-width="1.6"/>'
    )


def _offset_line(a: Point, b: Point, nx: float, ny: float, off: float) -> str:
    return _line(Point(a.x + nx * off, a.y + ny * off), Point(b.x + nx * off, b.y + ny * off))


def _atom_element(atom: Atom, pos: Point) -> str:
    label = _escape_xml(atom.element)
    if atom.isotope is not None:
        label = (
            f'<tspan font-size="10" dy="-4">{_escape_xml(atom.isotope)}</tspan>'
            f'<tspan dy="4">{_escape_xml(atom.element)}</tspan>'
        )
    if atom.charge is not None:
        label += f'<tspan font-size="10" dy="-5">{_escape_xml(atom.charge)}</tspan>'
    return (
        f'<text x="{_fmt(pos.x)}" y="{_fmt(pos.y + 5)}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{_fmt(FONT_SIZE)}" font-weight="500" fill="currentColor">{label}</text>'
    )


def _free_label_extent(positions: list[Point]) -> float:
    return max(p.x + 12 for p in positions)


def _view_box_of(svg: str) -> tuple[float, float, float, float]:
    marker = 'viewBox="'
    start = svg.find(marker)
    if start == -1:
        return (0.0, 0.0, 40.0, 40.0)
    rest = svg[start + len(marker):]
    end = rest.find('"')
    if end == -1:
        return (0.0, 0.0, 40.0, 40.0)
    parts = rest[:end].split()
    if len(parts) != 4:
        return (0.0, 0.0, 40.0, 40.0)
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        return (0.0, 0.0, 40.0, 40.0)
    return (x, y, w, h)


def _strip_outer(svg: str) -> str:
    open_end = svg.find(">")
    if open_end == -1:
        return svg
    close = svg.rfind("</svg>")
    if close == -1:
        return svg
    return svg[open_end + 1: close].strip()


def _wrap(elements: list[str], min_x: float, min_y: float, max_x: float, max_y: float) -> str:
    width = max(max_x - min_x, 40.0)
    height = max(max_y - min_y, 40.0)
    body = "\n".join(f"    {e}" for e in elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{_fmt(min_x)} {_fmt(min_y)} {_fmt(width)} {_fmt(height)}" '
        f'width="{_fmt(width)}" height="{_fmt(height)}" role="img">\n'
        '  <g fill="none" stroke="currentColor">\n'
        f"{body}\n  </g>\n</svg>"
    )


# -- linear ----------------------------------------------------------------


@dataclass
class _Segment:
    text: str
    dy: float
    size: float


def _seg(text: str, dy: float) -> _Segment:
    return _Segment(text, dy, LINE_FONT_SIZE if dy == 0 else LINE_FONT_SIZE * 0.72)


_BOND_GLYPH = {
    "single": "–",
    "double": "═",
    "triple": "≡",
    "quadruple": "≣",
    "wedge": "▲–",
    "hash": "–▨",
    "dative": "→",
    "wavy": "∼",
    "aromatic": ":",
}

_ARROW_GLYPH = {
    "forward": "→",
    "reverse": "←",
    "equilibrium": "⇌",
    "resonance": "↔",
}


def _segments_for(node: Node) -> list[_Segment]:
    if isinstance(node, Formula):
        return [s for child in node.nodes for s in _segments_for(child)]
    if isinstance(node, Atom):
        return _atom_segments(node)
    if isinstance(node, Group):
        segments = [_seg(node.open_char, 0.0)]
        for child in node.nodes:
            segments.extend(_segments_for(child))
        segments.append(_seg(node.close_char, 0.0))
        if node.multiplicity is not None:
            segments.append(_seg(node.multiplicity, 4.0))
        return segments
    if isinstance(node, Molecule):
        segments: list[_Segment] = []
        if node.stereo is not None:
            segments.append(_seg(f'({STEREO_LETTER[node.stereo]})-', 0.0))
        if node.coefficient is not None:
            segments.append(_seg(node.coefficient, 0.0))
        for child in node.nodes:
            segments.extend(_segments_for(child))
        return segments
    if isinstance(node, Bond):
        return [_seg(_BOND_GLYPH[node.kind], 0.0)]
    if isinstance(node, Reaction):
        segments = []
        for term in node.reactants:
            segments.extend(_segments_for(term))
        segments.append(_seg(" ", 0.0))
        segments.append(_seg(_ARROW_GLYPH[node.arrow], 0.0))
        _push_conditions(segments, node.conditions)
        segments.append(_seg(" ", 0.0))
        for term in node.products:
            segments.extend(_segments_for(term))
        return segments
    if isinstance(node, ReactionCascade):
        segments = []
        if node.steps:
            head = node.steps[0]
            for term in head.reactants:
                segments.extend(_segments_for(term))
            for step in node.steps:
                segments.append(_seg(" ", 0.0))
                segments.append(_seg(_ARROW_GLYPH[step.arrow], 0.0))
                _push_conditions(segments, step.conditions)
                segments.append(_seg(" ", 0.0))
                for term in step.products:
                    segments.extend(_segments_for(term))
        return segments
    if isinstance(node, ElectronConfiguration):
        segments = []
        for i, (orb, occ) in enumerate(node.orbitals):
            if i > 0:
                segments.append(_seg(" ", 0.0))
            segments.append(_seg(orb, 0.0))
            segments.append(_seg(occ, -6.0))
        return segments
    if isinstance(node, EmbeddedMath):
        return [_seg(f"`{node.source}`", 0.0)]
    if isinstance(node, Text):
        return [_seg(f'"{node.content}"', 0.0)]
    # Beyond-formulas nodes print the first line of their canonical text.
    return [_seg(render_text(node).split("\n")[0], 0.0)]


def _push_conditions(segments: list[_Segment], conditions) -> None:
    if conditions is None:
        return
    if conditions.above is not None:
        segments.append(_seg(f"[{conditions.above}]", -8.0))
    if conditions.below is not None:
        segments.append(_seg(f"[{conditions.below}]", 12.0))


def _atom_segments(atom: Atom) -> list[_Segment]:
    segments: list[_Segment] = []
    if atom.lone_pairs is not None:
        segments.append(_seg(":" * atom.lone_pairs, -6.0))
    if atom.isotope is not None:
        segments.append(_seg(atom.isotope, -6.0))
    segments.append(_seg(atom.element, 0.0))
    if atom.subscript is not None:
        segments.append(_seg(atom.subscript, 5.0))
    if atom.superscript is not None:
        segments.append(_seg(atom.superscript, -6.0))
    if atom.charge is not None:
        segments.append(_seg(atom.charge, -6.0))
    if atom.oxidation_state is not None:
        segments.append(_seg(f"({atom.oxidation_state})", -6.0))
    if atom.radical_electrons is not None:
        segments.append(_seg("·" * atom.radical_electrons, -6.0))
    if atom.ring_closures is not None:
        segments.append(_seg(atom.ring_closures, 0.0))
    return segments


def _linear_document(lines: list[list[_Segment]]) -> str:
    tspans = []
    for line_index, segments in enumerate(lines):
        baseline = 20 + line_index * LINE_HEIGHT
        x = 4.0
        out = ""
        for s in segments:
            out += (
                f'<tspan x="{_fmt(x)}" y="{_fmt(baseline + s.dy)}" font-size="{_fmt(s.size)}">'
                f"{_escape_xml(s.text)}</tspan>"
            )
            # Width counts characters (the JS reference's .length),
            # never bytes — bond glyphs are multibyte.
            x += len(s.text) * CHAR_WIDTH * (s.size / LINE_FONT_SIZE)
        tspans.append(
            f'    <text x="4" y="{_fmt(baseline)}" font-family="monospace" '
            f'font-size="{_fmt(LINE_FONT_SIZE)}">{out}</text>'
        )
    width = max(
        (sum(len(s.text) * CHAR_WIDTH for s in segments) for segments in lines),
        default=40.0,
    )
    width = max(width, 40.0)
    height = max(len(lines), 1) * LINE_HEIGHT + 12
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(width)} {_fmt(height)}" '
        f'width="{_fmt(width)}" height="{_fmt(height)}" role="img">\n' + "\n".join(tspans) + "\n  </svg>"
    )
