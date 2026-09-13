"""Canonical AsciiChem text formatter - the round-trip contract:
parse(s).to_text() == s for canonical s. Port of the reference
Formatter::Text."""
from __future__ import annotations

from .model import (Atom, Bond, Calculation, Crystal, ElectronConfiguration,
                    EmbeddedMath, Formula, Group, Mechanism, Molecule, Node,
                    Reaction, ReactionCascade, Spectrum, Text, ZMatrix,
                    ReactionConditions, STEREO_LETTER)

_FORMATTERS = {}


def _formats(cls):
    def register(fn):
        _FORMATTERS[cls] = fn
        return fn
    return register


def render(node: Node) -> str:
    return _FORMATTERS[type(node)](node)


@_formats(Formula)
def _formula(f: Formula) -> str:
    return " ".join(render(n) for n in f.nodes)


@_formats(Molecule)
def _molecule(m: Molecule) -> str:
    prefix = m.coefficient if m.coefficient else ""
    stereo = f"({STEREO_LETTER[m.stereo]})-" if m.stereo else ""
    body = "".join(render(n) for n in m.nodes)
    return stereo + prefix + body + _annotations(m)


def _annotations(m: Molecule) -> str:
    parts = []
    for n in m.names:
        parts.append(f'@name("{n.content}")')
    for i in m.identifiers:
        parts.append(f'@{i.convention}("{i.value}")')
    if m.title is not None:
        parts.append(f'@title("{m.title}")')
    for f in m.formulas:
        if f.get("concise"):
            parts.append(f'@formula("{f["concise"]}")')
    for l in m.labels:
        if l.get("value"):
            parts.append(f'@label("{l["value"]}")')
    for p in m.properties:
        if p.get("title") and p.get("value"):
            parts.append(f'@{p["title"]}("{p["value"]}")')
    for meta in m.metadata:
        parts.append(f'@meta("{meta["name"]}","{meta["content"]}")')
    return " " + " ".join(parts) if parts else ""


@_formats(Atom)
def _atom(a: Atom) -> str:
    parts = []
    if a.lone_pairs is not None:
        parts.append(":" * a.lone_pairs)
    if a.isotope is not None:
        parts.append("^" + a.isotope)
    parts.append(a.element)
    if a.subscript is not None:
        parts.append("_" + a.subscript)
    if a.superscript is not None:
        parts.append("^" + a.superscript)
    if a.charge is not None:
        parts.append("^" + a.charge)
    if a.oxidation_state is not None:
        parts.append(f"^({a.oxidation_state})")
    if a.radical_electrons is not None:
        parts.append("." * a.radical_electrons)
    if a.ring_closures is not None:
        parts.append(a.ring_closures)
    parts.append(_atom_annotation(a))
    return "".join(parts)


def _coord(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def _atom_annotation(a: Atom) -> str:
    parts = []
    if a.x2 is not None and a.y2 is not None:
        coord = f"@({_coord(a.x2)},{_coord(a.y2)}"
        if a.z2 is not None:
            coord += f",{_coord(a.z2)}"
        parts.append(coord + ")")
    if a.atom_parity is not None:
        parts.append("@" + a.atom_parity)
    if a.spin_multiplicity is not None:
        parts.append(f"@m({a.spin_multiplicity})")
    if a.atom_title is not None:
        parts.append(f'@t("{a.atom_title}")')
    if a.x_fract is not None and a.y_fract is not None and a.z_fract is not None:
        parts.append(f"@f({_coord(a.x_fract)},{_coord(a.y_fract)},{_coord(a.z_fract)})")
    return "".join(parts)


@_formats(Group)
def _group(g: Group) -> str:
    body = "".join(render(n) for n in g.nodes)
    suffix = f"_{g.multiplicity}" if g.multiplicity is not None else ""
    return f"{g.open_char}{body}{g.close_char}{suffix}"


@_formats(Bond)
def _bond(b: Bond) -> str:
    return b.ascii


@_formats(Reaction)
def _reaction(r: Reaction) -> str:
    left = " + ".join(render(n) for n in r.reactants)
    right = " + ".join(render(n) for n in r.products)
    return f"{left} {_arrow_with_conditions(r)} {right}"


def _arrow_with_conditions(r: Reaction) -> str:
    arrow = r.arrow_ascii
    if r.conditions is not None:
        if r.conditions.above is not None:
            arrow += f"[{r.conditions.above}]"
        if r.conditions.below is not None:
            arrow += f"[{r.conditions.below}]"
    return arrow


@_formats(ReactionCascade)
def _cascade(c: ReactionCascade) -> str:
    if not c.steps:
        return ""
    head = c.steps[0]
    out = f"{' + '.join(render(n) for n in head.reactants)} {_arrow_with_conditions(head)} {' + '.join(render(n) for n in head.products)}"
    for step in c.steps[1:]:
        out += f" {_arrow_with_conditions(step)} {' + '.join(render(n) for n in step.products)}"
    return out


@_formats(ElectronConfiguration)
def _ec(ec: ElectronConfiguration) -> str:
    parts = [f"{orb}^{occ}" for orb, occ in ec.orbitals]
    if ec.term_symbol:
        m, letter, j = ec.term_symbol
        parts.append(f"{m or ''}{letter or ''}" + (f"_{j}" if j else ""))
    return " ".join(parts)


@_formats(EmbeddedMath)
def _math(m: EmbeddedMath) -> str:
    return f"`{m.source}`"


@_formats(Text)
def _text(t: Text) -> str:
    return f'"{t.content}"'


@_formats(Crystal)
def _crystal(c: Crystal) -> str:
    parts = ["crystal"]
    if c.name is not None:
        parts.append(f"[{c.name}]")
    params = []
    for key, attr in (("a", "a"), ("b", "b"), ("c", "c"), ("alpha", "alpha"),
                      ("beta", "beta"), ("gamma", "gamma"), ("sg", "spacegroup")):
        value = getattr(c, attr)
        if value is not None:
            params.append(f"{key}={value}")
    if params:
        parts.append("(" + ",".join(params) + ")")
    if c.atoms:
        parts.append("{" + " ".join(render(a) for a in c.atoms) + "}")
    return "".join(parts)


@_formats(Spectrum)
def _spectrum(s: Spectrum) -> str:
    parts = ["spectrum"]
    if s.technique is not None:
        parts.append(f"[{s.technique}]")
    if s.params:
        parts.append("(" + ",".join(f"{k}={v}" for k, v in s.params.items()) + ")")
    if s.peaks:
        lines = []
        for p in s.peaks:
            line = f"{p.position}: {p.intensity}"
            if p.multiplicity is not None:
                line += f" {p.multiplicity}"
            if p.assignment is not None:
                line += f' "{p.assignment}"'
            lines.append(line)
        parts.append("{\n  " + "\n  ".join(lines) + "\n}")
    return "".join(parts)


@_formats(Calculation)
def _calculation(c: Calculation) -> str:
    parts = ["calc"]
    params = [p for p in (c.method, c.basis) if p is not None]
    if params:
        parts.append("(" + "/".join(params) + ")")
    if c.properties:
        lines = []
        for p in c.properties:
            line = f"{p.title}: {p.value}"
            if p.units is not None:
                line += f" {p.units}"
            lines.append(line)
        parts.append("{\n  " + "\n  ".join(lines) + "\n}")
    return "".join(parts)


@_formats(ZMatrix)
def _zmatrix(z: ZMatrix) -> str:
    parts = ["zmatrix"]
    if z.rows:
        lines = []
        for r in z.rows:
            tokens = [r.atom]
            if r.ref1 is not None:
                tokens += [r.ref1, r.distance]
            if r.ref2 is not None:
                tokens += [r.ref2, r.angle]
            if r.ref3 is not None:
                tokens += [r.ref3, r.dihedral]
            lines.append("  ".join(t for t in tokens if t is not None))
        parts.append("{\n  " + "\n  ".join(lines) + "\n}")
    return "".join(parts)


@_formats(Mechanism)
def _mechanism(m: Mechanism) -> str:
    parts = ["mechanism"]
    if m.steps or m.spectators:
        lines = [f"{s.label}: {s.reaction}" for s in m.steps]
        lines += [f"spectator: {sp}" for sp in m.spectators]
        parts.append("{\n  " + "\n  ".join(lines) + "\n}")
    return "".join(parts)
