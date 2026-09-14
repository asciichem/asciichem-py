"""Presentation MathML formatter - byte-identical port of the
reference AsciiChem::Formatter::Mathml (single-contract rule: the
shared mathml corpus goldens are exact string comparisons).

Emission mirrors Nokogiri's default pretty-printing: elements with
element children are multi-line with 2-space indent; text-only
elements are single-line; empty elements self-close."""
from __future__ import annotations

import re

from .model import (Atom, Bond, Calculation, Crystal, ElectronConfiguration,
                    EmbeddedMath, Formula, Group, Mechanism, Molecule, Node,
                    Reaction, ReactionCascade, Spectrum, Text, ZMatrix,
                    STEREO_LETTER, AsciiChemError)
from .parser import parse_text

MATHML_NS = "http://www.w3.org/1998/Math/MathML"

_FORMATTERS = {}


def _formats(cls):
    def register(fn):
        _FORMATTERS[cls] = fn
        return fn
    return register


class _El:
    """Immutable-ish XML element: either text-bearing or child-bearing."""

    __slots__ = ("name", "attrs", "text", "children")

    def __init__(self, name, attrs=None, text=None, children=None):
        self.name = name
        self.attrs = attrs
        self.text = text
        self.children = children or []


def _escape_text(value: str) -> str:
    return (value.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))


def _escape_attr(value: str) -> str:
    return _escape_text(value).replace('"', "&quot;")


def _serialize(el: _El, depth: int = 0) -> str:
    pad = " " * (depth * 2)
    attrs = ""
    if el.attrs:
        attrs = "".join(f' {k}="{_escape_attr(v)}"' for k, v in el.attrs.items())
    if el.text is not None:
        if el.text == "":
            return f"{pad}<{el.name}{attrs}/>"
        return f"{pad}<{el.name}{attrs}>{_escape_text(el.text)}</{el.name}>"
    if not el.children:
        return f"{pad}<{el.name}{attrs}/>"
    inner = "\n".join(_serialize(c, depth + 1) for c in el.children)
    return f"{pad}<{el.name}{attrs}>\n{inner}\n{pad}</{el.name}>"


def _el(name, attrs=None, children=None):
    return _El(name, attrs=attrs, children=children)


def _mi(content) -> _El:
    return _El("mi", attrs={"mathvariant": "normal"}, text=str(content))


def _mn(content) -> _El:
    return _El("mn", text=str(content))


def _mo(content) -> _El:
    return _El("mo", text=str(content))


def _mtext(content) -> _El:
    return _El("mtext", text=str(content))


def render(node: Node) -> _El:
    return _FORMATTERS[type(node)](node)


def render_mathml(formula) -> str:
    xml = _serialize(render(formula))
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml}\n'


@_formats(Formula)
def _formula(f: Formula) -> _El:
    return _el("math", {"xmlns": MATHML_NS}, [_el("mrow", None, [render(n) for n in f.nodes])])


@_formats(Molecule)
def _molecule(m: Molecule) -> _El:
    children = []
    if m.stereo:
        children.append(_mtext(f"({STEREO_LETTER[m.stereo]})-"))
    if m.coefficient is not None:
        children.append(_mn(m.coefficient))
    children.extend(render(n) for n in m.nodes)
    return _el("mrow", None, children)


@_formats(Atom)
def _atom(atom: Atom) -> _El:
    base = _mi(atom.element)
    if atom.isotope is not None:
        base = _el("mmultiscripts", None,
                   [base, _el("none"), _el("none"), _el("mprescripts"),
                    _el("none"), _mn(atom.isotope)])
    if atom.lone_pairs is not None:
        base = _el("mrow", None, [_mtext(":" * atom.lone_pairs), base])
    base = _wrap_sub_and_super(base, atom)
    if atom.radical_electrons is not None:
        base = _el("mrow", None, [base, _mtext("." * atom.radical_electrons)])
    if atom.ring_closures is not None:
        base = _el("mrow", None, [base, _mn(atom.ring_closures)])
    return base


def _wrap_sub_and_super(base: _El, atom: Atom) -> _El:
    has_sub = atom.subscript not in (None, "")
    super_node = _super_element(atom)
    if not has_sub and super_node is None:
        return base
    if has_sub and super_node is None:
        return _el("msub", None, [base, _mn(atom.subscript)])
    if not has_sub and super_node is not None:
        return _el("msup", None, [base, super_node])
    return _el("msubsup", None, [base, _mn(atom.subscript), super_node])


def _super_element(atom: Atom):
    # charge > oxidation state > raw superscript (reference priority).
    if atom.charge is not None:
        row = _charge_row(atom.charge)
        if row is not None:
            return row
    if atom.oxidation_state is not None:
        return _el("mrow", None, [_mi(atom.oxidation_state)])
    if atom.superscript is not None:
        return _mn(atom.superscript)
    return None


def _charge_row(charge: str):
    digits_sign = re.match(r"^(\d*)([+-])$", charge)
    sign_digits = re.match(r"^([+-])(\d*)$", charge) if digits_sign is None else None
    if digits_sign is None and sign_digits is None:
        return None
    digits = digits_sign.group(1) if digits_sign else sign_digits.group(2)
    sign = digits_sign.group(2) if digits_sign else sign_digits.group(1)
    children = []
    if digits != "":
        children.append(_mn(digits))
    children.append(_mo(sign))
    return _el("mrow", None, children)


@_formats(Bond)
def _bond(b: Bond) -> _El:
    return _mo(b.entity)


@_formats(Group)
def _group(g: Group) -> _El:
    children = [_mo(g.open_char)]
    children.extend(render(n) for n in g.nodes)
    children.append(_mo(g.close_char))
    row = _el("mrow", None, children)
    if g.multiplicity is None:
        return row
    return _el("msub", None, [row, _mn(g.multiplicity)])


def _add_terms(children: list, terms) -> None:
    for index, term in enumerate(terms):
        if index > 0:
            children.append(_mo("+"))
        children.append(render(term))


def _render_arrow(reaction: Reaction) -> _El:
    op = _mo(reaction.arrow_entity)
    if reaction.conditions is None:
        return op
    above = reaction.conditions.above
    below = reaction.conditions.below
    if above is None and below is None:
        return op
    if above is not None and below is not None:
        name = "munderover"
    elif above is not None:
        name = "mover"
    else:
        name = "munder"
    children = [op]
    if above is not None:
        children.append(_render_condition(above))
    if below is not None:
        children.append(_render_condition(below))
    return _el(name, None, children)


def _render_condition(text):
    if not text:
        return _mtext("")
    try:
        formula = parse_text(text)
        return _el("mrow", None, [render(n) for n in formula.nodes])
    except AsciiChemError:
        return _mtext(text)


@_formats(Reaction)
def _reaction(r: Reaction) -> _El:
    children = []
    _add_terms(children, r.reactants)
    children.append(_render_arrow(r))
    _add_terms(children, r.products)
    return _el("mrow", None, children)


@_formats(ReactionCascade)
def _cascade(c: ReactionCascade) -> _El:
    if not c.steps:
        return _el("mrow")
    children = []
    _add_terms(children, c.steps[0].reactants)
    for step in c.steps:
        children.append(_render_arrow(step))
        _add_terms(children, step.products)
    return _el("mrow", None, children)


@_formats(ElectronConfiguration)
def _electron_configuration(ec: ElectronConfiguration) -> _El:
    children = []
    for index, (orbital, occupancy) in enumerate(ec.orbitals):
        if index > 0:
            children.append(_mo(" "))
        children.append(_el("msup", None, [_mi(orbital), _mn(occupancy)]))
    return _el("mrow", None, children)


@_formats(EmbeddedMath)
def _embedded_math(em: EmbeddedMath) -> _El:
    # Spec'd loss (TODO.impl/61): the reference embeds Plurimath
    # MathML; this port has no AsciiMath engine, so the source
    # degrades to <mtext>.
    return _mtext(em.source)


@_formats(Text)
def _text(t: Text) -> _El:
    return _mtext(t.content)


def _named_bracket(content) -> _El:
    return _el("mrow", None, [_mo("["), _mtext(content), _mo("]")])


def _simple_table(rows) -> _El:
    return _el("mtable", None, [
        _el("mtr", None, [_el("mtd", None, [cell]) for cell in row])
        for row in rows
    ])


_CELL_LABELS = {"a": "a", "b": "b", "c": "c", "alpha": "α", "beta": "β", "gamma": "γ"}


@_formats(Crystal)
def _crystal(crystal: Crystal) -> _El:
    children = [_mi("crystal")]
    if crystal.name is not None:
        children.append(_named_bracket(crystal.name))
    params = []
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        value = getattr(crystal, key)
        if value is not None:
            params.append([_mi(_CELL_LABELS[key]), _mn(value)])
    if params:
        children.append(_simple_table(params))
    if crystal.atoms:
        rows = [[_mn(i + 1), render(atom)] for i, atom in enumerate(crystal.atoms)]
        children.append(_simple_table(rows))
    return _el("mrow", None, children)


@_formats(Spectrum)
def _spectrum(spectrum: Spectrum) -> _El:
    children = [_mi("spectrum")]
    if spectrum.technique is not None:
        children.append(_named_bracket(spectrum.technique))
    if spectrum.params:
        rows = [[_mi(key), _mtext(value)] for key, value in spectrum.params.items()]
        children.append(_simple_table(rows))
    if spectrum.peaks:
        rows = []
        for peak in spectrum.peaks:
            row = [_mn(peak.position or ""), _mn(peak.intensity or "")]
            if peak.multiplicity is not None:
                row.append(_mi(peak.multiplicity))
            if peak.assignment is not None:
                row.append(_mtext(peak.assignment))
            rows.append(row)
        children.append(_simple_table(rows))
    return _el("mrow", None, children)


@_formats(Calculation)
def _calculation(calc: Calculation) -> _El:
    children = [_mi("calc")]
    if calc.method is not None or calc.basis is not None:
        children.append(_named_bracket("/".join(p for p in (calc.method, calc.basis) if p)))
    if calc.properties:
        rows = []
        for prop in calc.properties:
            row = [_mi(prop.title), _mn(prop.value)]
            if prop.units is not None:
                row.append(_mi(prop.units))
            rows.append(row)
        children.append(_simple_table(rows))
    return _el("mrow", None, children)


@_formats(ZMatrix)
def _zmatrix(zm: ZMatrix) -> _El:
    children = [_mi("zmatrix")]
    if zm.rows:
        rows = []
        for row in zm.rows:
            cells = [_mi(row.atom)]
            if row.ref1 is not None:
                cells.extend([_mi(row.ref1), _mn(row.distance or "")])
            if row.ref2 is not None:
                cells.extend([_mi(row.ref2), _mn(row.angle or "")])
            if row.ref3 is not None:
                cells.extend([_mi(row.ref3), _mn(row.dihedral or "")])
            rows.append(cells)
        children.append(_simple_table(rows))
    return _el("mrow", None, children)


@_formats(Mechanism)
def _mechanism(mech: Mechanism) -> _El:
    children = [_mi("mechanism")]
    if mech.steps or getattr(mech, "spectators", []):
        rows = [[_mi(s.label), _mtext(s.reaction)] for s in mech.steps]
        for spectator in getattr(mech, "spectators", []):
            rows.append([_mi("spectator"), _mtext(spectator)])
        children.append(_simple_table(rows))
    return _el("mrow", None, children)
