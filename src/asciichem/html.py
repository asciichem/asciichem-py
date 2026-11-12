"""Inline HTML formatter - the Python port of the reference
`Formatter::Html` (asciichem-ruby). Renders the model tree as inline
HTML using `<sub>`/`<sup>` tags, suitable for pages that don't load
MathJax or render MathML:

    H_2O  ->  <span class="asciichem">H<sub>2</sub>O</span>

Every model construct has one registered function (the registry
pattern text.py/mathml.py use); the CLI's `convert -t html` rides the
same entry point. Beyond-formula constructs (crystal, spectrum, calc,
z-matrix, mechanism) render as a semantic
`<span class="asciichem-<kind>">` with `<dl>` for key/value pairs and
`<ol>`/`<table>` for ordered data - the reference's class hooks that
let downstream CSS style per-construct.

Text content is escaped through `escape` - the four XML-mandated
characters only; model content is plain text (digits, element
symbols, condition text)."""
from __future__ import annotations

from .model import (Atom, Bond, Calculation, Crystal, ElectronConfiguration,
                    EmbeddedMath, Formula, Group, Mechanism, Molecule, Node,
                    Reaction, ReactionCascade, Spectrum, Text, ZMatrix,
                    STEREO_LETTER)

_FORMATTERS = {}


def _formats(cls):
    def register(fn):
        _FORMATTERS[cls] = fn
        return fn
    return register


def render(node: Node) -> str:
    return _FORMATTERS[type(node)](node)


def render_html(formula) -> str:
    """The formula's inline HTML (the CLI/formatter surface name)."""
    return render(formula)


def escape(string) -> str:
    """Minimal HTML escaper: the four XML-mandated characters, in the
    reference's order. Content is model text - nothing else needs it."""
    if string is None:
        return ""
    return (str(string)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# Per-format cell-parameter labels (the reference's Crystal::CELL_LABELS):
# the Greek angle names render as entities in HTML.
CELL_LABELS = {"a": "a", "b": "b", "c": "c",
               "alpha": "α", "beta": "β", "gamma": "γ"}


@_formats(Formula)
def _formula(f: Formula) -> str:
    inner = "".join(render(n) for n in f.nodes)
    return f'<span class="asciichem">{inner}</span>'


@_formats(Molecule)
def _molecule(m: Molecule) -> str:
    prefix = escape(m.coefficient) if m.coefficient else ""
    stereo = f"({escape(STEREO_LETTER[m.stereo])})-" if m.stereo else ""
    body = "".join(render(n) for n in m.nodes)
    return f"{stereo}{prefix}{body}"


@_formats(Atom)
def _atom(a: Atom) -> str:
    parts = []
    if a.isotope is not None:
        parts.append(f"<sup>{escape(a.isotope)}</sup>")
    parts.append(escape(a.element))
    if a.subscript is not None:
        parts.append(f"<sub>{escape(a.subscript)}</sub>")
    if a.superscript is not None:
        parts.append(f"<sup>{escape(a.superscript)}</sup>")
    if a.charge is not None:
        parts.append(f"<sup>{escape(a.charge)}</sup>")
    if a.oxidation_state is not None:
        parts.append(f"<sup>({escape(a.oxidation_state)})</sup>")
    if a.ring_closures is not None:
        parts.append(escape(a.ring_closures))
    return "".join(parts)


@_formats(Group)
def _group(g: Group) -> str:
    body = "".join(render(n) for n in g.nodes)
    suffix = f"<sub>{escape(g.multiplicity)}</sub>" if g.multiplicity else ""
    return f"{escape(g.open_char)}{body}{escape(g.close_char)}{suffix}"


@_formats(Bond)
def _bond(b: Bond) -> str:
    return escape(b.ascii)


@_formats(Reaction)
def _reaction(r: Reaction) -> str:
    left = " + ".join(render(n) for n in r.reactants)
    right = " + ".join(render(n) for n in r.products)
    conds = r.conditions
    if conds is None:
        return f"{left} {escape(r.arrow_entity)} {right}"
    above = f"<sup>{escape(conds.above)}</sup>" if conds.above else ""
    below = f"<sub>{escape(conds.below)}</sub>" if conds.below else ""
    return f"{left} {above}{escape(r.arrow_entity)}{below} {right}"


@_formats(ReactionCascade)
def _cascade(c: ReactionCascade) -> str:
    if not c.steps:
        return ""
    head = c.steps[0]
    out = " + ".join(render(n) for n in head.reactants)
    for step in c.steps:
        conds = step.conditions
        if conds is not None:
            above = f"<sup>{escape(conds.above)}</sup>" if conds.above else ""
            below = f"<sub>{escape(conds.below)}</sub>" if conds.below else ""
            out += f" {above}{escape(step.arrow_entity)}{below}"
        else:
            out += f" {escape(step.arrow_entity)}"
        out += " " + " + ".join(render(n) for n in step.products)
    return out


@_formats(ElectronConfiguration)
def _electron_configuration(ec: ElectronConfiguration) -> str:
    return " ".join(f"{escape(orb)}<sup>{escape(occ)}</sup>"
                    for orb, occ in ec.orbitals)


@_formats(EmbeddedMath)
def _embedded_math(em: EmbeddedMath) -> str:
    from . import mathml
    return mathml.render_fragment(em)


@_formats(Text)
def _text(t: Text) -> str:
    return escape(t.content)


# -- Beyond-formula constructs --------------------------------------------
#
# Each renders as a semantic <span class="asciichem-<kind>"> with nested
# <dl> for key/value pairs and <ol>/<table> for ordered lists.

@_formats(Crystal)
def _crystal(crystal: Crystal) -> str:
    parts = ['<span class="asciichem-crystal">']
    if crystal.name is not None:
        parts.append(f'<span class="asciichem-name">'
                     f'[{escape(crystal.name)}]</span>')
    parts.append(_kv_dl([(CELL_LABELS[key], getattr(crystal, key))
                         for key in Crystal.CELL_PARAMS
                         if getattr(crystal, key) is not None]))
    if crystal.atoms:
        parts.append('<ol class="asciichem-atoms">')
        for a in crystal.atoms:
            parts.append(f"<li>{render(a)}</li>")
        parts.append("</ol>")
    parts.append("</span>")
    return "".join(parts)


@_formats(Spectrum)
def _spectrum(spectrum: Spectrum) -> str:
    parts = ['<span class="asciichem-spectrum">']
    if spectrum.technique is not None:
        parts.append(f'<span class="asciichem-type">'
                     f'[{escape(spectrum.technique)}]</span>')
    parts.append(_kv_dl(list(spectrum.params.items())))
    if spectrum.peaks:
        parts.append('<table class="asciichem-peaks">')
        parts.append("<thead><tr><th>position</th><th>intensity</th>"
                     "<th>multiplicity</th><th>assignment</th></tr></thead>")
        parts.append("<tbody>")
        for peak in spectrum.peaks:
            parts.append("<tr>")
            parts.append(f"<td>{escape(peak.position)}</td>")
            parts.append(f"<td>{escape(peak.intensity)}</td>")
            parts.append(f"<td>{escape(peak.multiplicity)}</td>")
            parts.append(f"<td>{escape(peak.assignment)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
    parts.append("</span>")
    return "".join(parts)


@_formats(Calculation)
def _calculation(calc: Calculation) -> str:
    parts = ['<span class="asciichem-calc">']
    if calc.method is not None or calc.basis is not None:
        label = "/".join(v for v in (calc.method, calc.basis) if v is not None)
        parts.append(f'<span class="asciichem-method">[{escape(label)}]</span>')
    if calc.properties:
        parts.append('<dl class="asciichem-properties">')
        for p in calc.properties:
            value = escape(p.value)
            if p.units is not None:
                value += f" {escape(p.units)}"
            parts.append(f"<dt>{escape(p.title)}</dt><dd>{value}</dd>")
        parts.append("</dl>")
    parts.append("</span>")
    return "".join(parts)


@_formats(ZMatrix)
def _zmatrix(zm: ZMatrix) -> str:
    parts = ['<span class="asciichem-zmatrix">']
    if not zm.rows:
        return "".join(parts) + "</span>"
    parts.append('<table class="asciichem-rows"><tbody>')
    for row in zm.rows:
        cells = [row.atom]
        if row.ref1 is not None:
            cells += [row.ref1, row.distance]
        if row.ref2 is not None:
            cells += [row.ref2, row.angle]
        if row.ref3 is not None:
            cells += [row.ref3, row.dihedral]
        parts.append("<tr>" + "".join(f"<td>{escape(c)}</td>"
                                      for c in cells) + "</tr>")
    parts.append("</tbody></table></span>")
    return "".join(parts)


@_formats(Mechanism)
def _mechanism(mech: Mechanism) -> str:
    parts = ['<span class="asciichem-mechanism">']
    if not mech.steps and not mech.spectators:
        return "".join(parts) + "</span>"
    parts.append('<dl class="asciichem-steps">')
    for step in mech.steps:
        parts.append(f"<dt>{escape(step.label)}</dt>"
                     f"<dd>{escape(step.reaction)}</dd>")
    for spectator in mech.spectators:
        parts.append(f"<dt>spectator</dt><dd>{escape(spectator)}</dd>")
    parts.append("</dl></span>")
    return "".join(parts)


def _kv_dl(pairs) -> str:
    """Generic key-value pairs as a <dl>; empty when no pairs."""
    if not pairs:
        return ""
    parts = ["<dl>"]
    for key, value in pairs:
        parts.append(f"<dt>{escape(key)}</dt><dd>{escape(value)}</dd>")
    parts.append("</dl>")
    return "".join(parts)
