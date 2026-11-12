"""LaTeX formatter - the Python port of the reference
`Formatter::Latex` (asciichem-ruby). Renders the model tree as LaTeX
wrapped in `mhchem`'s `\\ce{...}` at the formula level only. Output
requires `mhchem` in the LaTeX preamble:

    \\usepackage[version=4]{mhchem}

    H_2O  ->  \\ce{H2O}

Every model construct has one registered function (the same registry
pattern text.py/mathml.py use); the CLI's `convert -t latex` rides the
same entry point. mhchem accepts bare digit runs after an element as
subscripts and after `^` as superscripts, so digit-only values render
bare and anything else takes braces (`wrap`/`mhchem_subscript` - the
reference's helpers).

Beyond-formula constructs (crystal, spectrum, calc, z-matrix,
mechanism) render as `\\text{kind}[label]{ tabular body }` - clean
inline LaTeX that survives both pdflatex and xelatex, deliberately
outside mhchem's `\\ce{}` (cell params and peaks are not chemistry)."""
from __future__ import annotations

import re

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


def render_latex(formula) -> str:
    """The formula's LaTeX (the CLI/formatter surface name)."""
    return render(formula)


# The `within_ce` state: inside a formula body mhchem is already
# active, so bare molecules skip the per-molecule \\ce wrapper (the
# reference's @inside_ce flag on the formatter instance).
_inside_ce = False


class _within_ce:
    """Guard: set the flag for the duration of the formula body,
    restoring the previous value after (the reference's within_ce)."""

    def __enter__(self):
        global _inside_ce
        self._previous = _inside_ce
        _inside_ce = True
        return self

    def __exit__(self, *exc):
        global _inside_ce
        _inside_ce = self._previous
        return False


# Per-format cell-parameter labels (the reference's Crystal::CELL_LABELS):
# the Greek angle names render as LaTeX macros.
CELL_LABELS = {"a": "a", "b": "b", "c": "c",
               "alpha": "\\alpha", "beta": "\\beta", "gamma": "\\gamma"}


def mhchem_subscript(value) -> str:
    """mhchem accepts bare digits after an element as subscripts. Use
    the bare form for digit-only values; brace form for anything else."""
    s = str(value)
    return s if s.isdigit() else "_{" + s + "}"


def wrap(value) -> str:
    """mhchem accepts bare digits (and letters) for superscript values
    (`^14`, `_12`). Use braces only for non-alphanumeric values."""
    s = str(value)
    kept = s.isascii() and (s.isalnum())
    return s if kept else "{" + s + "}"


def _format_number(value) -> str:
    """Render a cell/peak value the way the reference prints it."""
    return str(value)


@_formats(Formula)
def _formula(f: Formula) -> str:
    with _within_ce():
        body = "".join(render(n) for n in f.nodes)
    return "\\ce{" + body + "}"


@_formats(Molecule)
def _molecule(m: Molecule) -> str:
    prefix = "" if m.coefficient is None or m.coefficient == "" else str(m.coefficient)
    stereo = f"({STEREO_LETTER[m.stereo]})-" if m.stereo else ""
    body = "".join(render(n) for n in m.nodes)
    if _inside_ce:
        return f"{stereo}{prefix}{body}"
    return f"\\ce{{{stereo}{prefix}{body}}}"


@_formats(Atom)
def _atom(a: Atom) -> str:
    parts = []
    if a.lone_pairs is not None:
        parts.append(":" * a.lone_pairs)
    if a.isotope is not None:
        parts.append("^" + wrap(a.isotope))
    parts.append(wrap(a.element))
    if a.subscript is not None:
        parts.append(mhchem_subscript(a.subscript))
    if a.charge is not None:
        parts.append("^" + wrap(a.charge))
    elif a.oxidation_state is not None:
        parts.append("^{(" + wrap(a.oxidation_state) + ")}")
    elif a.superscript is not None:
        parts.append("^" + wrap(a.superscript))
    if a.radical_electrons is not None:
        parts.append("." * a.radical_electrons)
    if a.ring_closures is not None:
        parts.append(str(a.ring_closures))
    return "".join(parts)


@_formats(Group)
def _group(g: Group) -> str:
    body = "".join(render(n) for n in g.nodes)
    open_char, close_char = BRACKET_CHARS.get(g.bracket, ("(", ")"))
    suffix = mhchem_subscript(g.multiplicity) if g.multiplicity is not None else ""
    return f"{open_char}{body}{close_char}{suffix}"


# Bracket characters per kind (the reference's bracket_chars; anything
# outside the three kinds falls back to parens).
BRACKET_CHARS = {"paren": ("(", ")"), "square": ("[", "]"),
                 "brace": ("{", "}")}

ARROWS = {"forward": "->", "reverse": "<-", "equilibrium": "<=>",
          "resonance": "<->"}


@_formats(Bond)
def _bond(b: Bond) -> str:
    if b.kind == "single":
        return "-"
    if b.kind == "double":
        return "="
    if b.kind == "triple":
        return "\\\\equiv{}"
    return b.ascii


@_formats(Reaction)
def _reaction(r: Reaction) -> str:
    left = " + ".join(render(n) for n in r.reactants)
    right = " + ".join(render(n) for n in r.products)
    arrow = ARROWS.get(r.arrow, "->")
    conds = r.conditions
    if conds is None:
        return f"{left} {arrow} {right}"
    above = _conditions_bracket(conds.above)
    below = _conditions_bracket(conds.below)
    return f"{left} {arrow}{above}{below} {right}"


@_formats(ReactionCascade)
def _cascade(c: ReactionCascade) -> str:
    if not c.steps:
        return ""
    head = c.steps[0]
    out = " + ".join(render(n) for n in head.reactants)
    for step in c.steps:
        arrow = ARROWS.get(step.arrow, "->")
        conds = step.conditions
        if conds is not None:
            above = _conditions_bracket(conds.above)
            below = _conditions_bracket(conds.below)
            out += f" {arrow}{above}{below}"
        else:
            out += f" {arrow}"
        out += " " + " + ".join(render(n) for n in step.products)
    return out


@_formats(ElectronConfiguration)
def _electron_configuration(ec: ElectronConfiguration) -> str:
    inner = " ".join(f"{wrap(orb)}^{wrap(occ)}" for orb, occ in ec.orbitals)
    return "\\ce{" + inner + "}"


@_formats(EmbeddedMath)
def _embedded_math(em: EmbeddedMath) -> str:
    # The reference delegates to Plurimath's to_latex; this package
    # carries no math-to-LaTeX converter, so the ASCII math source
    # rides inside $...$ verbatim (math-mode compatible for the
    # common cases - a documented deviation).
    return "$" + em.source + "$"


@_formats(Text)
def _text(t: Text) -> str:
    return t.content


def _conditions_bracket(value) -> str:
    if value is None or value == "":
        return ""
    return "[" + str(value) + "]"


# -- Beyond-formula constructs --------------------------------------------
#
# Each renders as `\\text{kind}[label]{ tabular body }`.

@_formats(Crystal)
def _crystal(crystal: Crystal) -> str:
    parts = ["\\text{crystal}"]
    if crystal.name is not None:
        parts.append("[" + _wrap_latex(crystal.name) + "]")
    cell_pairs = [(CELL_LABELS[key], _format_number(getattr(crystal, key)))
                  for key in Crystal.CELL_PARAMS
                  if getattr(crystal, key) is not None]
    if cell_pairs:
        parts.append(_latex_tabular(cell_pairs))
    if crystal.atoms:
        rows = [[render(a)] for a in crystal.atoms]
        parts.append(_latex_tabular(rows, columns=1))
    return "".join(parts)


@_formats(Spectrum)
def _spectrum(spectrum: Spectrum) -> str:
    parts = ["\\text{spectrum}"]
    if spectrum.technique is not None:
        parts.append("[" + _wrap_latex(spectrum.technique) + "]")
    if spectrum.params:
        parts.append(_latex_tabular(list(spectrum.params.items())))
    if spectrum.peaks:
        rows = []
        for peak in spectrum.peaks:
            cells = [str(peak.position), str(peak.intensity)]
            if peak.multiplicity is not None:
                cells.append(str(peak.multiplicity))
            if peak.assignment is not None:
                cells.append(f'"{peak.assignment}"')
            rows.append(cells)
        parts.append(_latex_tabular(rows))
    return "".join(parts)


@_formats(Calculation)
def _calculation(calc: Calculation) -> str:
    parts = ["\\text{calc}"]
    if calc.method is not None or calc.basis is not None:
        label = "/".join(v for v in (calc.method, calc.basis) if v is not None)
        parts.append("[" + _wrap_latex(label) + "]")
    if calc.properties:
        rows = []
        for p in calc.properties:
            cells = [str(p.title), str(p.value)]
            if p.units is not None:
                cells.append(str(p.units))
            rows.append(cells)
        parts.append(_latex_tabular(rows))
    return "".join(parts)


@_formats(ZMatrix)
def _zmatrix(zm: ZMatrix) -> str:
    parts = ["\\text{zmatrix}"]
    if not zm.rows:
        return "".join(parts)
    rows = []
    for row in zm.rows:
        cells = [str(row.atom)]
        if row.ref1 is not None:
            cells += [str(row.ref1), str(row.distance)]
        if row.ref2 is not None:
            cells += [str(row.ref2), str(row.angle)]
        if row.ref3 is not None:
            cells += [str(row.ref3), str(row.dihedral)]
        rows.append(cells)
    parts.append(_latex_tabular(rows))
    return "".join(parts)


@_formats(Mechanism)
def _mechanism(mech: Mechanism) -> str:
    parts = ["\\text{mechanism}"]
    if not mech.steps and not mech.spectators:
        return "".join(parts)
    rows = [[str(s.label), str(s.reaction)] for s in mech.steps]
    rows += [["spectator", str(sp)] for sp in mech.spectators]
    parts.append(_latex_tabular(rows))
    return "".join(parts)


def _wrap_latex(content) -> str:
    """Wrap content in \\text{} when it carries spaces or special
    characters, so tabular cells render cleanly in pdflatex and
    xelatex alike."""
    s = str(content)
    if re.fullmatch(r"[a-zA-Z0-9+\-\/.*]+", s):
        return s
    return "\\text{" + s + "}"


def _latex_tabular(rows, columns: int | None = None) -> str:
    """Build a tabular environment from rows of cells. Column count is
    the max row length (rows may be ragged, e.g. z-matrix rows)."""
    if not rows:
        return ""
    cols = columns if columns is not None else max(len(list(r)) for r in rows)
    alignment = "l" * cols
    lines = []
    for cells in rows:
        padded = list(cells)[:cols]
        padded += [""] * (cols - len(padded))
        lines.append(" & ".join(str(c) for c in padded) + " \\\\")
    body = "\n".join(lines)
    return f"\\begin{{tabular}}{{{alignment}}}\n{body}\n\\end{{tabular}}"
