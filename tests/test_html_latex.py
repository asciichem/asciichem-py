"""The html and latex formatters: byte parity with the reference
implementations (`asciichem-ruby`'s Formatter::Html / Formatter::Latex
- the shapes the corpus's `convert -t html` / `-t latex` CLI cases
pin), determinism, and the beyond-formula constructs' semantic
wrapper shapes."""
import pytest

from asciichem import parse_text
from asciichem.html import render as render_html
from asciichem.latex import render as render_latex


# -- the reference-pinned shapes ------------------------------------------

@pytest.mark.parametrize("source,html,latex", [
    ("H_2O",
     '<span class="asciichem">H<sub>2</sub>O</span>',
     "\\ce{H2O}"),
    ("^14C",
     '<span class="asciichem"><sup>14</sup>C</span>',
     "\\ce{^14C}"),
    ("Ca^2+",
     '<span class="asciichem">Ca<sup>2+</sup></span>',
     "\\ce{Ca^{2+}}"),
    ("Ca^(II)",
     '<span class="asciichem">Ca<sup>(II)</sup></span>',
     "\\ce{Ca^{(II)}}"),
    ("C#C",
     '<span class="asciichem">C#C</span>',
     "\\ce{C\\\\equiv{}C}"),
    ("C=C",
     '<span class="asciichem">C=C</span>',
     "\\ce{C=C}"),
    ("(OH)_2",
     '<span class="asciichem">(OH)<sub>2</sub></span>',
     "\\ce{(OH)2}"),
    ("[OH]_2",
     '<span class="asciichem">[OH]<sub>2</sub></span>',
     "\\ce{[OH]2}"),
    ("{OH}_2",
     '<span class="asciichem">{OH}<sub>2</sub></span>',
     "\\ce{{OH}2}"),
    ("2H_2 + O_2 -> 2H_2O",
     '<span class="asciichem">2H<sub>2</sub> + O<sub>2</sub> '
     "→ 2H<sub>2</sub>O</span>",
     "\\ce{2H2 + O2 -> 2H2O}"),
    ("A ->[cat] B",
     '<span class="asciichem">A <sup>cat</sup>→ B</span>',
     "\\ce{A ->[cat] B}"),
    ("A <=> B",
     '<span class="asciichem">A ⇌ B</span>',
     "\\ce{A <=> B}"),
    ("1s^2 2s^2",
     '<span class="asciichem">1s<sup>2</sup> 2s<sup>2</sup></span>',
     "\\ce{\\ce{1s^2 2s^2}}"),
    ('"note: toxic"',
     '<span class="asciichem">note: toxic</span>',
     "\\ce{note: toxic}"),
])
def test_byte_parity_with_the_reference(source, html, latex):
    formula = parse_text(source)
    assert render_html(formula) == html, source
    assert render_latex(formula) == latex, source


def test_html_escapes_markup_characters():
    # Free text rides escaped: the four XML-mandated characters only.
    out = render_html(parse_text('"a < b && c > d"'))
    assert out == ('<span class="asciichem">'
                   "a &lt; b &amp;&amp; c &gt; d</span>")


def test_html_wraps_in_the_asciichem_span():
    assert render_html(parse_text("H_2O")).startswith('<span class='
                                                      '"asciichem">')


def test_latex_wraps_bare_molecule_in_ce():
    # A bare molecule (no formula wrapper) still renders inside \ce{}.
    assert render_latex(parse_text("H_2O").nodes[0]) == "\\ce{H2O}"


def test_latex_inside_ce_skips_the_molecule_wrapper():
    # Inside a formula body mhchem is already active - no nested \ce
    # around the molecule itself (the electron configuration's own
    # \ce{} is the reference's shape and stays).
    out = render_latex(parse_text("2H_2O"))
    assert out == "\\ce{2H2O}"


def test_both_formatters_are_deterministic():
    for source in ("H_2O", "2H_2 + O_2 -> 2H_2O", "Ca(OH)_2"):
        formula = parse_text(source)
        assert render_html(formula) == render_html(formula)
        assert render_latex(formula) == render_latex(formula)


# -- beyond-formula constructs ---------------------------------------------

def test_crystal_html_uses_the_semantic_wrapper():
    out = render_html(parse_text("crystal[x](a=1,b=2,alpha=90){Na}"))
    assert '<span class="asciichem-crystal">' in out
    assert '<span class="asciichem-name">[x]</span>' in out
    assert "<dl>" in out
    assert "<dt>α</dt>" in out  # html angle labels are the Greek names
    assert '<ol class="asciichem-atoms">' in out


def test_crystal_latex_uses_tabular_and_macro_labels():
    out = render_latex(parse_text("crystal[x](a=1,alpha=90){Na}"))
    assert "\\text{crystal}[x]" in out
    assert "\\begin{tabular}{ll}" in out
    assert "\\alpha" in out


def test_mechanism_html_lists_steps():
    out = render_html(parse_text("mechanism{E1: A -> B}"))
    assert '<span class="asciichem-mechanism">' in out
    assert "<dt>E1</dt>" in out


def test_mechanism_latex_uses_tabular():
    out = render_latex(parse_text("mechanism{E1: A -> B}"))
    assert "\\text{mechanism}" in out
    assert "E1 & A -> B \\\\" in out


def test_spectrum_html_renders_a_peak_table():
    out = render_html(parse_text('spectrum[nmr]{1.0: 3 s "CH_3"}'))
    assert '<table class="asciichem-peaks">' in out
    assert "<td>1.0</td>" in out
    assert "<td>CH_3</td>" in out
