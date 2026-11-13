"""The crystal node (rs#28's authoritative shape): cell parameters
carry the AUTHORED strings in the model; the wire casts to numbers and
back; formatters consult `cell_params()`/`greek_label()` rather than
inlining field lists; the fractional coordinates ride the atoms and
are a spec'd wire loss (the v1 atom schema carries no coordinates)."""
import pytest

from asciichem import parse_text
from asciichem.model import Crystal, ParseError
from asciichem.text import render
from asciichem.wire import from_model_json, to_model_json


def crystal(source="crystal[NaCl](a=5.64,b=5.64,c=5.64,alpha=90,beta=90,gamma=90,sg=Fm-3m)"):
    return parse_text(source).nodes[0]


# -- model: authored strings, canonical iteration --------------------------

def test_cell_parameters_keep_the_authored_strings():
    parsed = crystal()
    assert parsed.a == "5.64"
    assert parsed.alpha == "90"
    assert parsed.spacegroup == "Fm-3m"
    assert parsed.name == "NaCl"


def test_cell_params_lists_the_set_fields_in_canonical_order():
    parsed = crystal("crystal(a=1,gamma=60)")
    assert parsed.cell_params() == [("a", "1"), ("gamma", "60")]
    assert crystal().cell_params() == [
        ("a", "5.64"), ("b", "5.64"), ("c", "5.64"),
        ("alpha", "90"), ("beta", "90"), ("gamma", "90")]


def test_text_round_trips_the_authored_form_exactly():
    source = "crystal(a=1,b=2,c=3)"
    assert render(parse_text(source)) == source


def test_greek_labels_per_format():
    assert Crystal.greek_label("alpha", "text") == "alpha"
    assert Crystal.greek_label("alpha", "mathml") == "α"
    assert Crystal.greek_label("gamma", "html") == "γ"
    assert Crystal.greek_label("beta", "latex") == "\\beta"
    assert Crystal.greek_label("a", "latex") == "a"


def test_fractional_coordinates_ride_the_atoms():
    parsed = crystal("crystal[NaCl](a=5.64){Na@f(0,0,0) Cl@f(0.5,0.5,0.5)}")
    assert [a.element for a in parsed.atoms] == ["Na", "Cl"]
    assert parsed.atoms[0].x_fract == 0.0
    assert parsed.atoms[1].z_fract == 0.5


# -- wire: numbers on the wire, authored strings in the model --------------

def test_wire_carries_numbers():
    import json
    wire = json.loads(to_model_json(parse_text(
        "crystal[NaCl](a=5.64,alpha=90,sg=Fm-3m)")))
    node = wire["nodes"][0]
    assert node["type"] == "crystal"
    assert node["a"] == 5.64
    assert node["alpha"] == 90.0
    assert node["spacegroup"] == "Fm-3m"


def test_wire_rejects_a_non_numeric_cell_param():
    parsed = crystal("crystal(a=soft)")
    with pytest.raises(ParseError, match="not a number"):
        to_model_json(parsed)


def test_wire_round_trip_casts_back_to_the_display_shape():
    source = "crystal[NaCl](a=5.64,b=2,alpha=90,sg=Fm-3m)"
    restored = from_model_json(to_model_json(parse_text(source)))
    node = restored.nodes[0]
    # 90 -> "90" (integral values drop the decimal), 5.64 -> "5.64".
    assert node.a == "5.64"
    assert node.alpha == "90"
    assert node.b == "2"
    assert render(restored) == source


def test_wire_round_trip_spec_loss_fractional_coordinates():
    # The v1 atom schema carries no coordinates; the wire round trip
    # loses the @f annotations (the reference's documented gap) while
    # the cell parameters and space group survive.
    source = "crystal[NaCl](a=5.64,sg=Fm-3m){Na@f(0,0,0)}"
    restored = from_model_json(to_model_json(parse_text(source)))
    assert render(restored) == "crystal[NaCl](a=5.64,sg=Fm-3m){Na}"


# -- renderers: the greek labels are the single source -----------------------

def test_mathml_renders_greek_labels_and_cell_table():
    from asciichem.mathml import render_mathml
    out = render_mathml(parse_text("crystal[NaCl](a=1,alpha=90){Na}"))
    assert '<mi mathvariant="normal">α</mi>' in out
    assert '<mi mathvariant="normal">a</mi>' in out
    assert '<mi mathvariant="normal">crystal</mi>' in out


def test_html_renders_greek_entities():
    from asciichem.html import render_html
    out = render_html(parse_text("crystal[x](a=1,alpha=90){Na}"))
    assert "<dt>α</dt><dd>90</dd>" in out


def test_latex_renders_greek_macros():
    from asciichem.latex import render_latex
    out = render_latex(parse_text("crystal[x](a=1,beta=90){Na}"))
    assert "\\text{crystal}[x]" in out
    assert "\\beta & 90" in out
