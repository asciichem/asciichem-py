import pytest
from asciichem import ParseError, parse_text
from asciichem.text import render


@pytest.mark.parametrize("source", [
    "H_2O", "^14C", "Ca^2+", "Cl^-", "Ca^(II)", "(OH)_2", "[OH]_2", "H-O-H",
    "HC#CH", "2H_2 + O_2 -> 2H_2O", "C1-C-C-C-C-C1", '"water"', "`x^2 + 1`",
    "1s^2 2s^2 2p^6", "A ->[heat] B", "A <=> B", "A -> B -> C -> D",
    "H_2O @name(\"water\") @cas(\"7732-18-5\")", "(R)-CH_3CH(OH)COOH",
    "N_2 + 3H_2 <=>[Fe][500C] 2NH_3",
])
def test_round_trips(source):
    assert render(parse_text(source)) == source


def test_prefix_isotope_binds_to_atom():
    atom = parse_text("^14C").nodes[0].nodes[0]
    assert (atom.element, atom.isotope) == ("C", "14")


def test_rejects_bare_isotope_prefix():
    with pytest.raises(ParseError):
        parse_text("^14")


@pytest.mark.parametrize("source", ["", "12", "@name(", "A ->", "crystal[", "spectrum{"])
def test_rejects_malformed(source):
    with pytest.raises(ParseError):
        parse_text(source)


def test_hydrogen_bare_subscript_canonicalises():
    assert render(parse_text("H2O")) == "H_2O"


def test_sign_first_charge_canonicalises():
    assert render(parse_text("Ca^+2")) == "Ca^2+"


def test_two_letter_elements_stay_whole():
    assert parse_text("He").nodes[0].nodes[0].element == "He"


def test_wire_form_water():
    from asciichem.wire import to_wire
    assert to_wire(parse_text("H_2O")) == {
        "type": "formula",
        "nodes": [{"type": "molecule", "nodes": [
            {"type": "atom", "element": "H", "subscript": "2"},
            {"type": "atom", "element": "O"}]}]}


def test_wire_drops_fuzz_junk():
    from asciichem.text import render as r
    from asciichem.wire import to_model_json
    import json
    assert r(parse_text("H_{2a}O")) == "H_{2a}O"
    assert "2a" not in to_model_json(parse_text("H_{2a}O"))


def test_wire_ingestion_round_trip():
    from asciichem.wire import from_model_json, to_model_json
    for source in ["H_2O", "C1-C-C-C-C-C1", "2H_2 + O_2 -> 2H_2O", '"plain text"']:
        assert render(from_model_json(to_model_json(parse_text(source)))) == source
