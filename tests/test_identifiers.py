"""Unit coverage for the offline identifier validators
(asciichem.identifiers) - lexical/checksum only, mirrors of the
reference gem's validator specs."""
import pytest

from asciichem import identifiers


@pytest.mark.parametrize("value", ["50-78-2", "74-82-8", "7732-18-5",
                                   "64-17-5", "8007-59-8"])
def test_cas_accepts_valid_checksums(value):
    assert identifiers.valid("cas", value)
    assert identifiers.diagnostic("cas", value) is None


def test_cas_rejects_bad_check_digit():
    message = identifiers.diagnostic("cas", "74-82-9")
    assert message is not None
    assert "check digit 9" in message
    assert "weighted sum" in message


@pytest.mark.parametrize("value", ["50782", "50-782-8", "5-07-82",
                                   "5007-82-86", "", "50-78-"])
def test_cas_rejects_malformed_shape(value):
    message = identifiers.diagnostic("cas", value)
    assert message is not None
    assert "not in DDDDDDD-DD-D form" in message


def test_cas_non_string_is_a_type_message():
    assert "must be a String" in identifiers.Cas.diagnostic(50782)  # type: ignore


def test_inchi_accepts_standard_prefix_and_formula_layer():
    value = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
    assert identifiers.valid("inchi", value)


def test_inchi_rejects_missing_prefix():
    message = identifiers.diagnostic("inchi", "garbage")
    assert message is not None
    assert 'must start with "InChI=1/" or "InChI=1S/"' in message


def test_inchi_rejects_empty_formula_layer():
    assert identifiers.diagnostic("inchi", "InChI=1S/") is not None


def test_inchi_rejects_unknown_element_in_formula_layer():
    assert "unknown element symbol 'Xx'" in \
        identifiers.diagnostic("inchi", "InChI=1S/Xx/c1-2")


def test_inchi_formula_counts_is_the_single_source():
    counts = identifiers.Inchi.formula_counts(
        "InChI=1S/H2O/h1H2")
    assert counts == {"H": 2, "O": 1}
    assert identifiers.Inchi.formula_counts("garbage") is None


def test_inchikey_format():
    assert identifiers.valid("inchikey", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert identifiers.valid("inchikey", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    message = identifiers.diagnostic("inchikey", "bsynrymutxbxsq-uhfffaoysa-n")
    assert message is not None and "27 uppercase letters" in message
    assert identifiers.diagnostic("inchikey", "BSYNRYMUTXBXSQ-UHFFFAOYSA") is not None


def test_smiles_sanity():
    assert identifiers.valid("smiles", "CC(=O)OC1=CC=CC=C1C(=O)O")
    assert identifiers.valid("smiles", "c1ccccc1")  # aromatic passes
    assert identifiers.valid("smiles", "[Na+]")
    assert identifiers.Smiles.element_set("CC(=O)Oc1ccccc1C(=O)O") == \
        ["C", "O"]


@pytest.mark.parametrize("value,fragment", [
    ("C(C", "parentheses are not balanced"),
    ("C[C", "square brackets are not balanced"),
    ("C1CC", "ring-bond digit 1"),
    ("C C", "must not contain whitespace"),
    ("", "must not be empty"),
    ("Cq", "malformed or unknown element token"),
])
def test_smiles_rejections(value, fragment):
    message = identifiers.diagnostic("smiles", value)
    assert message is not None and fragment in message


def test_unknown_conventions_are_not_checkable():
    assert not identifiers.known("iupac")
    assert identifiers.diagnostic("iupac", "anything") is None
    assert not identifiers.valid("cid", "2244")


def test_conventions_registered():
    assert sorted(identifiers.REGISTRY) == ["cas", "inchi", "inchikey",
                                            "smiles"]
