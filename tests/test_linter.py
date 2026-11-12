"""Unit coverage for the linter (asciichem.linter) - mirror of the
reference gem's linter specs: the check set, severity ordering,
label shapes, engine-scaled identity posture, and totality."""
import pytest

from asciichem import linter, parse_text
from asciichem.inchi import get_engine, set_engine
from asciichem.model import Crystal, Formula


def lint(source):
    return linter.run(parse_text(source))


def messages(source):
    return [(d.severity, d.label, d.message) for d in lint(source)]


@pytest.mark.parametrize("source", [
    "H_2O", "Ca(OH)_2 + 2HCl -> CaCl_2 + 2H_2O",
    "Ca^2+ + 2Cl^- -> CaCl_2", "C1-C-C-C-C-C1", "^2H", "H-C=C-H",
])
def test_clean_inputs_stay_silent(source):
    assert messages(source) == []


def test_balance_reports_element_diffs():
    diagnostics = lint("H_2 + O_2 -> H_2O")
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.severity == "error"
    assert diagnostic.label == "Reaction"
    assert diagnostic.message == "Reaction is not balanced: O: 2 vs 1"
    assert diagnostic.details == {"O": [2, 1]}


def test_charge_balance_messages():
    assert "reactants 0 vs products +1" in \
        lint("H^+ + OH^- -> H_2O^+")[0].message
    assert "reactants 0 vs products -1" in \
        lint("Na^+ + Cl^- -> NaCl^-")[0].message
    assert "reactants +1 vs products 0" in \
        lint("Ca^2+ + Cl^- -> CaCl")[0].message


def test_group_multiplicity_applies():
    # Ca(OH)_2: the group's multiplicity feeds both counters.
    assert messages("Ca(OH)_2") == []
    diagnostics = lint("H_2 + H_2O -> Ca(OH)_2")
    assert "Ca: 0 vs 1" in diagnostics[0].message
    assert "H: 4 vs 2" in diagnostics[0].message
    assert "O: 1 vs 2" in diagnostics[0].message


def test_element_validation_warns_with_labels():
    diagnostics = lint("Cy-H")
    assert [(d.severity, d.label) for d in diagnostics] == [
        ("warning", "Atom(Cy)"), ("info", "Atom(Cy)")]
    assert "Unknown element symbol 'Cy'" in diagnostics[0].message
    assert "not in valence table" in diagnostics[1].message


def test_isotope_sanity():
    diagnostics = lint("^5C")
    assert diagnostics[0].severity == "error"
    assert "Isotope mass 5 is less than atomic number 6 for C" in \
        diagnostics[0].message
    assert lint("^2H") == []


def test_unclosed_ring():
    diagnostics = lint("C-C-C1")
    assert diagnostics[0].severity == "error"
    assert diagnostics[0].label == "Atom(C)"
    assert "unmatched ring closure" in diagnostics[0].message


def test_valence_overload_is_an_error():
    # The graph walk counts exactly the bonds that are written
    # (the reference's tolerant build): a linear H chain overloads
    # hydrogen's valence of 1.
    diagnostics = lint("H-H-H")
    overloaded = [d for d in diagnostics if d.severity == "error"]
    assert any("Atom H has bond order 2; max valence is 1" in d.message
               for d in overloaded)


def test_identifier_format_check_digit():
    diagnostics = lint('C @cas("74-82-9")')
    assert diagnostics[0].severity == "error"
    assert "check digit 9" in diagnostics[0].message
    assert "cas identifier on C" in diagnostics[0].message


def test_identifier_format_skips_uncheckable_conventions():
    assert lint('C @iupac("methane")') == []


def test_identifier_consistency_inchi_mismatch():
    diagnostics = lint("H_2O @inchi(\"InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-"
                       "7(8)9(11)12/h2-5H,1H3,(H,11,12)\")")
    assert any(d.severity == "error" and "does not match molecule" in d.message
               for d in diagnostics)


def test_identifier_consistency_smiles_set_membership():
    unexpected = lint('H_2O @smiles("CCO")')
    assert unexpected[0].severity == "warning"
    assert 'unexpected ["C"]' in unexpected[0].message
    missing = lint('C_2H_6O @smiles("CC")')
    assert missing[0].severity == "warning"
    assert 'missing ["O"]' in missing[0].message
    assert lint('C_2H_6O @smiles("CCO")') == []
    # Aromatic SMILES pass element-set membership.
    assert lint('C_9H_8O_4 @smiles("CC(=O)Oc1ccccc1C(=O)O")') == []


def test_crystal_sanity_messages():
    diagnostics = lint("crystal[x](a=-1){Na@f(0,0,0)}")
    assert diagnostics[0].severity == "error"
    assert diagnostics[0].label == "Crystal(x)"
    assert diagnostics[0].message == "Crystal a must be positive, got -1"
    assert "must be in (0, 180]" in lint(
        "crystal[x](a=1,alpha=200){Na@f(0,0,0)}")[0].message
    warnings = lint("crystal[x](a=1){Na@f(1.5,0,0)}")
    assert warnings[0].severity == "warning"
    assert "outside [0, 1)" in warnings[0].message
    assert lint("crystal[x](a=1){Na@f(0,0,0)}") == []


def test_spectrum_peaks():
    from asciichem.model import Spectrum, SpectrumPeak

    formula = Formula(nodes=[Spectrum(
        technique="nmr",
        peaks=[SpectrumPeak(position="abc", intensity=None),
               SpectrumPeak(position="-3.5", intensity="-2H")])])
    diagnostics = linter.run(formula)
    assert any("Spectrum peak position 'abc' is not numeric" in d.message
               for d in diagnostics)
    assert any("Spectrum peak position -3.5 is negative" in d.message
               for d in diagnostics)
    assert any("Spectrum peak intensity -2 is negative" in d.message
               for d in diagnostics)


def test_zmatrix_references():
    from asciichem.model import ZMatrix, ZRow

    formula = Formula(nodes=[ZMatrix(rows=[
        ZRow(atom="C"),
        ZRow(atom="O", ref1="N", distance="1.4"),
        ZRow(atom="N", ref1="C", ref2="O", distance="1.1",
             angle="90", dihedral="0"),
    ])])
    diagnostics = linter.run(formula)
    assert any("references \"N\" before it is defined" in d.message
               for d in diagnostics)


def test_diagnostics_sort_errors_first_then_message():
    diagnostics = lint("Hx_2O")
    severities = [d.severity for d in diagnostics]
    assert severities == sorted(severities, key=linter.SEVERITY_ORDER.get)
    assert severities[0] == "warning" and severities[-1] == "info"


def test_diagnostic_rendering():
    diagnostics = lint("H^+ + OH^- -> H_2O^+")
    assert str(diagnostics[0]) == (
        "[error] Reaction: Reaction charge is not balanced: "
        "reactants 0 vs products +1")


def test_registry_names_mirror_the_reference():
    assert list(linter.REGISTRY) == [
        "balance", "bracket_balance", "charge_balance", "crystal_sanity",
        "element_validation", "identifier_consistency", "identifier_format",
        "identity_cross_check", "isotope_sanity", "spectrum_peaks",
        "unclosed_ring", "valence", "zmatrix_references"]


def test_composed_registry_runs_a_subset():
    registry = {"element_validation": linter.element_validation}
    formula = parse_text("H_2 + O_2 -> H_2O")
    assert linter.run(formula, registry=registry) == []
    assert linter.run(formula) != []


# -- identity cross-check (engine-scaled) --------------------------------

@pytest.fixture
def no_engine():
    previous = get_engine()
    set_engine(None)
    yield
    set_engine(previous)


def test_identity_check_warns_without_engine(no_engine):
    diagnostics = lint('C_9H_8O_4 @inchi("InChI=1S/C9H8O4/c1-6(10)13-8-5-'
                       '3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)")')
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == "warning"
    assert "no InChI engine configured" in diagnostics[0].message


def test_identity_check_silent_for_unannotated_molecules(no_engine):
    assert lint("H_2O") == []


def test_identity_check_engine_missing_is_total(no_engine):
    # A molecule with identity annotations and no engine warns once,
    # whatever the annotation count.
    diagnostics = lint('H_2O @inchi("InChI=1S/H2O/h1H2") '
                       'H_2O_2 @inchi("InChI=1S/H2O2/c1-2/h1-2H")')
    assert len(diagnostics) == 1
    assert "2 annotated molecule(s)" in diagnostics[0].message


def test_identity_check_passes_with_engine(no_engine):
    pytest.importorskip("rdkit")
    from asciichem.inchi import RdkitEngine

    set_engine(RdkitEngine())
    try:
        assert lint('H-O-H @inchi("InChI=1S/H2O/h1H2")') == []
    finally:
        set_engine(None)


def test_identity_check_reports_drift_with_engine(no_engine):
    pytest.importorskip("rdkit")
    from asciichem.inchi import RdkitEngine

    set_engine(RdkitEngine())
    try:
        diagnostics = lint('H-O-H @inchi("InChI=1S/C9H8O4/c1-6(10)13-8-'
                           '5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)")')
        assert any("does not match molecule" in d.message for d in diagnostics)
        assert any("does not match the drawn structure" in d.message
                   for d in diagnostics)
    finally:
        set_engine(None)


def test_identity_check_unanalysable_structure_warns(no_engine):
    pytest.importorskip("rdkit")
    from asciichem.inchi import RdkitEngine

    set_engine(RdkitEngine())
    try:
        # H_2O carries no bonds - the engine cannot derive a
        # structure InChI; the check reports, never raises.
        diagnostics = lint('H_2O @inchi("InChI=1S/C9H8O4/c1-6(10)13-8-'
                           '5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)")')
        assert any("cannot derive InChI" in d.message for d in diagnostics)
    finally:
        set_engine(None)


def test_errors_predicate():
    assert linter.errors(parse_text("^5C"))
    assert not linter.errors(parse_text("H_2O"))


def test_linter_is_total_over_every_model_node():
    # Direct model construction (no parser) still runs every check.
    formula = Formula(nodes=[Crystal(name="bare", a=-2.0)])
    diagnostics = linter.run(formula)
    assert diagnostics[0].label == "Crystal(bare)"
    assert "must be positive, got -2" in diagnostics[0].message
