"""The CLI: the contract cases beyond the corpus's pinned ones - the
exit-code table (parse 1, format 2, resolve 3, cite 4, identity 5),
the per-command stdin/file/postures, `emit`'s puts semantics, usage
errors at 1, and `tree`'s reference shape. `run` is the seam: every
case dispatches in-process over captured streams."""
import io
import json

import pytest

import asciichem
from asciichem import cli


def invoke(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


# -- the exit-code table ---------------------------------------------------

@pytest.mark.parametrize("argv,prefix,code", [
    (["convert", "-i", "   "], "Parse error: ", 1),
    (["roundtrip", "-i", "("], "Parse error: ", 1),
    (["lint", "-i", "("], "Parse error: ", 1),
    (["validate", "-i", "("], "Parse error: ", 1),
    (["convert", "-i", "H", "-t", "wav"], "Format error: ", 2),
])
def test_parse_and_format_exit_codes(argv, prefix, code):
    got, _, err = invoke(*argv)
    assert got == code, err
    assert err.startswith(prefix)


def test_convert_without_input_names_the_flag_contract():
    _, _, err = invoke("convert")
    assert err == "Parse error: provide -i INPUT or -f FILE\n"


def test_unknown_format_names_the_formatter():
    _, _, err = invoke("convert", "-i", "H", "-t", "wav")
    assert err == "Format error: unknown formatter :wav\n"


# -- usage errors (the Thor posture) ---------------------------------------

def test_usage_errors_exit_1():
    code, _, err = invoke("roundtrip")            # -i is required
    assert code == 1
    assert "required" in err
    code, _, err = invoke("convert", "--from", "wat", "-i", "H_2O")
    assert code == 1


def test_bare_invocation_prints_help_with_exit_0():
    code, out, _ = invoke()
    assert code == 0
    assert "convert" in out


def test_version_flag_mirrors_the_subcommand():
    code, out, _ = invoke("--version")
    assert code == 0
    assert out == f"asciichem {asciichem.__version__}\n"
    code, out, _ = invoke("-v")
    assert out == f"asciichem {asciichem.__version__}\n"


# -- convert ----------------------------------------------------------------

def test_convert_reads_from_a_file(tmp_path):
    source = tmp_path / "input.chem"
    source.write_text("H_2O")
    code, out, _ = invoke("convert", "-f", str(source), "-t", "text")
    assert code == 0
    assert out == "H_2O\n"


def test_convert_default_format_is_mathml():
    code, out, _ = invoke("convert", "-i", "H_2O")
    assert code == 0
    assert out.startswith('<?xml version="1.0"?>') or "<math" in out


@pytest.mark.parametrize("fmt,marker", [
    ("text", "H_2O"),
    ("html", '<span class="asciichem">'),
    ("latex", "\\ce{"),
    ("svg", "<svg"),
    ("structural-svg", "<svg"),
    ("model-json", '"type": "formula"'),
    ("cml", "<cml "),
])
def test_convert_targets_emit_their_format(fmt, marker):
    code, out, _ = invoke("convert", "-i", "H_2O", "-t", fmt)
    assert code == 0, out
    assert marker in out


def test_convert_smiles_needs_a_structure():
    # H_2O is a formula, not a structure (no bonds) - the writer
    # rejects it through the parse family (exit 1).
    code, _, err = invoke("convert", "-i", "H_2O", "-t", "smiles")
    assert code == 1
    assert err.startswith("Parse error: ")
    code, out, _ = invoke("convert", "-i", "C-C-O", "-t", "smiles")
    assert code == 0
    assert out == "CCO\n"


def test_convert_from_smiles_land_in_the_same_model():
    code, out, _ = invoke("convert", "-i", "CCO", "--from", "smiles",
                          "-t", "text")
    assert code == 0
    assert out == "C-C-O\n"


# -- roundtrip --------------------------------------------------------------

def test_roundtrip_prints_the_canonical_form():
    code, out, _ = invoke("roundtrip", "-i", "H_2O")
    assert code == 0
    assert out == "H_2O\n"


def test_roundtrip_mismatch_reports_both_sides():
    _, _, err = invoke("roundtrip", "-i", "H2O")
    assert err.startswith("round-trip mismatch:\n")
    assert "input:    'H2O'" in err
    assert "rendered: 'H_2O'" in err


# -- lint -------------------------------------------------------------------

def test_lint_text_reports_and_exits_nonzero():
    code, out, _ = invoke("lint", "-i", "H_2 + O_2 -> H_2O")
    assert code == 1
    assert out == "[error] Reaction: Reaction is not balanced: O: 2 vs 1\n"


def test_lint_clean_prints_nothing():
    code, out, _ = invoke("lint", "-i", "H_2O")
    assert code == 0
    assert out == ""


def test_lint_json_payload():
    code, out, _ = invoke("lint", "-i", "H_2 + O_2 -> H_2O", "-f", "json")
    assert code == 1
    assert json.loads(out) == [{
        "severity": "error",
        "message": "Reaction is not balanced: O: 2 vs 1",
        "node": "Reaction",
    }]


# -- validate ---------------------------------------------------------------

def test_validate_status_column_and_exit():
    code, out, _ = invoke("validate", "-i",
                          'H_2O @cas("7732-18-5") @cas("50-7-8")')
    assert code == 1
    assert out == ("cas          7732-18-5"
                   "                                ok\n"
                   "cas          50-7-8"
                   "                                   INVALID\n")


def test_validate_unknown_convention_does_not_fail_the_run():
    code, out, _ = invoke("validate", "-i", 'H_2O @iupac("water")')
    assert code == 0
    assert "unknown convention" in out


def test_validate_without_annotations():
    code, out, _ = invoke("validate", "-i", "H_2O")
    assert code == 0
    assert out == "no identifier annotations found\n"


# -- parse-cml --------------------------------------------------------------

def test_parse_cml_emits_text():
    xml = ('<cml xmlns="http://www.xml-cml.org/schema"><molecule id="m1">'
           "<atomArray>"
           '<atom id="a1" elementType="H" count="2"/>'
           '<atom id="a2" elementType="O"/>'
           "</atomArray></molecule></cml>")
    code, out, _ = invoke("parse-cml", "-i", xml)
    assert code == 0
    assert out == "H_2O\n"


def test_parse_cml_alias_form():
    xml = ('<cml xmlns="http://www.xml-cml.org/schema">'
           "<molecule><atomArray><atom elementType=\"O\"/>"
           "</atomArray></molecule></cml>")
    code, out, _ = invoke("parse_cml", "-i", xml)
    assert code == 0
    assert out == "O\n"


def test_parse_cml_errors_exit_1():
    code, _, err = invoke("parse-cml", "-i", "<not-cml/>")
    assert code == 1
    assert err.startswith("CML parse error: ")


# -- version / tree ---------------------------------------------------------

def test_version_prints_the_package_version():
    code, out, _ = invoke("version")
    assert code == 0
    assert out == f"asciichem {asciichem.__version__}\n"


def test_tree_mirrors_the_reference_shape():
    code, out, _ = invoke("tree")
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "ascii_chem:cli"
    assert lines[1] == ("  ├─ cite (Resolve a substance and emit a "
                        "dataset-type Relaton bibitem (XML))")
    assert lines[2].startswith("  ├─ convert (Convert INPUT to FORMAT ")
    assert lines[-1] == ("  └─ version (Print the asciichem version)")
    # tree itself is not listed (Thor's posture)
    assert "  ├─ tree" not in out


# -- emit semantics ----------------------------------------------------------

def test_emit_adds_exactly_one_newline():
    out = io.StringIO()
    cli._emit(out, "H_2O")
    cli._emit(out, "ends-with\n")
    assert out.getvalue() == "H_2O\nends-with\n"


# -- identity (engine-scaled; the seam the corpus leaves to unit tests) ------

def test_identity_without_engine_exits_5():
    from asciichem import inchi
    previous = inchi.get_engine()
    inchi.set_engine(None)
    try:
        code, _, err = invoke("identity", "-i", "C-C-O")
        assert code == 5
        assert err.startswith("Identity error: ")
    finally:
        inchi.set_engine(previous)


def test_identity_rejects_engine_bin():
    code, _, err = invoke("identity", "-i", "C-C-O", "--engine-bin",
                          "/usr/bin/inchi-1")
    assert code == 5
    assert "--engine-bin" in err


def test_identity_needs_exactly_one_molecule():
    pytest.importorskip("rdkit")
    from asciichem.inchi import RdkitEngine, get_engine, set_engine
    previous = get_engine()
    set_engine(RdkitEngine())
    try:
        code, _, err = invoke("identity", "-i", "H_2 + O_2 -> H_2O")
        assert code == 5
        assert "exactly one molecule, got 3" in err
    finally:
        set_engine(previous)


def test_identity_derives_and_prints_the_inchi():
    pytest.importorskip("rdkit")
    from asciichem.inchi import RdkitEngine, get_engine, set_engine
    previous = get_engine()
    set_engine(RdkitEngine())
    try:
        code, out, _ = invoke("identity", "-i", "C-C-O")
        assert code == 0
        lines = out.splitlines()
        assert lines[0].startswith("InChI=1S/C2H6O/")
        assert len(lines[1]) == 27  # the InChIKey shape
    finally:
        set_engine(previous)
