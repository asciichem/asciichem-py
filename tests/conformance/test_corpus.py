"""Shared-corpus conformance: runs the asciichem-tests fixtures against
this implementation. Claimed levels: parse/reject, L0 (emission
schema-validated), L1 (Text round-trip) at 100%."""
import json
import os
from pathlib import Path

import pytest

from asciichem import ParseError, parse_text
from asciichem.svg import render_structural_svg
from asciichem.text import render
from asciichem.wire import to_wire

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))
SCHEMAS = Path(os.environ.get("ASCIICHEM_MODEL",
                              Path.cwd().parent / "asciichem-model")) / "schemas" / "v1"

pytestmark = pytest.mark.skipif(
    not (CORPUS / "corpus" / "fixtures").is_dir(),
    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")


def fixtures():
    cases = []
    directory = CORPUS / "corpus" / "fixtures"
    for path in sorted(directory.glob("*.json")):
        cases.extend(json.loads(path.read_text()))
    return cases


def input_fixtures():
    return [c for c in fixtures() if isinstance(c.get("input"), str)]


def _validator():
    import yaml
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schemas = {}
    id_by_file = {}
    for path in sorted(SCHEMAS.glob("*.yaml")):
        schema = yaml.safe_load(path.read_text())
        schemas[path.stem] = schema
        id_by_file[path.stem] = schema["$id"]

    def rewrite(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                file, _, pointer = ref.partition("#")
                target = id_by_file.get(file.removesuffix(".yaml"))
                if target:
                    node["$ref"] = f"{target}#/{pointer}" if pointer else target
            for value in node.values():
                rewrite(value)
        elif isinstance(node, list):
            for item in node:
                rewrite(item)

    for schema in schemas.values():
        rewrite(schema)
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema, default_specification=DRAFT202012))
        for schema in schemas.values())
    return Draft202012Validator(schemas["formula"], registry=registry)


_validator_cache = {}


@pytest.fixture(scope="module")
def validator():
    if "v" not in _validator_cache:
        _validator_cache["v"] = _validator()
    return _validator_cache["v"]


def test_corpus_loads():
    assert len(input_fixtures()) > 200


@pytest.mark.parametrize("case", input_fixtures(), ids=lambda c: c["id"])
def test_parse_reject(case):
    if case["parses"]:
        parse_text(case["input"])  # must not raise
    else:
        with pytest.raises(ParseError):
            parse_text(case["input"])


@pytest.mark.parametrize("case", [c for c in input_fixtures() if c["parses"]],
                         ids=lambda c: c["id"])
def test_l0_schema_valid(case, validator):
    errors = sorted(validator.iter_errors(to_wire(parse_text(case["input"]))),
                    key=lambda e: e.json_path)
    assert not errors, f'{case["id"]}: ' + "; ".join(
        f"{e.json_path} {e.message[:80]}" for e in errors[:3])


@pytest.mark.parametrize("case", [c for c in input_fixtures() if c.get("roundTrip")],
                         ids=lambda c: c["id"])
def test_l1_round_trip(case):
    assert render(parse_text(case["input"])) == case["input"]


# -- structure interchange (TODO.v2 09): opt-in levels ----------------------

from asciichem.molfile import parse_molfile, write_molfile
from asciichem.smiles import parse_smiles, write_smiles
from asciichem.structure import build_graph


def smiles_fixtures():
    return [c for c in fixtures() if isinstance(c.get("smiles"), str)]


@pytest.mark.parametrize("case", smiles_fixtures(), ids=lambda c: c["id"])
def test_smiles_ingestion(case):
    if case["parses"]:
        parse_smiles(case["smiles"])  # must not raise
        if case.get("smilesRoundTrip"):
            assert write_smiles(parse_smiles(case["smiles"])) == case["smiles"]
        if isinstance(case.get("canonicalSmiles"), str):
            # Canonical aromatic form for inputs whose spelling is not
            # the fixed point (Kekule rings).
            assert (write_smiles(parse_smiles(case["smiles"]))
                    == case["canonicalSmiles"])
    else:
        with pytest.raises(ParseError):
            parse_smiles(case["smiles"])


# The recorded identity intersection (asciichem-tests#29): an
# identity-capable engine derives exactly the recorded InChI.
# Engine-gated like the other runtimes' runners — a missing engine is
# a posture, not a failure. The recorded intersection excludes the
# cases the molfile transport cannot express (the [nH] class), so the
# pipe agrees on every recorded case by construction.
@pytest.mark.parametrize("case", [c for c in smiles_fixtures()
                                  if isinstance(c.get("inchi"), str)],
                         ids=lambda c: c["id"])
def test_smiles_recorded_identity(case):
    from asciichem.inchi import EngineMissingError, identity_for

    molecule = parse_smiles(case["smiles"]).nodes[0]
    try:
        identity = identity_for(molecule)
    except EngineMissingError:
        pytest.skip("no identity engine configured")
    assert identity.inchi == case["inchi"], case["id"]


# The structure-reference family's recorded text column
# (asciichem-tests#31): where the canonical text was recorded, the
# runtime produces exactly that string (cross-runtime text parity of
# the ingest edge).
@pytest.mark.parametrize("case", [c for c in fixtures()
                                  if isinstance(c.get("input"), str)
                                  and isinstance(c.get("text"), str)],
                         ids=lambda c: c["id"])
def test_structure_reference_text(case):
    assert render(parse_text(case["input"])) == case["text"], case["id"]


# The smiles family's recorded text column (the branch-projection
# text, asciichem-ruby#208's recording): the smiles-ingested model's
# canonical text matches the cross-runtime recording byte-for-byte.
@pytest.mark.parametrize("case", [c for c in fixtures()
                                  if isinstance(c.get("smiles"), str)
                                  and isinstance(c.get("text"), str)
                                  and c.get("parses")],
                         ids=lambda c: c["id"])
def test_smiles_recorded_text(case):
    assert render(parse_smiles(case["smiles"])) == case["text"], case["id"]


def molfile_fixtures():
    return [c for c in fixtures() if isinstance(c.get("molfile"), str)]


@pytest.mark.parametrize("case", molfile_fixtures(), ids=lambda c: c["id"])
def test_molfile_ingestion(case):
    if case["parses"]:
        molecule = parse_molfile(case["molfile"])
        atoms, edges = build_graph(molecule)
        assert len(atoms) == case["atoms"], case["id"]
        assert len(edges) == case["bonds"], case["id"]
        if case.get("molfileRoundTrip"):
            shape = sorted((e.from_, e.to, e.kind) for e in edges)
            again_atoms, again_edges = build_graph(
                parse_molfile(write_molfile(molecule)))
            assert len(again_atoms) == len(atoms), case["id"]
            assert sorted((e.from_, e.to, e.kind)
                          for e in again_edges) == shape, case["id"]
    else:
        with pytest.raises(ParseError):
            parse_molfile(case["molfile"])


# L2: MathML golden parity - exact-string comparison against the
# reference implementation's output (single-contract rule).
def mathml_fixtures():
    return [
        c
        for c in fixtures()
        if isinstance(c.get("mathml"), str) and not c.get("referenceRender")
    ]


@pytest.mark.parametrize("case", mathml_fixtures(), ids=lambda c: c["id"])
def test_l2_mathml_golden(case):
    from asciichem.mathml import render_mathml
    assert render_mathml(parse_text(case["input"])) == case["mathml"], case["id"]


def _render_golden_fixtures(field):
    # referenceRender rows embed the reference's math-engine output
    # (Plurimath); this implementation degrades embedded math by
    # contract (corpus schema: "the skip is the documented
    # degradation, not a silent drop").
    return [
        c
        for c in fixtures()
        if isinstance(c.get(field), str) and not c.get("referenceRender")
    ]


@pytest.mark.parametrize("case", _render_golden_fixtures("html"), ids=lambda c: c["id"])
def test_l2_html_golden(case):
    from asciichem.html import render_html
    assert render_html(parse_text(case["input"])) == case["html"], case["id"]


@pytest.mark.parametrize("case", _render_golden_fixtures("latex"), ids=lambda c: c["id"])
def test_l2_latex_golden(case):
    from asciichem.latex import render_latex
    assert render_latex(parse_text(case["input"])) == case["latex"], case["id"]


# L3: CML round-trip - parse -> CML wire -> parse -> canonical text
# equals the input (the three-way contract; the wire carries the
# aci: extension channels for what plain CML does not cover).
def cml_fixtures():
    return [c for c in fixtures() if c.get("cmlRoundTrip")]


@pytest.mark.parametrize("case", cml_fixtures(), ids=lambda c: c["id"])
def test_l3_cml_round_trip(case):
    from asciichem import cml
    xml = cml.to_cml(parse_text(case["input"]))
    restored = cml.parse_cml(xml)
    # An alias-spelling input lands on the canonical text (the
    # fixture's `text` field records it; defaults to the input).
    expected = case.get("text") or case["input"]
    assert render(restored) == expected, case["id"]


# The depiction family's golden bytes (asciichem-rs#82): every row's
# structural SVG re-renders byte-exactly — the enantiomer-pair rows are
# the wedge-synthesis visual acceptance.
def depiction_fixtures():
    return [c for c in fixtures() if isinstance(c.get("svg"), str)]


@pytest.mark.parametrize("case", depiction_fixtures(), ids=lambda c: c["id"])
def test_depiction_golden_renders(case):
    assert render_structural_svg(parse_text(case["input"])) == case["svg"]
