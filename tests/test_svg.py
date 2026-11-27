"""SVG contract: byte parity with the TypeScript implementation's
structural and linear renderers (the committed reference snapshot was
generated from a built asciichem-ts dist), determinism across runs,
and graceful degradation (bond-less molecules render typographically,
unclosed rings annotate with `?`).
"""
import json
from pathlib import Path

import pytest

from asciichem import ParseError, parse_text
from asciichem.svg import render_structural_svg, render_svg

REFERENCE = Path(__file__).parent / "fixtures" / "svg-reference.jsonl"


def _cases():
    return [json.loads(line) for line in REFERENCE.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["input"])
def test_svg_matches_the_typescript_reference(case):
    node = parse_text(case["input"])
    assert render_structural_svg(node) == case["svg"], case["input"]
    assert render_svg(node) == case["linear"], case["input"]


@pytest.mark.parametrize(
    "source",
    ["C1-C-C-C-C-C1", "C1CCC(CCCCC1)", "C12-C-C1-C2", "A-B -> B"],
)
def test_svg_is_deterministic_across_runs(source):
    node = parse_text(source)
    assert render_structural_svg(node) == render_structural_svg(node)


def test_bondless_molecules_render_typographically():
    out = render_structural_svg(parse_text("H_2O"))
    assert "<line" not in out
    assert "<tspan" in out


def test_unclosed_rings_are_annotated_not_silent():
    out = render_structural_svg(parse_text("C1-C-C"))
    assert ">?</text>" in out


def test_structural_molecules_close_all_ring_bonds():
    out = render_structural_svg(parse_text("C1-C-C-C-C-C1"))
    assert out.count("<line") == 6


def test_invalid_input_raises_parse_error():
    with pytest.raises(ParseError):
        render_structural_svg(parse_text("Na+"))


def test_enantiomer_pair_renders_distinguishable_drawings():
    # The enantiomer-pair test (asciichem-rs#82): the @ parity draws
    # its ligand away (hash), the @@ parity toward (wedge) — opposite
    # out-of-plane directions, never two identical flat drawings.
    left = render_structural_svg(parse_text('@smiles("C[C@H](O)N")'))
    right = render_structural_svg(parse_text('@smiles("C[C@@H](O)N")'))
    assert left != right
    assert "<polygon" not in left
    assert 'stroke-width="1.2"' in left  # hash hatches
    assert "<polygon" in right  # wedge
    assert 'stroke-width="1.2"' not in right


def test_synthesis_declines_non_tetrahedral_ligand_sets():
    # C[C@H](O): two drawn neighbours + the stated hydrogen — three
    # ligands, no drawable tetrahedron, no invented glyph.
    out = render_structural_svg(parse_text('@smiles("C[C@H](O)")'))
    assert "<polygon" not in out
    assert 'stroke-width="1.2"' not in out


def test_molecules_without_parity_render_as_before():
    out = render_structural_svg(parse_text("C-C(-O)-N"))
    assert "<polygon" not in out
    assert 'stroke-width="1.2"' not in out
