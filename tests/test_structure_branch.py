"""The branch-projection port (asciichem-ruby#208's reference): the
SMILES reader shapes the linearizer, so smiles-ingested molecules text
as branch groups (no carbonyl ring digits) and round-trip through CML
with chirality intact — byte-identical with the ruby reference's text
for the same inputs."""
import pytest

from asciichem.cml import parse_cml, to_cml
from asciichem.smiles import parse_smiles, write_smiles
from asciichem.text import render

REFERENCE_TEXT = [
    ("CC(=O)O", "C-C(=O)-O"),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "C-C(=O)-O-C1=C-C=C-C=C1-C(=O)-O"),
    ("N[C@@H](C)C(=O)O", "N-C@TH2(-C)-C(=O)-O"),
    ("C1=CC=CC=C1", "C1=C-C=C-C=C1"),
]


@pytest.mark.parametrize("source,expected", REFERENCE_TEXT)
def test_texts_smiles_molecules_as_branch_groups(source, expected):
    assert render(parse_smiles(source)) == expected


def test_canonical_smiles_write_stays_stable_through_the_new_shape():
    for source, _ in REFERENCE_TEXT:
        written = write_smiles(parse_smiles(source))
        again = write_smiles(parse_smiles(written))
        assert again == written, source


@pytest.mark.parametrize("source,_", REFERENCE_TEXT)
def test_cml_round_trip_on_branch_grouped_models(source, _):
    formula = parse_smiles(source)
    back = parse_cml(to_cml(formula))
    assert render(back) == render(formula), source


def test_cml_carries_chirality():
    back = parse_cml(to_cml(parse_smiles("N[C@@H](C)C(=O)O")))
    chiral = [n for n in back.nodes[0].nodes if getattr(n, "chirality", None)]
    assert [n.chirality for n in chiral] == ["TH2"]
