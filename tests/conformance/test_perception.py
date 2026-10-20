"""Perception conformance (TODO.v2/13 item 2): implicit hydrogens and
aromatic rings are shared semantics fixed by the corpus
(fixtures/perception.json, generated from the Ruby reference). Skips
when the corpus is not cloned as a sibling."""
import json
import os
from pathlib import Path

import pytest

from asciichem.model import Molecule
from asciichem.perception import (
    aromatic_rings,
    hydrogen_valence_sum,
    implicit_hydrogens,
)
from asciichem.smiles import parse_smiles
from asciichem.structure import build_graph

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

PERCEPTION_FILE = CORPUS / "corpus" / "fixtures" / "perception.json"


def perception_cases():
    if not PERCEPTION_FILE.is_file():
        return []
    return json.loads(PERCEPTION_FILE.read_text())


@pytest.mark.skipif(not PERCEPTION_FILE.is_file(),
                    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")
def test_has_perception_cases():
    assert len(perception_cases()) >= 14


def perceived(smiles: str):
    molecule = parse_smiles(smiles).nodes[0]
    assert isinstance(molecule, Molecule)
    atoms, edges = build_graph(molecule)
    hydrogens = [implicit_hydrogens(atom, hydrogen_valence_sum(i, edges))
                 for i, atom in enumerate(atoms)]
    rings = sorted((sorted(ring) for ring in aromatic_rings(atoms, edges)))
    return hydrogens, rings


@pytest.mark.parametrize("case", perception_cases(), ids=lambda c: c["id"])
def test_perceives_shared_semantics(case):
    hydrogens, rings = perceived(case["smiles"])
    expected = case["perception"]
    assert hydrogens == expected["implicitHydrogens"]
    assert rings == expected["aromaticRings"]
