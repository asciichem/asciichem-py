"""The Kekule text projection of aromatic rings (asciichem-ruby#188):
':' never re-parses in text, alternating '='/'-' does. Even rings and
fused systems project; odd rings keep the aromatic glyph (the spec'd
text loss). The model keeps its aromatic bonds - only the text view
projects."""
import pytest
from asciichem import parse
from asciichem.kekule import bond_kinds
from asciichem.model import Bond
from asciichem.smiles import parse_smiles
from asciichem.structure import build_graph
from asciichem.text import render


def molecule_for(smiles):
    return parse_smiles(smiles).nodes[0]


def double_count(text):
    molecule = parse(text).nodes[0]
    return sum(1 for n in molecule.nodes
               if isinstance(n, Bond) and n.kind == "double")


def test_assigns_alternating_kinds_closure_single():
    molecule = molecule_for("c1ccccc1")
    _atoms, edges = build_graph(molecule)

    kinds = bond_kinds(molecule)

    chain_kinds = [kinds[(min(e.from_, e.to), max(e.from_, e.to))]
                   for e in edges if max(e.from_, e.to) - min(e.from_, e.to) == 1]
    assert chain_kinds == ["double", "single", "double", "single", "double"]


def test_model_never_touched():
    molecule = molecule_for("c1ccccc1")
    bond_kinds(molecule)
    atoms, edges = build_graph(molecule)
    assert all(atom.aromatic for atom in atoms)
    assert all(edge.kind == "aromatic" for edge in edges)


def test_odd_rings_decline():
    assert bond_kinds(molecule_for("c1ccoc1")) == {}


def test_fused_systems_kekulize():
    molecule = molecule_for("c1ccc2ccccc2c1")
    kinds = bond_kinds(molecule)
    assert kinds
    assert "double" in kinds.values()
    assert "single" in kinds.values()


def test_no_aromatic_bonds_no_projection():
    assert bond_kinds(molecule_for("C1CCCCC1")) == {}


def test_benzene_reparseable_kekule_text():
    text = render(molecule_for("c1ccccc1"))
    assert text == "C1=C-C=C-C=C1"
    assert double_count(text) == 3
    assert render(parse(text)) == text


def test_pyridine_reparseable_kekule_text():
    text = render(molecule_for("c1ccncc1"))
    assert text == "C1=C-C=N-C=C1"
    assert double_count(text) == 3
    assert render(parse(text)) == text


def test_kekule_spelling_stable():
    assert render(parse("C1=CC=CC=C1")) == "C1=CC=CC=C1"


def test_odd_rings_keep_aromatic_glyph():
    assert render(molecule_for("c1ccoc1")) == "C1:C:C:O:C1"
