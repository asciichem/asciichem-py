import pytest
from asciichem.model import ParseError
from asciichem.smiles import parse_smiles, write_smiles
from asciichem.structure import build_graph


def test_chain_parses_into_bonded_molecule():
    molecule = parse_smiles("CCO").nodes[0]
    atoms, edges = build_graph(molecule)
    assert [a.element for a in atoms] == ["C", "C", "O"]
    assert [e.kind for e in edges] == ["single", "single"]


def test_aromatic_rings_marked():
    atoms, edges = build_graph(parse_smiles("c1ccccc1").nodes[0])
    assert all(a.aromatic for a in atoms)
    assert all(e.kind == "aromatic" for e in edges)
    assert len(edges) == 6


def test_bracket_atoms():
    atom = parse_smiles("[13CH4]").nodes[0].nodes[0]
    assert (atom.element, atom.isotope, atom.hydrogens) == ("C", "13", 4)
    assert write_smiles(parse_smiles("[Fe++]")) == "[Fe+2]"


# Stereo moved from the v1 rejections to supported features
# (corpus 0.5.0): @/@@ parse into Atom.chirality (TH1/TH2) and
# / \ into Bond.direction.
@pytest.mark.parametrize("bad", ["C1CCC", "C=1CCCCC=1", "CC!X", "[C"])
def test_rejects_v1_subset(bad):
    with pytest.raises(ParseError):
        parse_smiles(bad)


@pytest.mark.parametrize("smiles", [
    "CC(=O)OC1=CC=CC=C1C(=O)O", "c1ccc2ccccc2c1", "CC(C)(C)C",
    # Ring-closure digits stay out of branches (#103): a
    # closure-bearing child continues the chain.
    "C1=CCC(C)C(C)C1", "C1CC2CCC1C2", "C1CC(C)CCC1"])
def test_writer_is_stable_and_structure_preserving(smiles):
    # Canonicalization maps many models to one spelling (Kekule ->
    # aromatic), so model equality is not the invariant — composition
    # and connectivity must survive and the form must be a fixed
    # point.
    once = write_smiles(parse_smiles(smiles))
    assert write_smiles(parse_smiles(once)) == once
    before = build_graph(parse_smiles(smiles).nodes[0])
    after = build_graph(parse_smiles(once).nodes[0])
    assert len(after[0]) == len(before[0])
    assert len(after[1]) == len(before[1])


@pytest.mark.parametrize("kekule, canonical", [
    ("C1=CC=CC=C1", "c1ccccc1"),
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ClC1=CC=CC=C1", "Clc1ccccc1"),
    ("C1=CC=NC=C1", "c1ccncc1"),
    ("C1=CC=C2C=CC=CC2=C1", "c1ccc2ccccc2c1"),
    # Hückel five-membered rings (corpus 0.8.0).
    ("C1=COC=C1", "c1cocc1"),
    ("C1=CNC=C1", "c1c[nH]cc1"),
    ("C1=CSC=C1", "c1cscc1"),
])
def test_canonicalizes_perceived_aromatic_rings(kekule, canonical):
    assert write_smiles(parse_smiles(kekule)) == canonical
    assert write_smiles(parse_smiles(canonical)) == canonical


@pytest.mark.parametrize("smiles", ["O=C1C=CC=CC1", "O=C1C=CC(=O)C=C1",
                                   "C1=CCC=C1", "C1=CC1"])
def test_keeps_non_aromatic_rings_kekule(smiles):
    assert write_smiles(parse_smiles(smiles)) == smiles


def atom_line(x, y, z, sym):
    return f"{x:10.4f}{y:10.4f}{z:10.4f} {sym:<3} 0  0  0  0  0  0  0  0  0  0  0  0"


def bond_line(a, b, t, s=0):
    return f"{a:3d}{b:3d}{t:3d}{s:3d}  0  0  0  0  0  0  0"


BENZENE = "\n".join([
    "benzene", "  f", "", "  6  6  0  0  0  0  0  0  0  0999 V2000",
    *[atom_line(x, y, 0, "C") for x, y in
      [(-0.5, 0.866), (0.5, 0.866), (1.0, 0), (0.5, -0.866), (-0.5, -0.866), (-1.0, 0)]],
    *[bond_line(a, b, 4) for a, b in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]],
    "M  END"]) + "\n"


def test_molfile_aromatic_round_trip():
    from asciichem.molfile import parse_molfile, write_molfile
    molecule = parse_molfile(BENZENE)
    atoms, edges = build_graph(molecule)
    assert all(a.aromatic for a in atoms)
    shape = sorted((e.from_, e.to, e.kind) for e in edges)
    again_atoms, again_edges = build_graph(parse_molfile(write_molfile(molecule)))
    assert sorted((e.from_, e.to, e.kind) for e in again_edges) == shape
    assert len(again_atoms) == len(atoms)


def test_molfile_rejects_malformed_counts():
    from asciichem.molfile import parse_molfile
    with pytest.raises(ParseError):
        parse_molfile("bad\n  x\n\nzzz  0  0  0  0  0  0  0  0999 V2000\nM  END\n")
