"""The CML layer: the three-way round-trip contract on constructs
beyond the corpus's cml-roundtrip family (rings, cascade arrows,
isotopes with charges), the aci: channel contract (namespace-URI
classification, not written prefixes), the
declarations-only-when-needed posture, determinism, and the
report-don't-drop rule for constructs with no wire representation."""
import pytest

from asciichem import parse_text
from asciichem.cml import (ACI_NAMESPACE, CML_NAMESPACE, CmlError,
                           parse_cml, to_cml)
from asciichem.model import Group
from asciichem.text import render as render_text


def round_trips(source: str) -> None:
    xml = to_cml(parse_text(source))
    assert render_text(parse_cml(xml)) == source, source


# -- the round-trip contract, beyond the corpus family ---------------------

@pytest.mark.parametrize("source", [
    "C1-C-C-C-C-C1",            # ring closure -> canonical ring bond
    "A -> B -> C",              # cascade
    "A <=> B",                  # equilibrium arrow survives the wire
    "A <- B",                   # reverse arrow
    "^14Ca^2+",                 # isotope and charge on one molecule
    "2Ca(OH)_2",                # molecule coefficient over a group
    "C-C-O",                    # positional bonds
])
def test_round_trips(source):
    round_trips(source)


def test_emit_is_deterministic():
    assert to_cml(parse_text("Ca(OH)_2")) == to_cml(parse_text("Ca(OH)_2"))


# -- the wire shapes ---------------------------------------------------------

def test_plain_molecule_declares_only_the_core_namespace():
    xml = to_cml(parse_text("H_2O"))
    assert f'xmlns="{CML_NAMESPACE}"' in xml
    assert "aci" not in xml


def test_extensions_declare_the_aci_namespace():
    xml = to_cml(parse_text("::O"))
    assert f'xmlns:aci="{ACI_NAMESPACE}"' in xml
    assert 'aci:lonePairs="2"' in xml


def test_atom_attribute_channel_covers_the_frozen_registry():
    xml = to_cml(parse_text("Fe^(II)"))
    assert 'aci:oxidationState="II"' in xml
    xml = to_cml(parse_text("N."))
    assert 'aci:radicalElectrons="1"' in xml


def test_group_channel_preserves_bracket_and_multiplicity():
    xml = to_cml(parse_text("[OH]_2"))
    assert '<aci:group multiplicity="2" bracket="square" atomRefs="a1 a2"/>' in xml


def test_top_level_channel_preserves_position():
    xml = to_cml(parse_text('`x^2` "note"'))
    assert '<aci:embeddedMath position="0">x^2</aci:embeddedMath>' in xml
    assert '<aci:text position="1">"note"</aci:text>' in xml


def test_reaction_wire_uses_reactant_and_product_lists():
    xml = to_cml(parse_text("2H_2 + O_2 -> 2H_2O"))
    assert '<reaction id="r1" title="forward" type="forward">' in xml
    assert '<substance role="reactant">' in xml
    assert '<molecule id="m1" count="2">' in xml


def test_bond_orders_map_to_cml_codes():
    xml = to_cml(parse_text("C=C"))
    assert 'order="D"' in xml
    xml = to_cml(parse_text("C#C"))
    assert 'order="T"' in xml


def test_ring_closures_yield_a_canonical_ring_bond():
    xml = to_cml(parse_text("C1-C-C-C-C-C1"))
    assert xml.count("<bond ") == 6


# -- read side: namespace-URI classification --------------------------------

def test_read_classifies_channels_by_namespace_uri_not_prefix():
    # The extension prefix is written `x:` here, bound to the aci
    # namespace URI - the channel must still classify.
    xml = ('<cml xmlns="http://www.xml-cml.org/schema" '
           'xmlns:x="https://asciichem.org/cml-ext">'
           "<molecule><atomArray>"
           '<atom elementType="O" x:lonePairs="2"/>'
           "</atomArray></molecule></cml>")
    assert render_text(parse_cml(xml)) == "::O"


def test_read_ignores_foreign_namespaces():
    xml = ('<cml xmlns="http://www.xml-cml.org/schema" '
           'xmlns:other="https://example.com/other">'
           "<molecule><atomArray>"
           '<atom elementType="O" other:noise="1"/>'
           "</atomArray></molecule>"
           "<other:noise>ignored</other:noise></cml>")
    assert render_text(parse_cml(xml)) == "O"


def test_read_tolerates_undeclared_cml():
    xml = ('<cml><molecule id="m1"><atomArray>'
           '<atom id="a1" elementType="H" count="2"/>'
           '<atom id="a2" elementType="O"/>'
           "</atomArray></molecule></cml>")
    assert render_text(parse_cml(xml)) == "H_2O"


def test_read_requires_a_cml_root():
    with pytest.raises(CmlError):
        parse_cml("<not-cml/>")


def test_read_rejects_malformed_xml():
    with pytest.raises(CmlError):
        parse_cml("<cml><molecule>")


def test_read_rejects_atoms_without_element_type():
    xml = ('<cml xmlns="http://www.xml-cml.org/schema">'
           "<molecule><atomArray><atom id=\"a1\"/></atomArray>"
           "</molecule></cml>")
    with pytest.raises(CmlError):
        parse_cml(xml)


# -- report, never drop ------------------------------------------------------

@pytest.mark.parametrize("source", [
    "crystal[a=1]{Na}",       # no wire, no channel
    "spectrum{}",             # no wire, no channel
    "A ->[heat] B",           # the conditions channel is not in this slice
])
def test_unsupported_constructs_raise_rather_than_drop(source):
    with pytest.raises(CmlError):
        to_cml(parse_text(source))


def test_bare_atoms_are_rejected_outside_a_molecule():
    from asciichem.model import Atom, Formula
    with pytest.raises(CmlError):
        to_cml(Formula(nodes=[Atom("O")]))


def test_read_splices_group_multiplicity_back():
    xml = to_cml(parse_text("Ca(OH)_2"))
    molecule = parse_cml(xml).nodes[0]
    group = molecule.nodes[1]
    assert isinstance(group, Group)
    assert group.multiplicity == "2"
    assert [atom.element for atom in group.nodes] == ["O", "H"]
    assert all(atom.subscript is None for atom in group.nodes)
