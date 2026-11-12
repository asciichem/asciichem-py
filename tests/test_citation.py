"""Unit coverage for the citation track (asciichem.citation) - mirrors
of the reference gem's citation specs: source profiles, the provenance
note, rejection of provenance-less substances, and the cite syntax
(@cite property annotations resolved through the resolver)."""
import json
import tempfile
from pathlib import Path

import pytest

from asciichem import citation, parse_text
from asciichem.model import AsciiChemError
from asciichem import resolver
from asciichem.resolver.substance import (Identifier, Provenance, Property,
                                          Substance)

PUBCHEM_ASPIRIN = {
    "PropertyTable": {"Properties": [{
        "CID": 2244,
        "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "MolecularFormula": "C9H8O4",
        "MolecularWeight": "180.16",
        "IUPACName": "2-acetyloxybenzoic acid",
        "InChI": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    }]}}


class RecordedFetch:
    def __init__(self, body):
        self.body = body if body is None else \
            (json.dumps(body) if isinstance(body, dict) else body)

    def get(self, url, headers=None):
        return self.body


@pytest.fixture(autouse=True)
def hermetic_key_chain(monkeypatch):
    from asciichem.resolver import common_chemistry

    monkeypatch.delenv("COMMON_CHEMISTRY_API_KEY", raising=False)
    monkeypatch.setattr(common_chemistry, "KEY_FILE",
                        Path("/nonexistent/cas-common-chemistry-api.key"))


def pubchem_aspirin(cache):
    return resolver.resolve(
        value="50-78-2", convention="cas", sources=["pubchem"],
        fetch=RecordedFetch(PUBCHEM_ASPIRIN), cache=cache,
        now="2026-01-05T09:30:00Z")


@pytest.fixture
def cache():
    return resolver.Cache(dir=Path(tempfile.mkdtemp(prefix="asciichem-cite-")))


def test_builds_a_dataset_bibitem_from_a_resolved_substance(cache):
    xml = citation.to_xml(pubchem_aspirin(cache))
    assert 'type="dataset"' in xml
    assert "2-acetyloxybenzoic acid" in xml
    assert "PubChem CID 2244" in xml
    assert "https://pubchem.ncbi.nlm.nih.gov/compound/2244" in xml
    assert '<date type="accessed">' in xml
    assert "<on>2026-01-05</on>" in xml
    assert 'id="PubChemCID2244"' in xml


def test_carries_every_identifier_as_keywords(cache):
    xml = citation.to_xml(pubchem_aspirin(cache))
    assert "inchikey=BSYNRYMUTXBXSQ-UHFFFAOYSA-N" in xml
    assert "canonical-smiles=CC(=O)OC1=CC=CC=C1C(=O)O" in xml


def test_provenance_note_survives_into_the_document(cache):
    xml = citation.to_xml(pubchem_aspirin(cache))
    assert '<note type="provenance">' in xml
    assert "Retrieved from pubchem" in xml
    assert "PubChem, U.S. National Library of Medicine" in xml


def test_source_version_in_the_note():
    provenance = Provenance(source="example",
                            retrieved_at="2026-11-10T00:00:00Z",
                            source_version="2026.11",
                            attribution="Example Data Source")
    substance = Substance(identifiers=[Identifier(value="50-78-2",
                                                  convention="cas",
                                                  provenance=provenance)])
    xml = citation.to_xml(substance)
    assert "Retrieved from example; version 2026.11; Example Data Source" in xml


def test_note_suppressed_when_it_would_say_nothing_beyond_the_source():
    provenance = Provenance(source="example",
                            retrieved_at="2026-11-10T00:00:00Z")
    substance = Substance(identifiers=[Identifier(value="50-78-2",
                                                  convention="cas",
                                                  provenance=provenance)])
    xml = citation.to_xml(substance)
    assert "<note" not in xml
    # The default profile names the docidentifier after the first
    # identifier and the publisher after the source itself.
    assert '<docidentifier type="example">cas: 50-78-2</docidentifier>' in xml
    assert "<name>example</name>" in xml
    assert 'id="cas50782"' in xml


def test_raises_for_substances_without_provenance():
    bare = Substance(identifiers=[Identifier(value="50-78-2",
                                             convention="cas")])
    with pytest.raises(AsciiChemError, match="no provenance"):
        citation.bibitem(bare)


def test_payload_fields_are_available_without_xml(cache):
    payload = citation.bibitem(pubchem_aspirin(cache))
    assert payload.bibitem_type == "dataset"
    assert payload.docidentifier_type == "pubchem"
    assert payload.publisher == "PubChem, U.S. National Library of Medicine"
    assert payload.accessed_on == "2026-01-05"
    assert payload.note.startswith("Retrieved from pubchem")


def test_derived_id_strips_non_word_characters():
    assert citation.derived_id("PubChem CID 2244") == "PubChemCID2244"
    assert citation.derived_id("CAS RN 50-78-2") == "CASRN50782"
    assert citation.derived_id("cas: 50-78-2") == "cas50782"
    assert citation.derived_id("///") is None


def test_for_molecule_resolves_one_bibitem_per_cited_source(cache):
    formula = parse_text('H_2O @name("water") @cite("pubchem")')
    molecule = formula.nodes[0]
    assert molecule.properties[-1] == {"title": "cite", "value": "pubchem"}
    pairs = citation.for_molecule(molecule, cache=cache,
                                  fetch=RecordedFetch(PUBCHEM_ASPIRIN))
    assert [source for source, _ in pairs] == ["pubchem"]
    assert 'type="dataset"' in pairs[0][1].title or pairs[0][1].bibitem_type \
        == "dataset"


def test_for_molecule_empty_without_cite_annotations(cache):
    molecule = parse_text('H_2O @name("water")').nodes[0]
    assert citation.for_molecule(molecule, cache=cache) == []


def test_for_molecule_respects_the_allow_list(cache):
    molecule = parse_text('H_2O @name("water") @cite("pubchem") '
                          '@cite("opsin")').nodes[0]
    pairs = citation.for_molecule(molecule, cache=cache,
                                  fetch=RecordedFetch(PUBCHEM_ASPIRIN),
                                  sources=["pubchem"])
    assert [source for source, _ in pairs] == ["pubchem"]


def test_for_molecule_requires_a_resolvable_identifier(cache):
    molecule = parse_text("H_2O @cite(\"pubchem\")").nodes[0]
    with pytest.raises(AsciiChemError, match="no resolvable identifier"):
        citation.for_molecule(molecule, cache=cache,
                              fetch=RecordedFetch(PUBCHEM_ASPIRIN))
