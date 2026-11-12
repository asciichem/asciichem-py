"""Unit coverage for the resolver (asciichem.resolver) - mirrors of
the reference gem's resolver specs: registry fan-out, first-answer
wins, per-source failure isolation, cross-check conflict, TTL cache
offline service, the canonical wire substance-record round trip, and
the Common Chemistry key chain (hermetic: no real key enters these
specs)."""
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from asciichem.model import AsciiChemError
from asciichem import resolver
from asciichem.resolver import common_chemistry
from asciichem.resolver.substance import Identifier, Provenance, Substance
from asciichem.wire import substance_from_wire, substance_to_wire

PUBCHEM_ASPIRIN = {
    "PropertyTable": {"Properties": [{
        "CID": 2244,
        "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "IsomericSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "MolecularFormula": "C9H8O4",
        "MolecularWeight": "180.16",
        "IUPACName": "2-acetyloxybenzoic acid",
        "InChI": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    }]}}

CC_ASPIRIN = {
    "rn": "50-78-2", "name": "Aspirin", "formula": "C9H8O4",
    "molecularMass": 180.16,
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
    "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    "iupacName": "2-acetyloxybenzoic acid",
    "experimentalProperties": [],
}


class RecordedFetch:
    """Answers every request with the recorded body (the value fake,
    never a double - real injected transport)."""

    def __init__(self, body):
        self.body = body if body is None else \
            (json.dumps(body) if isinstance(body, dict) else body)
        self.calls = []

    def get(self, url, headers=None):
        self.calls.append((url, dict(headers or {})))
        return self.body


class OfflineFetch:
    @staticmethod
    def get(url, headers=None):
        raise AsciiChemError(f"network must not be touched: {url}")


@pytest.fixture(autouse=True)
def hermetic_key_chain(monkeypatch):
    monkeypatch.delenv("COMMON_CHEMISTRY_API_KEY", raising=False)
    monkeypatch.setattr(common_chemistry, "KEY_FILE",
                        Path("/nonexistent/cas-common-chemistry-api.key"))


@pytest.fixture
def cache():
    return resolver.Cache(dir=Path(tempfile.mkdtemp(prefix="asciichem-resolver-")))


PINNED = "2026-01-05T09:30:00Z"


def test_registry_carries_the_shipped_sources_in_order():
    assert list(resolver.adapters) == ["pubchem", "opsin", "common_chemistry"]


def test_unknown_source_is_a_clear_error():
    with pytest.raises(AsciiChemError, match="unknown resolver source"):
        resolver.resolve_all(value="aspirin", convention="name",
                             sources=["nope"], fetch=OfflineFetch())


def test_pubchem_resolves_a_substance(cache):
    substance = resolver.resolve(
        value="aspirin", convention="name", sources=["pubchem"],
        fetch=RecordedFetch(PUBCHEM_ASPIRIN), cache=cache, now=PINNED)
    assert substance.preferred_name == "2-acetyloxybenzoic acid"
    assert substance.identifier_value("pubchem-cid") == "2244"
    assert substance.identifier_value("inchikey") == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert substance.molecular_weight == 180.16
    assert substance.structure is not None
    provenance = substance.provenance
    assert provenance.source == "pubchem"
    assert provenance.retrieved_at == PINNED
    assert provenance.attribution == "PubChem, U.S. National Library of Medicine"


def test_pubchem_builds_the_pug_rest_url():
    adapter = resolver.PubChem()
    assert adapter.url_for("aspirin", "name") == (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/"
        "property/CanonicalSMILES,IsomericSMILES,MolecularFormula,"
        "MolecularWeight,IUPACName,InChI,InChIKey/JSON")
    assert adapter.url_for("2244", "pubchem-cid") == (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/"
        "property/CanonicalSMILES,IsomericSMILES,MolecularFormula,"
        "MolecularWeight,IUPACName,InChI,InChIKey/JSON")
    # CAS numbers are indexed as names (the Common Chemistry mirror).
    assert adapter.url_for("50-78-2", "cas") == adapter.url_for("50-78-2", "name")
    assert adapter.supports("canonical-smiles")
    assert not adapter.supports("iupac-name")


def test_pubchem_not_found_is_none(cache):
    substance = resolver.resolve(value="not-a-molecule", convention="name",
                                 sources=["pubchem"],
                                 fetch=RecordedFetch(None), cache=cache)
    assert substance is None


def test_pubchem_malformed_json_raises_a_source_error(cache):
    # Direct adapter call: the module-level resolve() isolates source
    # failures (skip, not raise) - this asserts the failure exists.
    with pytest.raises(AsciiChemError, match="malformed JSON"):
        resolver.PubChem().resolve(value="aspirin", convention="name",
                                   fetch=RecordedFetch("<html>nope</html>"),
                                   cache=cache)


def test_pubchem_keeps_unparseable_smiles_as_identifier(cache):
    payload = {"PropertyTable": {"Properties": [{
        "CID": 7, "CanonicalSMILES": "not a smiles at all",
        "MolecularFormula": "X", "MolecularWeight": "1"}]}}
    substance = resolver.resolve(value="x", convention="name",
                                 sources=["pubchem"],
                                 fetch=RecordedFetch(payload), cache=cache)
    assert substance.structure is None
    assert substance.identifier_value("canonical-smiles") == "not a smiles at all"


def test_common_chemistry_key_chain(monkeypatch, cache):
    # Explicit None = no key: the env/file lookup does not run.
    assert resolver.CommonChemistry(api_key=None).api_key is None
    # The env var feeds the default.
    monkeypatch.setenv("COMMON_CHEMISTRY_API_KEY", "env-key")
    assert resolver.CommonChemistry().api_key == "env-key"
    # The token file is the fallback.
    monkeypatch.delenv("COMMON_CHEMISTRY_API_KEY")
    key_file = Path(tempfile.mkdtemp()) / "token.key"
    key_file.write_text("file-key\n")
    monkeypatch.setattr(common_chemistry, "KEY_FILE", key_file)
    assert resolver.CommonChemistry().api_key == "file-key"
    # Neither: None.
    monkeypatch.setattr(common_chemistry, "KEY_FILE",
                        Path("/nonexistent/key"))
    assert resolver.CommonChemistry().api_key is None


def test_common_chemistry_sends_the_key_headers(cache):
    fetch = RecordedFetch(CC_ASPIRIN)
    resolver.CommonChemistry(api_key="secret").resolve(
        value="50-78-2", convention="cas", fetch=fetch, cache=cache)
    url, headers = fetch.calls[0]
    assert url == ("https://commonchemistry.cas.org/direct-api/"
                   "detail?cas_rn=50-78-2")
    assert headers == {"Accept": "application/json",
                       "x-origin": "https://commonchemistry.cas.org/api-overview",
                       "X-API-KEY": "secret"}


def test_common_chemistry_without_key_sends_no_api_header(cache):
    fetch = RecordedFetch(CC_ASPIRIN)
    resolver.CommonChemistry(api_key=None).resolve(
        value="50-78-2", convention="cas", fetch=fetch, cache=cache)
    _, headers = fetch.calls[0]
    assert "X-API-KEY" not in headers


def test_common_chemistry_carries_the_cc_by_nc_attribution(cache):
    substance = resolver.resolve(
        value="50-78-2", convention="cas", sources=["common_chemistry"],
        fetch=RecordedFetch(CC_ASPIRIN), cache=cache, now=PINNED)
    assert substance.preferred_name == "Aspirin"
    assert substance.provenance.attribution == "CAS Common Chemistry (CC BY-NC 4.0)"
    assert substance.identifier_value("iupac-name") == "2-acetyloxybenzoic acid"
    # Every identifier carries the provenance.
    assert all(i.provenance.source == "common_chemistry"
               for i in substance.identifiers)


def test_opsin_supports_only_iupac_names(cache):
    fetch = RecordedFetch("CC(C)C(C)\n")
    substance = resolver.resolve(
        value="2,3-dimethylbutane", convention="iupac-name",
        sources=["opsin"], fetch=fetch, cache=cache, now=PINNED)
    assert fetch.calls[0][0] == \
        "https://opsin.ch.cam.ac.uk/opsin/2%2C3-dimethylbutane.smiles"
    assert substance.identifier_value("canonical-smiles") == "CC(C)C(C)"
    assert substance.structure is not None
    assert substance.provenance.attribution == \
        "OPSIN, University of Cambridge Centre for Molecular Informatics"
    assert not resolver.Opsin().supports("name")


def test_resolve_all_answers_one_record_per_answering_source(cache):
    """Every answering source produces its own record; a source-level
    failure (HTTP error, rejected request) skips that source - one
    dead or key-gated endpoint must not poison the others."""
    def source(name, key=None, failing=False):
        class _Source(resolver.Adapter):
            source_name = name
            supported_conventions = ("cas",)

            def fetch_record(self, *, value, convention, fetch, now=None):
                if failing:
                    raise AsciiChemError("HTTP 401: Unauthorized")
                identifiers = [Identifier(value=value, convention="cas",
                                          provenance=self.default_provenance(now))]
                if key:
                    identifiers.append(Identifier(value=key,
                                                  convention="inchikey",
                                                  provenance=self.default_provenance(now)))
                return Substance(identifiers=identifiers)

        return _Source

    registry = {"healthy_a": source("healthy_a", key=None),
                "key_gated": source("key_gated", failing=True),
                "healthy_b": source("healthy_b", key=None)}
    results = resolver.resolve_all(value="50-78-2", convention="cas",
                                   sources=["healthy_a", "key_gated",
                                            "healthy_b"],
                                   fetch=OfflineFetch(), cache=cache,
                                   registry=registry)
    assert [s.provenance.source for s in results] == ["healthy_a", "healthy_b"]


def test_resolve_all_skips_sources_that_do_not_support_the_convention(cache):
    results = resolver.resolve_all(
        value="50-78-2", convention="cas",
        sources=["pubchem", "common_chemistry", "opsin"],
        fetch=RecordedFetch(CC_ASPIRIN), cache=cache)
    # PubChem parses the CC payload as PUG REST (no PropertyTable)
    # and answers none; OPSIN does not support "cas" at all; Common
    # Chemistry answers.
    assert [s.provenance.source for s in results] == ["common_chemistry"]


def test_resolve_checked_conflicts_on_inchikey_disagreement(cache):
    def source(name, key):
        class _Source(resolver.Adapter):
            source_name = name
            supported_conventions = ("cas",)

            def fetch_record(self, *, value, convention, fetch, now=None):
                identifiers = [Identifier(value=value, convention="cas",
                                          provenance=self.default_provenance(now))]
                if key:
                    identifiers.append(Identifier(value=key,
                                                  convention="inchikey",
                                                  provenance=self.default_provenance(now)))
                return Substance(identifiers=identifiers)

        return _Source

    registry = {"a": source("a", "AAAA-BBBB-C"),
                "b": source("b", "CCCC-DDDD-D")}
    with pytest.raises(resolver.ResolverConflict,
                       match="sources disagree on substance identity"):
        resolver.resolve_checked(value="50-78-2", convention="cas",
                                 sources=["a", "b"],
                                 fetch=RecordedFetch("x"), cache=cache,
                                 registry=registry)


def test_resolve_checked_passes_when_keys_agree(cache):
    def source(name):
        class _Source(resolver.Adapter):
            source_name = name
            supported_conventions = ("cas",)

            def fetch_record(self, *, value, convention, fetch, now=None):
                return Substance(identifiers=[
                    Identifier(value=value, convention="cas",
                               provenance=self.default_provenance(now)),
                    Identifier(value="SAME-KEY-C", convention="inchikey",
                               provenance=self.default_provenance(now))])

        return _Source

    registry = {"a": source("a"), "b": source("b")}
    winner = resolver.resolve_checked(value="50-78-2", convention="cas",
                                      sources=["a", "b"],
                                      fetch=RecordedFetch("x"),
                                      cache=cache, registry=registry)
    assert winner.identifier_value("inchikey") == "SAME-KEY-C"


def test_cache_round_trip_and_offline_service(cache):
    adapter = resolver.PubChem()
    fetch = RecordedFetch(PUBCHEM_ASPIRIN)
    first = adapter.resolve(value="aspirin", convention="name",
                            fetch=fetch, cache=cache, now=PINNED)
    assert len(fetch.calls) == 1
    # A seeded entry answers without re-fetching.
    again = adapter.resolve(value="aspirin", convention="name",
                            fetch=OfflineFetch(), cache=cache)
    assert substance_to_wire(again) == substance_to_wire(first)


def test_cache_key_formula(cache):
    assert cache.key_for("pubchem", "cas", "50-78-2") == \
        hashlib.sha256(b"pubchem:cas:50-78-2").hexdigest()
    assert cache.ttl == 604800


def test_cache_stale_entries_do_not_serve(cache, monkeypatch):
    adapter = resolver.PubChem()
    fetch = RecordedFetch(PUBCHEM_ASPIRIN)
    adapter.resolve(value="aspirin", convention="name", fetch=fetch,
                    cache=cache, now=PINNED)
    # Age the single entry past the TTL.
    entry_path = next(cache.dir.glob("*.json"))
    entry = json.loads(entry_path.read_text())
    entry["cached_at"] = "2020-01-01T00:00:00Z"
    entry_path.write_text(json.dumps(entry))
    with pytest.raises(AsciiChemError, match="network must not be touched"):
        adapter.resolve(value="aspirin", convention="name",
                        fetch=OfflineFetch(), cache=cache)


def test_refresh_bypasses_the_cache(cache):
    adapter = resolver.PubChem()
    adapter.resolve(value="aspirin", convention="name",
                    fetch=RecordedFetch(PUBCHEM_ASPIRIN), cache=cache)
    fetch = RecordedFetch(PUBCHEM_ASPIRIN)
    adapter.resolve(value="aspirin", convention="name", fetch=fetch,
                    cache=cache, refresh=True)
    assert len(fetch.calls) == 1


def test_substance_wire_round_trip():
    from asciichem.resolver.substance import Property

    provenance = Provenance(source="example", retrieved_at=PINNED,
                            source_version="2026.11",
                            attribution="Example Data Source")
    substance = Substance(
        preferred_name="Aspirin", synonyms=["acetylsalicylic acid"],
        formula="C9H8O4", molecular_weight=180.16,
        identifiers=[Identifier(value="50-78-2", convention="cas",
                                provenance=provenance),
                     Identifier(value="BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                                convention="inchikey",
                                provenance=provenance)],
        properties=[Property(name="molecular-weight", value="180.16",
                             units="g/mol", provenance=provenance)])
    wire = substance_to_wire(substance)
    assert wire["type"] == "substance-record"
    assert wire["preferredName"] == "Aspirin"
    assert wire["identifiers"][0]["provenance"]["retrievedAt"] == PINNED
    assert wire["properties"][0]["units"] == "g/mol"
    rebuilt = substance_from_wire(json.dumps(wire))
    assert substance_to_wire(rebuilt) == wire
    assert rebuilt.identifier_value("cas") == "50-78-2"
    assert rebuilt.provenance.source_version == "2026.11"


def test_provenance_note_text():
    assert Provenance(source="s").note_text() is None
    assert Provenance(source="s", retrieved_at=PINNED,
                      attribution="A").note_text() == "Retrieved from s; A"
    assert Provenance(source="s", retrieved_at=PINNED,
                      source_version="1.0").note_text() == \
        "Retrieved from s; version 1.0"
    assert Provenance(source="s", source_version="1.0",
                      attribution="A").note_text() == \
        "Retrieved from s; version 1.0; A"
