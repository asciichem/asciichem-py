"""PubChem source adapter (TODO.impl 39; TODO.v2 07 Layer 2) - the
Python mirror of `AsciiChem::Resolver::PubChem`. PubChem is the
default source: keyless PUG-REST, permissive terms, and the bulk
mirror of Common Chemistry's validated CAS RNs (CAS numbers are
indexed as names). Politeness: one request per resolution, 10s
timeout, no retries - PubChem asks for max 5 requests/second.

Citation profile (TODO.v2 08): publisher NCBI, per-access versioning
(retrievedAt provenance), compound link
https://pubchem.ncbi.nlm.nih.gov/compound/{cid}."""
from __future__ import annotations

import json
from urllib.parse import quote_plus

from ..model import AsciiChemError
from ..smiles import parse_smiles
from .adapter import Adapter, Fetch
from .substance import Identifier, Property, Substance

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
PROPERTIES = ["CanonicalSMILES", "IsomericSMILES", "MolecularFormula",
              "MolecularWeight", "IUPACName", "InChI", "InChIKey"]

# CAS numbers are indexed as PubChem names (Common Chemistry mirror),
# so both map to the name namespace.
NAMESPACES = {"name": "name", "cas": "name", "pubchem-cid": "cid",
              "inchikey": "inchikey", "inchi": "inchi",
              "smiles": "smiles", "canonical-smiles": "smiles"}


class PubChem(Adapter):
    source_name = "pubchem"
    supported_conventions = ("name", "cas", "pubchem-cid", "inchikey",
                             "inchi", "smiles", "canonical-smiles")

    def fetch_record(self, *, value: str, convention: str,
                     fetch: Fetch, now=None) -> Substance | None:
        body = fetch.get(self.url_for(value, convention))
        if body is None:
            return None
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise AsciiChemError(
                f"pubchem returned malformed JSON: {exc}") from exc
        properties = (payload.get("PropertyTable", {}).get("Properties")
                      or [None])[0]
        if not properties:
            return None
        return self.substance_from(properties, now)

    def attribution(self) -> str:
        return "PubChem, U.S. National Library of Medicine"

    def url_for(self, value: str, convention: str) -> str:
        namespace = NAMESPACES[str(convention)]
        return (f"{BASE}/{namespace}/{quote_plus(str(value))}"
                f"/property/{','.join(PROPERTIES)}/JSON")

    def substance_from(self, props: dict, now=None) -> Substance:
        provenance = self.default_provenance(now)
        identifiers = []
        if props.get("CID"):
            identifiers.append(Identifier(value=str(props["CID"]),
                                          convention="pubchem-cid",
                                          provenance=provenance))
        for key, convention in (("InChIKey", "inchikey"), ("InChI", "inchi"),
                                ("CanonicalSMILES", "canonical-smiles")):
            if props.get(key):
                identifiers.append(Identifier(value=props[key],
                                              convention=convention,
                                              provenance=provenance))
        return Substance(
            preferred_name=props.get("IUPACName"),
            formula=props.get("MolecularFormula"),
            molecular_weight=_to_float(props.get("MolecularWeight")),
            identifiers=identifiers,
            properties=[p for p in (
                Property(name="molecular-formula",
                         value=props.get("MolecularFormula"),
                         provenance=provenance)
                if props.get("MolecularFormula") else None,
                Property(name="molecular-weight",
                         value=props.get("MolecularWeight"), units="g/mol",
                         provenance=provenance)
                if props.get("MolecularWeight") else None,
            ) if p is not None],
            # Best effort: PubChem canonical SMILES may use constructs
            # outside the supported subset (stereo); then the SMILES
            # stays as an identifier and structure is None - no
            # silent loss.
            structure=parse_structure(props.get("CanonicalSMILES")))


def parse_structure(smiles: str | None):
    if not smiles:
        return None
    try:
        return parse_smiles(smiles).nodes[0]
    except AsciiChemError:
        return None


def _to_float(raw) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
