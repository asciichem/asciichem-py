"""OPSIN source adapter (TODO.v2/13 item 3) - the Python mirror of
`AsciiChem::Resolver::Opsin`: IUPAC name -> structure - name-to-
structure perception for systematic names, the piece PubChem's name
lookup cannot do (PubChem indexes a curated name list; OPSIN
*derives* the structure from any systematic IUPAC name). Uses the
public OPSIN web service (https://opsin.ch.cam.ac.uk), which answers
with CML or SMILES.

Citation profile: publisher University of Cambridge Centre for
Molecular Informatics; per-access retrievedAt provenance."""
from __future__ import annotations

from urllib.parse import quote_plus

from ..model import AsciiChemError
from .adapter import Adapter, Fetch
from .pubchem import parse_structure
from .substance import Identifier, Substance

BASE = "https://opsin.ch.cam.ac.uk/opsin"


class Opsin(Adapter):
    source_name = "opsin"
    # Derivation only: 'name' is the curated-lookup convention
    # (PubChem/Common Chemistry). OPSIN DERIVES structures from
    # systematic names - reachable via 'iupac-name' so it can never
    # preempt a lookup source.
    supported_conventions = ("iupac-name",)

    def fetch_record(self, *, value: str, convention: str = "",
                     fetch: Fetch, now=None) -> Substance | None:
        try:
            body = fetch.get(f"{BASE}/{quote_plus(str(value))}.smiles")
        except AsciiChemError as exc:
            raise AsciiChemError(f"opsin request failed: {exc}") from exc
        if body is None:
            return None
        return self.substance_from(body.strip(), now)

    def attribution(self) -> str:
        return "OPSIN, University of Cambridge Centre for Molecular Informatics"

    def substance_from(self, smiles: str, now=None) -> Substance | None:
        if not smiles:
            return None
        provenance = self.default_provenance(now)
        return Substance(
            identifiers=[Identifier(value=smiles, convention="canonical-smiles",
                                    provenance=provenance)],
            # OPSIN answers SMILES that may use features outside the
            # supported subset (wildcards, exotic stereo); then the
            # SMILES stays as an identifier and structure is None -
            # no silent loss.
            structure=parse_structure(smiles))
