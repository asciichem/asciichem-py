"""CAS Common Chemistry source adapter (TODO.impl 40; TODO.v2 07) -
the Python mirror of `AsciiChem::Resolver::CommonChemistry`.

License posture (maintainer decision 2026-11-11): Common Chemistry
data is CC BY-NC 4.0 - acceptable to enable by default; every result
carries the required attribution, and the citation track carries it
into rendered documents. Documents that must stay NC-free limit
sources explicitly (`resolve(sources=...)` or the metanorma plugin's
`asciichem-sources` attribute).

API access: CAS key-gated the API (401 without a key; request one at
cas.org/services/commonchemistry-api). The access email defines the
contract: GET /direct-api/detail with the token as the X-API-KEY
header plus x-origin and accept headers. The key comes per instance
(`api_key=`), from COMMON_CHEMISTRY_API_KEY, or from
~/.config/cas-common-chemistry-api.key (the token's stored
location). Tokens are personal and expire after 12 months (CAS
emails a renewal notice). Without a key, live lookups answer 401 and
the resolver skips this source; seeded caches keep working."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote_plus

from ..model import AsciiChemError
from .adapter import Adapter, Fetch
from .pubchem import parse_structure
from .substance import Identifier, Property, Substance

BASE = "https://commonchemistry.cas.org/direct-api/detail"
ORIGIN = "https://commonchemistry.cas.org/api-overview"
KEY_FILE = Path.home() / ".config" / "cas-common-chemistry-api.key"


def default_api_key() -> str | None:
    """The key chain: the env var first, then the stored token file.
    Returns None when neither carries a key (the source then sends no
    X-API-KEY header and the live endpoint answers 401 - the source
    is skipped, its cached entries keep serving)."""
    env = os.environ.get("COMMON_CHEMISTRY_API_KEY")
    if env:
        return env
    try:
        return KEY_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


class CommonChemistry(Adapter):
    source_name = "common_chemistry"
    supported_conventions = ("cas",)

    # The key-chain sentinel: a Python default argument evaluates at
    # class-definition time, so omission is marked by this object -
    # the env/file lookup runs per construction. Explicit None keeps
    # its Ruby meaning: no key, ever.
    _UNSET = object()

    def __init__(self, api_key=_UNSET):
        if api_key is CommonChemistry._UNSET:
            api_key = default_api_key()
        self.api_key = api_key

    def fetch_record(self, *, value: str, convention: str,
                     fetch: Fetch, now=None) -> Substance | None:
        headers = {
            "Accept": "application/json",
            "x-origin": ORIGIN,
        }
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        body = fetch.get(f"{BASE}?cas_rn={quote_plus(str(value))}", headers)
        if body is None:
            return None
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise AsciiChemError(
                f"common_chemistry returned malformed JSON: {exc}") from exc
        if not data:
            return None
        return self.substance_from(data, now)

    def attribution(self) -> str:
        return "CAS Common Chemistry (CC BY-NC 4.0)"

    def substance_from(self, data: dict, now=None) -> Substance:
        provenance = self.default_provenance(now)
        identifiers = [Identifier(value=data["rn"], convention="cas",
                                  provenance=provenance)]
        for key, convention in (("inchikey", "inchikey"),
                                ("smiles", "canonical-smiles"),
                                ("inchi", "inchi"),
                                ("iupacName", "iupac-name")):
            if data.get(key):
                identifiers.append(Identifier(value=data[key],
                                              convention=convention,
                                              provenance=provenance))
        molecular_mass = data.get("molecularMass")
        return Substance(
            preferred_name=data.get("name"),
            formula=data.get("formula"),
            identifiers=identifiers,
            properties=[p for p in (
                Property(name="molecular-formula", value=data.get("formula"),
                         provenance=provenance)
                if data.get("formula") else None,
                Property(name="molecular-weight",
                         value=str(molecular_mass) if molecular_mass is not None else None,
                         units="g/mol", provenance=provenance)
                if molecular_mass is not None else None,
            ) if p is not None],
            # Best effort within the supported SMILES subset; the
            # SMILES stays as an identifier when stereo keeps it out.
            structure=parse_structure(data.get("smiles")))
