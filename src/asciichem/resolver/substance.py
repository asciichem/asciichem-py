"""Domain value objects for resolution results - the Python mirror of
`AsciiChem::Resolver::Substance`. These are data carriers; (de)serialization
to the canonical wire form goes through the wire module only
(`asciichem.wire.substance_to_wire` / `substance_from_wire`)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Provenance:
    """Where a piece of identity data came from: the source adapter,
    the retrieval timestamp, the source version, and the attribution
    line licensing requires (CAS Common Chemistry: CC BY-NC 4.0)."""
    source: str
    retrieved_at: Optional[str] = None
    source_version: Optional[str] = None
    attribution: Optional[str] = None

    def note_text(self) -> Optional[str]:
        """The provenance record as one sentence (the citation track's
        bibitem note): retrieval source and version plus the source's
        required attribution - licence terms (CAS Common Chemistry is
        CC BY-NC 4.0) must survive into the rendered document. None
        when there is nothing beyond the source name itself."""
        details = [self.source_version and f"version {self.source_version}",
                   self.attribution]
        details = [d for d in details if d]
        if not details:
            return None
        return f"Retrieved from {self.source}; {'; '.join(details)}"


@dataclass
class Identifier:
    value: str
    convention: str
    provenance: Optional[Provenance] = None


@dataclass
class Property:
    name: str
    value: str
    units: Optional[str] = None
    provenance: Optional[Provenance] = None


@dataclass
class Substance:
    """A resolved substance: identity + provenance + best-effort
    structure. `structure` is a Model Molecule when the source's
    SMILES parses within the supported subset; otherwise None and the
    SMILES stays available as an identifier - nothing is dropped."""
    preferred_name: Optional[str] = None
    synonyms: list = field(default_factory=list)
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    identifiers: list = field(default_factory=list)
    properties: list = field(default_factory=list)
    structure: object = None

    def identifier_value(self, convention: str) -> Optional[str]:
        for identifier in self.identifiers:
            if identifier.convention == convention:
                return identifier.value
        return None

    @property
    def provenance(self) -> Optional[Provenance]:
        return self.identifiers[0].provenance if self.identifiers else None
