"""Citation track (TODO.v2 08; TODO.impl 44) - the Python mirror of
`AsciiChem::Citation`: a bibitem is a function of (substance, source)
- one dataset-type Relaton bibitem per source the user chooses to
cite, because two databases agreeing on a substance are still two
different documents.

Source-specific fields (link shape, publisher string, docidentifier
scheme) live in PROFILES; adding a source is one entry (OCP).

The bibitem XML is BUILT and SERIALIZED through leptris (the
mandated serialization layer, the family's lxml-shaped binding of
libleptris): `relaton_payload` assembles the typed Bibitem payload
and `to_xml` emits it via leptris's programmatic construction API
(Document.create / create_child / set / .text) and serializer - no
hand-rolled XML strings; escaping, well-formedness, and the
emitted bytes all come from libleptris. The corpus citation goldens
are byte-captures of the reference implementation's emission for the
same substances; a wire-shape change re-records them against the
reference in a corpus PR - never per-implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from leptris import Document, tostring

from .model import AsciiChemError, Molecule
from .resolver import Cache, Fetch, Substance, adapters as source_adapters
from .resolver.substance import Provenance

# The relaton-models schema version relaton-bib 2.1.9 renders on
# every bibitem (`Relaton.schema_versions["relaton-models"]`); the
# goldens pin it, so a relaton ride that bumps it re-records them
# deliberately.
SCHEMA_VERSION = "v1.5.6"


class CitationError(AsciiChemError):
    pass


@dataclass
class Profile:
    """Source-specific citation profile. Default fields fall back to
    the substance's provenance."""
    publisher: Optional[str] = None
    link_for: Callable[[Substance], Optional[str]] = lambda s: None
    identifier_for: Callable[[Substance], Optional[str]] = lambda s: None


def _pubchem_link(substance: Substance) -> Optional[str]:
    cid = substance.identifier_value("pubchem-cid")
    return f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None


def _pubchem_identifier(substance: Substance) -> Optional[str]:
    cid = substance.identifier_value("pubchem-cid")
    return f"PubChem CID {cid}" if cid else None


def _cc_link(substance: Substance) -> Optional[str]:
    cas = substance.identifier_value("cas")
    return f"https://commonchemistry.cas.org/detail?cas_rn={cas}" if cas else None


def _cc_identifier(substance: Substance) -> Optional[str]:
    cas = substance.identifier_value("cas")
    return f"CAS RN {cas}" if cas else None


def _default_identifier(substance: Substance) -> Optional[str]:
    key = substance.identifiers[0] if substance.identifiers else None
    return f"{key.convention}: {key.value}" if key else None


# The known sources. Keys are the resolver's source names.
PROFILES: dict[str, Profile] = {
    "pubchem": Profile(
        publisher="PubChem, U.S. National Library of Medicine",
        link_for=_pubchem_link,
        identifier_for=_pubchem_identifier,
    ),
    "common_chemistry": Profile(
        publisher="CAS Common Chemistry",
        link_for=_cc_link,
        identifier_for=_cc_identifier,
    ),
}

# The fallback profile for sources without a dedicated entry: no
# publisher override, no link, and the document identifier is the
# substance's first identifier.
DEFAULT_PROFILE = Profile(identifier_for=_default_identifier)


@dataclass
class Bibitem:
    """The dataset bibitem payload: the version-independent fields the
    citation track assembles, translated to the wire shape by
    `to_xml`."""
    bibitem_type: str
    title: str
    docidentifier_type: str
    docidentifier: str
    publisher: str
    accessed_on: str
    link: Optional[str] = None
    keywords: list = field(default_factory=list)
    note: Optional[str] = None


def bibitem(substance: Substance) -> Bibitem:
    """Builds the Relaton dataset-type bibitem payload for the
    substance as cited from its recorded source. Raises when the
    substance carries no provenance (hand-built, not resolved)."""
    provenance = substance.provenance
    if provenance is None or not provenance.source:
        raise CitationError(
            "substance has no provenance - resolve it first "
            "(asciichem.resolver)")
    profile = PROFILES.get(provenance.source, DEFAULT_PROFILE)
    publisher = profile.publisher or provenance.source
    identifier = profile.identifier_for(substance) or \
        f"{provenance.source} substance"
    return Bibitem(
        bibitem_type="dataset",
        title=f"{_title_base(substance)} - {publisher} substance record",
        docidentifier_type=provenance.source,
        docidentifier=identifier,
        publisher=publisher,
        accessed_on=_accessed_on(provenance),
        link=profile.link_for(substance),
        keywords=[f"{i.convention}={i.value}" for i in substance.identifiers],
        note=provenance.note_text(),
    )


def to_xml(substance: Substance) -> str:
    """The bibitem XML (what a document pipeline embeds)."""
    return _serialize(bibitem(substance))


def relaton_payload(substance: Substance) -> Bibitem:
    """Alias of `bibitem` (the payload-building half, exposed for
    consumers that want the fields without the XML)."""
    return bibitem(substance)


def for_molecule(molecule: Molecule, cache: Optional[Cache] = None,
                 fetch: Optional[Fetch] = None,
                 sources=None) -> list[tuple[str, Bibitem]]:
    """The cite syntax (TODO.impl 45): a molecule annotated
    `@cite("pubchem")` (a property annotation - the grammar needs no
    extension) declares *which source to cite it from*. This resolves
    the molecule's identifiers and emits one bibitem per cited
    source. Returns (source, Bibitem) pairs; empty when the molecule
    has no @cite annotations.

    `sources` limits the citable set (an allow-list): cited sources
    outside it are skipped - the document pipeline (the metanorma
    plugin's `asciichem-sources` attribute) warns about the
    exclusions; the library stays silent and total."""
    citable = _citation_sources(molecule)
    if sources is not None:
        citable = [s for s in citable if s in [str(x) for x in sources]]
    if not citable:
        return []

    convention, value = _lookup_key(molecule)
    if not value:
        raise CitationError(
            "molecule carries no resolvable identifier for citation "
            "(annotate @cas/@inchikey/@smiles or @name)")

    pairs = []
    for source in citable:
        adapter_class = source_adapters.get(source)
        if adapter_class is None:
            continue
        substance = adapter_class().resolve(value=value,
                                            convention=convention,
                                            cache=cache, fetch=fetch)
        if substance is None:
            continue
        pairs.append((source, bibitem(substance)))
    return pairs


# -- payload fields ------------------------------------------------------

def _title_base(substance: Substance) -> str:
    return (substance.preferred_name
            or substance.identifier_value("cas")
            or substance.identifier_value("inchikey")
            or "Substance")


def _accessed_on(provenance: Provenance) -> str:
    if provenance.retrieved_at:
        return provenance.retrieved_at[:10]
    return time.strftime("%Y-%m-%d", time.gmtime())


def _citation_sources(molecule: Molecule) -> list[str]:
    """The property annotation whose title is "cite": values are the
    source names to cite from."""
    return [p["value"] for p in molecule.properties
            if p.get("title") == "cite" and p.get("value")]


def _lookup_key(molecule: Molecule) -> tuple[str, Optional[str]]:
    """First identifier the resolver can look up by, in preference
    order: unambiguous registry keys before names."""
    for identifier in molecule.identifiers:
        if identifier.convention in ("cas", "inchikey", "pubchem-cid"):
            return identifier.convention, identifier.value
    if molecule.names:
        return "name", molecule.names[0].content
    return "", None


# -- XML emission (through leptris) --------------------------------------

def derived_id(docidentifier: str) -> Optional[str]:
    """The bibitem `id`: relaton derives it from the document
    identifier by stripping every non-word character (ASCII letters,
    digits, underscore). An identifier that reduces to nothing
    carries no id."""
    id_chars = [c for c in docidentifier
                if c.isascii() and (c.isalnum() or c == "_")]
    return "".join(id_chars) or None


def _serialize(payload: Bibitem) -> str:
    """Serializes through leptris: the document is built with
    leptris's construction API and emitted with its serializer (two-
    space indentation, no XML declaration, self-closing empty
    elements) - the Relaton bibitem emission shape the goldens pin."""
    document = Document.create()
    root = document.create_element("bibitem")
    id_value = derived_id(payload.docidentifier)
    if id_value:
        root.set("id", id_value)
    root.set("type", payload.bibitem_type)
    root.set("schema-version", SCHEMA_VERSION)

    title = root.create_child("title")
    title.set("type", "main")
    title.text = payload.title

    if payload.link:
        uri = root.create_child("uri")
        uri.set("type", "src")
        uri.text = payload.link

    docidentifier = root.create_child("docidentifier")
    docidentifier.set("type", payload.docidentifier_type)
    docidentifier.text = payload.docidentifier

    date = root.create_child("date")
    date.set("type", "accessed")
    on = date.create_child("on")
    on.text = payload.accessed_on

    contributor = root.create_child("contributor")
    role = contributor.create_child("role")
    role.set("type", "publisher")
    organization = contributor.create_child("organization")
    name = organization.create_child("name")
    name.text = payload.publisher

    if payload.note:
        note = root.create_child("note")
        note.set("type", "provenance")
        note.text = payload.note

    for keyword in payload.keywords:
        keyword_element = root.create_child("keyword")
        vocab = keyword_element.create_child("vocab")
        vocab.text = keyword

    document.set_root(root)
    return tostring(document, pretty_print=True, encoding="unicode")
