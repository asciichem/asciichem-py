"""Resolution turns identifiers into substance records (TODO.v2 07
Layer 2; TODO.impl 38) - the Python mirror of `AsciiChem::Resolver`.
Multi-source by design: every answering source produces its own
record with provenance, because the citation track (TODO.v2 08) emits
one bibitem per (substance, source) the user chooses to cite.

Sources are adapters registered in this module's registry (same open
pattern as the linter). All shipped sources are on by default; every
result carries its source's required attribution (Common Chemistry:
CC BY-NC 4.0 - default-on per the maintainer decision of 2026-11-11).
Callers limit the set explicitly: `resolve_all(sources=...)` or the
metanorma plugin's `asciichem-sources` attribute for NC-free
documents.

Failure posture: a source-level failure (HTTP error, rejected
request, malformed payload) skips that source - one dead or key-gated
endpoint must not poison the others; callers see what answered.
`ResolverConflict` is the exception: cross-check mode
(`resolve_checked`, the Ruby `resolve!`) raises when more than one
answering source provides an InChIKey and they disagree - the message
lists the disagreement, never hidden.

The transport is injectable through the `Fetch` protocol - the
offline seam the recorded-fixture runs go through."""
from __future__ import annotations

from typing import Optional

from ..model import AsciiChemError
from .adapter import Adapter, Fetch, NetFetch
from .cache import Cache
from .common_chemistry import CommonChemistry
from .opsin import Opsin
from .pubchem import PubChem
from .substance import Identifier, Property, Provenance, Substance

__all__ = ["Adapter", "Fetch", "NetFetch", "Cache", "CommonChemistry",
           "Opsin", "PubChem", "Identifier", "Property", "Provenance",
           "Substance", "ResolverConflict", "adapters", "register",
           "resolve", "resolve_all", "resolve_checked"]


class ResolverConflict(AsciiChemError):
    """Sources disagree on substance identity (cross-check mode)."""


# The source registry: name => adapter CLASS. Classes, not instances,
# populate the registry because construction can consult the
# environment - the Common Chemistry key chain reads the env var and
# the token file at build time.
adapters: dict[str, type] = {}


def register(name: str, adapter_class) -> None:
    adapters[str(name)] = adapter_class


def _select(registry: dict, sources) -> list[type]:
    chosen = []
    for source in sources:
        name = str(source)
        if name not in registry:
            raise AsciiChemError(
                f"unknown resolver source {source!r} (registered: "
                f"{', '.join(sorted(registry))})")
        chosen.append(registry[name])
    return chosen


def resolve_all(*, value: str, convention: str,
                sources=None, fetch: Optional[Fetch] = None,
                cache: Optional[Cache] = None, refresh: bool = False,
                now=None, registry: Optional[dict] = None) -> list[Substance]:
    """Resolves against every registered source (or the selected
    subset), returning one Substance per source that answers (nil
    entries dropped). Sources answer in registration order. A
    source-level failure (HTTP error, rejected request) skips that
    source - one dead or key-gated endpoint must not poison the
    others; callers see what answered. `registry` overrides the
    module registry (a composed source set, the tests' seam)."""
    source_registry = registry if registry is not None else adapters
    selected = (_select(source_registry, sources)
                if sources is not None else list(source_registry.values()))
    results = []
    for adapter_class in selected:
        adapter = adapter_class()
        if not adapter.supports(convention):
            continue
        try:
            substance = adapter.resolve(value=value, convention=convention,
                                        fetch=fetch, cache=cache,
                                        refresh=refresh, now=now)
        except AsciiChemError:
            continue
        if substance is not None:
            results.append(substance)
    return results


def resolve(**kwargs) -> Optional[Substance]:
    """First answering source wins."""
    results = resolve_all(**kwargs)
    return results[0] if results else None


def resolve_checked(**kwargs) -> Optional[Substance]:
    """Cross-check mode: all sources must agree on the primary key
    (InChIKey) when more than one source provides one."""
    substances = resolve_all(**kwargs)
    keys = []
    for substance in substances:
        key = substance.identifier_value("inchikey")
        if key and key not in keys:
            keys.append(key)
    if len(keys) > 1:
        raise ResolverConflict(
            "sources disagree on substance identity "
            f"(InChIKeys: {' vs '.join(keys)})")
    return substances[0] if substances else None


# Every shipped source self-registers at import (the Ruby autoload
# class-body order). CommonChemistry is CC BY-NC 4.0 and default-on
# per the maintainer decision of 2026-11-11; callers that need
# NC-free resolution limit sources explicitly.
register("pubchem", PubChem)
register("opsin", Opsin)
register("common_chemistry", CommonChemistry)
