"""A resolver source adapter - the Python mirror of
`AsciiChem::Resolver::Adapter`. Subclasses declare `source_name` and
`supported_conventions` and implement `fetch_record`; `resolve`
handles caching around it. The transport (`fetch`) is injectable so
specs run offline; `NetFetch` (stdlib urllib - the zero-dependency
policy holds except the declared runtime deps) is the live default."""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Optional, Protocol

from ..model import AsciiChemError
from .cache import Cache
from .substance import Provenance, Substance


class Fetch(Protocol):
    """The transport seam: answers `get(url, headers)` with the body
    String, None for not-found (HTTP 404), and raises AsciiChemError
    for other HTTP failures. Specs inject recorded responses."""

    def get(self, url: str, headers: Optional[dict] = None) -> Optional[str]:
        ...  # pragma: no cover


class Adapter:
    """Base class for source adapters. `resolve` handles caching
    around `fetch_record`; `now` is the injectable clock (a UTC
    timestamp callable) the recorded-fixture runs pin."""

    source_name: str = ""
    supported_conventions: tuple = ()

    def supports(self, convention) -> bool:
        return str(convention) in self.supported_conventions

    def resolve(self, *, value: str, convention: str,
                fetch: Optional[Fetch] = None,
                cache: Optional[Cache] = None,
                refresh: bool = False,
                now=None) -> Optional[Substance]:
        cache = cache if cache is not None else Cache()
        fetch = fetch if fetch is not None else NetFetch()
        key = cache.key_for(self.source_name, convention, value)

        if not refresh:
            cached = cache.read(key)
            if cached is not None:
                return cached

        substance = self.fetch_record(value=value, convention=convention,
                                      fetch=fetch, now=now)
        if substance is not None:
            for identifier in substance.identifiers:
                if identifier.provenance is None:
                    identifier.provenance = self.default_provenance(now)
            for prop in substance.properties:
                if prop.provenance is None:
                    prop.provenance = self.default_provenance(now)
            cache.write(key, substance)
        return substance

    def fetch_record(self, *, value: str, convention: str,
                     fetch: Fetch, now=None) -> Optional[Substance]:
        """Not-found is None, not an error - sources legitimately
        answer "unknown substance" during multi-source resolution."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement fetch_record")

    def default_provenance(self, now=None) -> Provenance:
        timestamp = now() if callable(now) else (
            now if isinstance(now, str) else _utcnow())
        return Provenance(source=self.source_name,
                          retrieved_at=timestamp,
                          attribution=self.attribution())

    def attribution(self) -> Optional[str]:
        return None


class NetFetch:
    """Minimal stdlib HTTP transport: GET with at most three
    redirects, 5s connect / 10s read timeouts, one request per
    resolution. 404 answers None (not-found); any other non-success
    raises AsciiChemError (the per-source isolation owns the
    skip)."""

    def get(self, url: str, headers: Optional[dict] = None,
            limit: int = 3) -> Optional[str]:
        if limit == 0:
            raise AsciiChemError("too many redirects")
        merged = {"User-Agent": "asciichem-resolver"}
        if headers:
            merged.update(headers)
        request = urllib.request.Request(url, headers=merged)
        try:
            response = urllib.request.urlopen(
                request, timeout=10)  # noqa: S310 - https sources only
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise AsciiChemError(f"HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise AsciiChemError(f"request failed: {exc}") from exc
        if 300 <= response.status < 400:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise AsciiChemError("redirect without a location")
            from urllib.parse import urljoin
            return self.get(urljoin(url, location), headers, limit - 1)
        if 200 <= response.status < 300:
            return response.read().decode("utf-8")
        response.close()
        raise AsciiChemError(f"HTTP {response.status}")


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
