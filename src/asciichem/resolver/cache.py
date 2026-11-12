"""Resolution cache in the USER cache dir (never the package install
dir): entries carry source, retrieved-at, attribution, TTL - the
Python mirror of `AsciiChem::Resolver::Cache`. Chemistry identifiers
are stable; the default TTL is one week."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .substance import Substance

DEFAULT_TTL = 7 * 24 * 60 * 60  # chemistry identifiers are stable


def _parse_utc(raw: str) -> Optional[datetime]:
    """Parse the wire timestamp form back to an aware datetime
    (loose on garbage: None)."""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None


def default_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "asciichem" / "resolver"


class Cache:
    def __init__(self, dir: Optional[Path] = None, ttl: int = DEFAULT_TTL):
        self.dir = Path(dir) if dir is not None else default_dir()
        self.ttl = ttl

    def key_for(self, source: str, convention: str, value: str) -> str:
        """The shared caching contract's key formula:
        sha256("source:convention:value")."""
        return hashlib.sha256(
            f"{source}:{convention}:{value}".encode("utf-8")).hexdigest()

    def read(self, key: str) -> Optional[Substance]:
        path = self.dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if self._stale(entry):
            return None
        from ..wire import substance_from_wire
        try:
            return substance_from_wire(entry["substance"])
        except (KeyError, TypeError, ValueError):
            return None

    def write(self, key: str, substance: Substance) -> Substance:
        self.dir.mkdir(parents=True, exist_ok=True)
        from ..wire import substance_to_wire
        entry = {
            "cached_at": _utcnow(),
            "ttl_seconds": self.ttl,
            "substance": substance_to_wire(substance),
        }
        (self.dir / f"{key}.json").write_text(
            json.dumps(entry, indent=2), encoding="utf-8")
        return substance

    def _stale(self, entry: dict) -> bool:
        cached_at = _parse_utc(entry.get("cached_at"))
        if cached_at is None:
            return True
        now = datetime.now(timezone.utc)
        age = (now - cached_at).total_seconds()
        return age > entry.get("ttl_seconds", self.ttl)


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
