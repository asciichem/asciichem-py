"""Shared-corpus conformance: the resolver family (suite 0.11.0).
The runner replays the recorded request/response through the named
source adapter: the adapter must build exactly `request.url` for
(source, convention, value) and send the adapter-declared
`request.headers` (`{api-key}` is the caller's key placeholder).
Feeding `response.body` must yield semantically the
`expectation.substance` wire record - every identifier and property
carrying its provenance with the pinned `retrievedAt` - plus
`structureParses` (source SMILES within the supported subset ->
structure attached; otherwise identifier-only, no silent loss). The
cache markers pin the shared caching contract: ttlSeconds,
keyFormula (sha256("source:convention:value")), servesOffline (a
seeded entry answers without re-fetching - and without a key, for
key-gated sources). A `skipped` outcome (the key-absent 401) must
drop the source without poisoning the others.

Hermeticity: the Common Chemistry key chain reads the environment
and the stored token file; the corpus's key-absent case demands the
no-key posture, so this module scrubs both for its duration (the
Ruby specs' approach). No real key ever enters this module."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from asciichem.model import AsciiChemError
from asciichem import resolver
from asciichem.wire import substance_to_wire

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

# The pinned retrieval time every expected record carries (the
# corpus pins it so records and the citation goldens that consume
# them are stable bytes).
PINNED_TIMESTAMP = "2026-01-05T09:30:00Z"
FAKE_KEY = "corpus-replay-test-key"  # never a real credential


pytestmark = [
    pytest.mark.skipif(
        not (CORPUS / "corpus" / "fixtures").is_dir(),
        reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)"),
]


@pytest.fixture(autouse=True)
def hermetic_key_chain(monkeypatch):
    """Key-hermetic: the corpus runs with no key unless the case
    explicitly carries one, whatever the machine carries in the env
    or the stored token file."""
    from asciichem.resolver import common_chemistry

    monkeypatch.delenv("COMMON_CHEMISTRY_API_KEY", raising=False)
    monkeypatch.setattr(common_chemistry, "KEY_FILE",
                        Path("/nonexistent/cas-common-chemistry-api.key"))


def resolver_fixtures():
    if not (CORPUS / "corpus" / "fixtures").is_dir():
        return []  # corpus absent entirely - the skip mark owns this
    path = CORPUS / "corpus" / "fixtures" / "resolver.json"
    if not path.is_file():
        pytest.fail("resolver.json missing from the corpus - the pin "
                    "does not carry suite 0.11.0")
    return json.loads(path.read_text())


def recorded_body(case_id: str) -> dict:
    """A recorded 200 response body by case id (re-used to seed the
    cache for the skipped case's servesOffline check)."""
    for candidate in resolver_fixtures():
        if candidate["id"] == case_id:
            return candidate["response"]["body"]
    raise AssertionError(f"recorded case {case_id} not found")


class ReplayFetch:
    """Answers with the recorded response; captures every request for
    the URL/header assertions."""

    def __init__(self, case):
        self._case = case
        self.calls = []

    def get(self, url, headers=None):
        self.calls.append((url, dict(headers or {})))
        status = self._case["response"]["status"]
        body = self._case["response"].get("body")
        if status == 404:
            return None
        if status != 200:
            raise AsciiChemError(f"HTTP {status}")
        return json.dumps(body) if isinstance(body, dict) else body


class OfflineFetch:
    """Network access from a cache-seeded lookup would be a bug."""

    @staticmethod
    def get(url, headers=None):
        raise AsciiChemError(f"network must not be touched: {url}")


def adapter_for(case):
    """The named source adapter, with the caller's key when the
    recorded request carries the `{api-key}` placeholder and no key
    when it does not (the key-absent case records the request WITHOUT
    the X-API-KEY header)."""
    headers = case["request"].get("headers", {})
    keyed = "{api-key}" in headers.values()
    if case["source"] == "common_chemistry":
        return resolver.CommonChemistry(api_key=FAKE_KEY if keyed else None)
    return resolver.adapters[case["source"]]()


def expected_headers(case):
    headers = dict(case["request"].get("headers", {}))
    if "{api-key}" in headers.values():
        headers["X-API-KEY"] = FAKE_KEY
    return headers


@pytest.mark.parametrize("case", resolver_fixtures(), ids=lambda c: c["id"])
def test_resolver_replay(case):
    cache = resolver.Cache(dir=Path(tempfile.mkdtemp(prefix="asciichem-corpus-")))
    fetch = ReplayFetch(case)
    outcome = case["expectation"]["outcome"]

    if outcome == "skipped":
        # A source-level failure drops the source without poisoning
        # the others: resolve_all answers empty for this source.
        results = resolver.resolve_all(
            value=case["value"], convention=case["convention"],
            sources=[case["source"]], fetch=fetch, cache=cache,
            now=PINNED_TIMESTAMP)
        assert results == [], case["id"]
    else:
        substance = adapter_for(case).resolve(
            value=case["value"], convention=case["convention"],
            fetch=fetch, cache=cache, now=PINNED_TIMESTAMP)
        assert substance is not None, case["id"]
        wire = substance_to_wire(substance)
        assert wire == case["expectation"]["substance"], (
            f'{case["id"]}: substance record mismatch')
        structure_parses = case["expectation"]["structureParses"]
        assert (substance.structure is not None) == structure_parses, case["id"]

    # The adapter must build exactly the recorded request.
    assert len(fetch.calls) == 1, case["id"]
    url, headers = fetch.calls[0]
    assert url == case["request"]["url"], (
        f'{case["id"]}: {url!r} != {case["request"]["url"]!r}')
    assert headers == expected_headers(case), (
        f'{case["id"]}: request headers {headers!r} != '
        f'{expected_headers(case)!r}')

    # The cache markers pin the shared caching contract.
    expectation = case["expectation"]
    key = cache.key_for(case["source"], case["convention"], case["value"])
    assert cache.ttl == expectation["cache"]["ttlSeconds"] == 604800, case["id"]
    import hashlib
    assert key == hashlib.sha256(
        f"{case['source']}:{case['convention']}:{case['value']}"
        .encode("utf-8")).hexdigest(), case["id"]
    if expectation.get("cache", {}).get("servesOffline"):
        # A seeded entry answers without re-fetching - and without a
        # key, for key-gated sources.
        if outcome == "resolved":
            served = adapter_for(case).resolve(
                value=case["value"], convention=case["convention"],
                fetch=OfflineFetch(), cache=cache)
            assert substance_to_wire(served) == \
                expectation["substance"], case["id"]
        else:
            # The key-absent source still serves its cache: seed it
            # through the keyed adapter, then answer offline, no key.
            seed = resolver.CommonChemistry(api_key=FAKE_KEY)
            seed.resolve(value=case["value"], convention=case["convention"],
                         fetch=ReplayFetch({
                             **case,
                             "response": {"status": 200,
                                          "body": recorded_body(
                                              "resolver/common-chemistry-cas-aspirin")},
                         }),
                         cache=cache, now=PINNED_TIMESTAMP)
            served = resolver.resolve_all(
                value=case["value"], convention=case["convention"],
                sources=[case["source"]], fetch=OfflineFetch(),
                cache=cache, now=PINNED_TIMESTAMP)
            assert len(served) == 1, case["id"]
