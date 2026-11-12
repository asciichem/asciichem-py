"""Shared-corpus conformance: the linter-expectations family
(suite 0.11.0). Match semantics per the runner contract: every
expectation must match at least one actual diagnostic (same severity,
message containing `messageIncludes`, and - when given - the same
`node` label) AND every actual diagnostic must match at least one
expectation. `lint: []` demands a completely clean run.

Engine posture: the identity cross-check is engine-scaled. Cases
marked `engine: "inchi"` encode the with-engine posture, cases marked
`engine: "none"` the no-engine posture. A runner whose posture
differs from the marker SKIPS the case and reports the skip - a
missing engine is a posture, not a failure. This implementation's
posture follows the installed engine (rdkit -> "inchi"; CI runs both
postures as separate jobs)."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from asciichem import ParseError, parse_text
from asciichem import linter

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

pytestmark = pytest.mark.skipif(
    not (CORPUS / "corpus" / "fixtures").is_dir(),
    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")


def _has_rdkit() -> bool:
    return importlib.util.find_spec("rdkit") is not None


if _has_rdkit():
    # Engine-present posture: the corpus's `engine: "inchi"` cases
    # exercise the identity cross-check through the real engine.
    from asciichem.inchi import RdkitEngine, set_engine

    set_engine(RdkitEngine())

POSTURE = "inchi" if _has_rdkit() else "none"


def linter_fixtures():
    if not (CORPUS / "corpus" / "fixtures").is_dir():
        return []  # corpus absent entirely - the skip mark owns this
    path = CORPUS / "corpus" / "fixtures" / "linter-expectations.json"
    if not path.is_file():
        pytest.fail("linter-expectations.json missing from the corpus - "
                    "the pin does not carry suite 0.11.0")
    return json.loads(path.read_text())


def matches(expectation, diagnostic) -> bool:
    if diagnostic.severity != expectation["severity"]:
        return False
    if expectation["messageIncludes"] not in diagnostic.message:
        return False
    node = expectation.get("node")
    if node is not None and diagnostic.label != node:
        return False
    return True


@pytest.mark.parametrize("case", linter_fixtures(), ids=lambda c: c["id"])
def test_linter_expectations(case):
    marked = case.get("engine")
    if marked is not None and marked != POSTURE:
        pytest.skip(f"engine posture {marked!r} != this run's {POSTURE!r}; "
                    "the case is exercised by the matching-posture job")
    if case["parses"]:
        formula = parse_text(case["input"])  # must not raise
    else:
        with pytest.raises(ParseError):
            parse_text(case["input"])
        return

    actual = linter.run(formula)
    expectations = case["lint"]
    for expectation in expectations:
        assert any(matches(expectation, d) for d in actual), (
            f'{case["id"]}: no diagnostic matches {expectation}; '
            f'actual: {[(d.severity, d.label, d.message) for d in actual]}')
    for diagnostic in actual:
        assert any(matches(expectation, diagnostic)
                   for expectation in expectations), (
            f'{case["id"]}: unexpected diagnostic '
            f'({diagnostic.severity}, {diagnostic.label}, '
            f'{diagnostic.message!r})')
