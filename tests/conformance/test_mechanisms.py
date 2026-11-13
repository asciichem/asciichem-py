"""Mechanism-flow conformance (the corpus's mechanisms.json family):
the per-step electron-flow markers are shared semantics fixed by the
corpus - each fixture's `flows` map pins the (source, target) pairs
every step must carry, keyed by step label. Parse/reject, wire (L0)
and text round-trip (L1) run through the generic families in
test_corpus.py; this gate is the flow-marker depth. Skips when the
corpus is not cloned as a sibling."""
import json
import os
from pathlib import Path

import pytest

from asciichem.model import Mechanism
from asciichem import parse_text

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

MECHANISMS_FILE = CORPUS / "corpus" / "fixtures" / "mechanisms.json"


def mechanisms_cases():
    if not MECHANISMS_FILE.is_file():
        return []
    return json.loads(MECHANISMS_FILE.read_text())


@pytest.mark.skipif(not MECHANISMS_FILE.is_file(),
                    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")
def test_has_mechanism_cases():
    assert len(mechanisms_cases()) >= 3


@pytest.mark.parametrize("case", mechanisms_cases(), ids=lambda c: c["id"])
def test_derives_shared_flows(case):
    parsed = parse_text(case["input"])
    mechanism = next((n for n in parsed.nodes if isinstance(n, Mechanism)), None)
    assert mechanism is not None, f'{case["id"]}: no Mechanism node'
    by_label = {s.label: [(f.source, f.target) for f in s.flows]
                for s in mechanism.steps}
    expected = {label: [(f["source"], f["target"]) for f in flows]
                for label, flows in case["flows"].items()}
    assert by_label == expected, case["id"]
