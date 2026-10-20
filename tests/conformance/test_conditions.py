"""Condition-entry conformance (TODO.v2/13 item 4, slice 1): typed
`role: value` entries and free text are shared semantics fixed by the
corpus (fixtures/conditions.json). Skips when the corpus is not cloned
as a sibling."""
import json
import os
from pathlib import Path

import pytest

from asciichem.model import Reaction
from asciichem.parser import parse_text

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

CONDITIONS_FILE = CORPUS / "corpus" / "fixtures" / "conditions.json"


def conditions_cases():
    if not CONDITIONS_FILE.is_file():
        return []
    return json.loads(CONDITIONS_FILE.read_text())


@pytest.mark.skipif(not CONDITIONS_FILE.is_file(),
                    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")
def test_has_condition_cases():
    assert len(conditions_cases()) >= 8


@pytest.mark.parametrize("case", conditions_cases(), ids=lambda c: c["id"])
def test_derives_shared_entries(case):
    reaction = parse_text(case["input"]).nodes[0]
    assert isinstance(reaction, Reaction)
    entries = [{"role": e.role, "value": e.value, "side": e.side}
               for e in (reaction.conditions.entries if reaction.conditions else [])]
    assert entries == case["conditionEntries"]
