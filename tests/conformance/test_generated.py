"""Cross-implementation differential gate (TODO.v2/13 item 5): the
generated cases come from the Ruby reference's property harness
(asciichem-tests/corpus/generated/models.json — wire JSON plus
canonical text, deterministic seed). This implementation must ingest
the wire into the model, render the exact canonical text, and
re-parse to an equal wire form. Skips when the corpus is absent."""
import json
import os
from pathlib import Path

import pytest

from asciichem import parse_text
from asciichem.text import render
from asciichem.wire import from_model_json, to_model_json

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))
GENERATED = CORPUS / "corpus" / "generated" / "models.json"

pytestmark = pytest.mark.skipif(
    not GENERATED.is_file(),
    reason="generated cases not present (clone asciichem-tests main)")


def cases():
    if not GENERATED.is_file():
        return []
    data = json.loads(GENERATED.read_text())
    return data["cases"]


@pytest.mark.parametrize("case", cases(), ids=[str(i) for i in range(len(cases()))])
def test_renders_and_reparses_identically(case):
    formula = from_model_json(case["wire"])
    assert render(formula) == case["text"]
    reparsed = parse_text(case["text"])
    assert json.loads(to_model_json(reparsed)) == case["wire"]
