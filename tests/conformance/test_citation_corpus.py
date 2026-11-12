"""Shared-corpus conformance: the citation family (suite 0.11.0).
For `substance` (the wire record, retrievedAt pinned) the citation
layer must emit exactly the `bibitem` bytes - byte-equal, no
trailing newline. A case with `rejected: true` must be rejected (a
substance without provenance is not citable)."""
import json
import os
from pathlib import Path

import pytest

from asciichem.model import AsciiChemError
from asciichem import citation
from asciichem.wire import substance_from_wire

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

pytestmark = pytest.mark.skipif(
    not (CORPUS / "corpus" / "fixtures").is_dir(),
    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")


def citation_fixtures():
    if not (CORPUS / "corpus" / "fixtures").is_dir():
        return []  # corpus absent entirely - the skip mark owns this
    path = CORPUS / "corpus" / "fixtures" / "citation.json"
    if not path.is_file():
        pytest.fail("citation.json missing from the corpus - the pin "
                    "does not carry suite 0.11.0")
    return json.loads(path.read_text())


@pytest.mark.parametrize("case", citation_fixtures(), ids=lambda c: c["id"])
def test_citation_golden(case):
    substance = substance_from_wire(case["substance"])
    if case.get("rejected"):
        with pytest.raises(AsciiChemError):
            citation.to_xml(substance)
        return
    emitted = citation.to_xml(substance)
    assert emitted == case["bibitem"], (
        f'{case["id"]}: bibitem bytes differ from the golden '
        "(byte-equal, no trailing newline)")
    # The documented child-element order (the sequence in which
    # element kinds appear; keywords repeat per identifier).
    order = _child_order(emitted)
    assert order == case["ordering"], (
        f'{case["id"]}: child order {order} != {case["ordering"]}')


def _child_order(xml: str) -> list[str]:
    """The bibitem's child-element kinds, in first-occurrence order."""
    import re
    depth = 0
    names: list[str] = []
    for match in re.finditer(r"<(/?)([a-z-]+)[^>]*?(/?)>", xml):
        closing, name, self_closing = match.groups()
        if closing:
            depth -= 1
        else:
            if depth == 1 and name not in names:  # root's children
                names.append(name)
            if not self_closing:
                depth += 1
    return names
