"""Shared-corpus conformance: the CLI family (suite 0.11.0), in the
runner contract's pending posture. Each case runs `invocation`
(the argv after the program name) against the implementation's CLI
dispatch and asserts exit plus the stdout/stderr matchers.

asciichem-py ships no CLI yet - the corpus expects the offline
command set (convert, validate, lint, roundtrip, parse-cml, version,
tree). Per the runner contract, `pending` implementations skip and
report - never fail silently: every case below reports a skip naming
the missing capability, so the family's absence stays visible in the
conformance output until the CLI lands."""
import json
import os
from pathlib import Path

import pytest

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

pytestmark = pytest.mark.skipif(
    not (CORPUS / "corpus" / "fixtures").is_dir(),
    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")


def cli_fixtures():
    if not (CORPUS / "corpus" / "fixtures").is_dir():
        return []  # corpus absent entirely - the skip mark owns this
    path = CORPUS / "corpus" / "fixtures" / "cli.json"
    if not path.is_file():
        pytest.fail("cli.json missing from the corpus - the pin does "
                    "not carry suite 0.11.0")
    return json.loads(path.read_text())


@pytest.mark.parametrize("case", cli_fixtures(), ids=lambda c: c["id"])
def test_cli_invocation(case):
    pytest.skip("asciichem-py ships no CLI yet (pending) - the offline "
                "command set is a follow-up convergence slice")
