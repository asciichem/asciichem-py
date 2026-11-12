"""Shared-corpus conformance: the CLI family (suite 0.11.0). Each
case runs `invocation` (the argv after the program name) against the
implementation's CLI dispatch - in-process through
`asciichem.cli.run` (no subprocesses) - and asserts `exit` plus the
stdout/stderr matchers: `stdout` exact (with `{version}` standing for
this implementation's own version), `stdoutIncludes` (every entry
appears), `stdoutJson` (semantic equality with the parsed stdout
JSON), `stderrIncludes` (every entry appears in stderr). Exactly one
stdout matcher per case.

Cases marked `pending` listing this implementation skip and report -
never fail silently."""
import io
import json
import os
from pathlib import Path

import pytest

import asciichem
from asciichem import cli

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
    pending = case.get("pending") or []
    if "asciichem-py" in pending:
        pytest.skip(f'{case["id"]} is pending for asciichem-py in the '
                    "corpus - the implementation lacks the marked "
                    "capability; the case is exercised when the mark "
                    "is lifted")
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(list(case["invocation"]), out=out, err=err)
    stdout, stderr = out.getvalue(), err.getvalue()
    context = (f'{case["id"]}: exit {code}, stdout {stdout!r}, '
               f"stderr {stderr!r}")
    assert code == case["exit"], context
    if "stdout" in case:
        expected = case["stdout"].replace("{version}",
                                          asciichem.__version__)
        assert stdout == expected, context
    elif "stdoutIncludes" in case:
        for entry in case["stdoutIncludes"]:
            assert entry in stdout, f"{context}; missing {entry!r}"
    elif "stdoutJson" in case:
        assert json.loads(stdout) == case["stdoutJson"], context
    for entry in case.get("stderrIncludes", []):
        assert entry in stderr, f"{context}; stderr missing {entry!r}"
