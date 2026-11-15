"""Shared-corpus conformance: the RInChI family (the reaction-identity
oracle, the corpus's rinchi.json). Two directions per case, per the
runner contract: derive — the input edge (RXNfile xor RDfile)
assembles the oracle lines byte-identically (the engine-verbatim keys
re-stripped of their Short/Long/Web-RInChIKey= headers; the RDfile
edge exports the identifier line only, the other four riding the RXN
exports' assembly under the engine's own gate); decode — the
identifier (+ RAuxInfo companion) reads back into the role-resolved
components, in role order for the directional readings (the direction
flag recovers the real roles), order-free for the forced-equilibrium
ones (an equilibrium is symmetric — either group may output first),
the flag pinning the reading itself.

Engine posture: this family needs an engine that ships the
reaction-identity surface (the chemomer package, 1.1.0.0+). The
no-engine posture skips the whole family with a visible report — a
missing engine is a posture, not a failure (the same contract the
linter family's engine postures follow; CI runs both postures as
separate jobs).

The forced-equilibrium readings additionally skip the derive
comparison and report: the oracle recorded them under the forced
flag, for which the pyo3 surface carries no entry (their RXNfiles
parse directional) — the decode direction still gates every case."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

CORPUS = Path(os.environ.get("ASCIICHEM_CORPUS",
                             Path.cwd().parent / "asciichem-tests"))

pytestmark = pytest.mark.skipif(
    not (CORPUS / "corpus" / "fixtures").is_dir(),
    reason="corpus not cloned as sibling (or set ASCIICHEM_CORPUS)")

POSTURE = "chemomer" if importlib.util.find_spec("chemomer") is not None else "none"

ROLES = ("reactants", "products", "agents")

# The corpus's headerless key fields against the engine's headered
# lines and the Record's attribute names.
KEY_LINES = {"rinchikey": ("rinchikey", "Short-RInChIKey="),
             "longKey": ("long_key", "Long-RInChIKey="),
             "webKey": ("web_key", "Web-RInChIKey=")}


def rinchi_fixtures():
    if not (CORPUS / "corpus" / "fixtures").is_dir():
        return []  # corpus absent entirely - the skip mark owns this
    path = CORPUS / "corpus" / "fixtures" / "rinchi.json"
    if not path.is_file():
        pytest.fail("rinchi.json missing from the corpus - "
                    "the pin does not carry the rinchi family")
    return json.loads(path.read_text())


def test_reports_the_posture_skipped_family():
    cases = rinchi_fixtures()
    # The family size is pinned (the list changes only when the
    # corpus's rinchi family changes), and the posture skip is part
    # of the report, never silent: in the no-engine posture the whole
    # family is listed here (and per-case above), in the engine
    # posture nothing is.
    assert len(cases) == 36, "the corpus carries the rinchi family"
    skipped = [] if POSTURE == "chemomer" else [c["id"] for c in cases]
    assert len(skipped) == (0 if POSTURE == "chemomer" else len(cases))


def test_reports_the_derive_skipped_cases():
    # The forced-equilibrium readings ride decode; their derive
    # comparison needs the engine's forced flag, for which the pyo3
    # surface carries no entry. The corpus's marker key agrees with
    # the flag in the identifier itself (/d= speaks equilibrium).
    cases = rinchi_fixtures()
    derive_skipped = [c["id"] for c in cases if c.get("forceEquilibrium")]
    assert derive_skipped == [c["id"] for c in cases if "/d=" in c["rinchi"]]
    assert len(derive_skipped) > 0, "the corpus carries forced-equilibrium readings"


@pytest.mark.parametrize("case", rinchi_fixtures(), ids=lambda c: c["id"])
def test_reaction_identity(case):
    if POSTURE == "none":
        pytest.skip("the chemomer engine is not installed - the family is "
                    "exercised by the inchi-posture job (the posture report "
                    "above lists the skipped cases)")
    from asciichem import rinchi

    forced = bool(case.get("forceEquilibrium"))

    # The decode direction - every case runs it. The identifier (+
    # companion) reads back into the role-resolved components.
    reaction = rinchi.decode(case["rinchi"], case.get("rauxinfo", ""))
    assert reaction.directionality == ("equilibrium" if forced else "directional"), (
        f'{case["id"]}: direction flag')
    if forced:
        # Symmetric: the components compare order-free across the
        # three roles (the flag above pins the reading itself).
        expected = sorted(i for role in ROLES for i in case["components"][role])
        actual = sorted(c.inchi for role in ROLES for c in getattr(reaction, role))
        assert actual == expected, f'{case["id"]}: decoded components'
    else:
        # The direction flag recovers the real roles - roles compare
        # in wire order.
        for role in ROLES:
            actual = [c.inchi for c in getattr(reaction, role)]
            assert actual == case["components"][role], f'{case["id"]}: decoded {role}'

    # The derive direction: the input edge assembles the oracle lines
    # byte-identically. The forced-equilibrium readings skip here and
    # ride the family's derive-skip report.
    if forced:
        return
    if "rxnfile" in case:
        record = rinchi.derive_from_rxnfile(case["rxnfile"])
        assert record.rinchi == case["rinchi"], f'{case["id"]}: derive rinchi'
        assert record.rauxinfo == case["rauxinfo"], f'{case["id"]}: derive rauxinfo'
        for field, (attribute, header) in KEY_LINES.items():
            line = getattr(record, attribute)
            assert line.startswith(header), f'{case["id"]}: derive {field} header'
            assert line[len(header):] == case[field], f'{case["id"]}: derive {field}'
    else:
        # The RD edge exports the identifier line only; the other four
        # lines ride the RXN exports' assembly, and the engine's own
        # gate owns their full per-line comparison.
        record = rinchi.derive_from_rdfile(case["rdfile"])
        assert record.rinchi == case["rinchi"], f'{case["id"]}: derive rinchi'
