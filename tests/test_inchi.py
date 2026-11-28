import json
import os
from pathlib import Path

import pytest

from asciichem.inchi import (EngineMissingError, Identity, RdkitEngine,
                             get_engine, identity_for, set_engine)
from asciichem import parse_text
from asciichem.smiles import parse_smiles


@pytest.fixture
def ethanol():
    return parse_text("C-C-O").nodes[0]


def test_missing_engine_raises_with_guidance(ethanol):
    previous = get_engine()
    set_engine(None)
    try:
        with pytest.raises(EngineMissingError, match="no InChI engine configured"):
            identity_for(ethanol)
    finally:
        set_engine(previous)


def test_engine_registration_and_identity(ethanol):
    engine = RdkitEngine()
    engine.identity = lambda m: Identity("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                                         "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    try:
        set_engine(engine)
        assert get_engine() is engine
        assert identity_for(ethanol).inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    finally:
        set_engine(None)


def test_engine_passed_per_call_wins(ethanol):
    per_call = RdkitEngine()
    per_call.identity = lambda m: Identity("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
    assert identity_for(ethanol, engine=per_call).inchi.startswith("InChI=1S/C2H6O")


def test_rdkit_engine_missing_extra_message(ethanol):
    try:
        import rdkit  # noqa: F401
        pytest.skip("rdkit installed; missing-extra path not reachable")
    except ImportError:
        pass
    with pytest.raises(EngineMissingError, match="rdkit is not installed"):
        RdkitEngine().identity(ethanol)


def test_rdkit_identity_reads_the_recorded_stereo_identities():
    # The identity pipe's stereo gate: the molfile pipe (the converged
    # geometry-aware writer) through RDKit must derive the recorded
    # oracle identities for the stereo rows — the same catch the rs
    # battery runs through chemomer (the RDKit dialect strips the
    # parity/H-count columns, both oracle-ignored for 2D input; the
    # wedge carries the layer).
    pytest.importorskip("rdkit")
    corpus_dir = os.environ.get(
        "ASCIICHEM_CORPUS",
        str(Path(__file__).resolve().parents[2] / "asciichem-tests"),
    )
    fixtures = sorted((Path(corpus_dir) / "corpus" / "fixtures").glob("*.json"))
    if not fixtures:
        pytest.skip("the corpus checkout gates in CI")
    recorded = {}
    for path in fixtures:
        for case in json.loads(path.read_text()):
            if case.get("inchi") and isinstance(case.get("smiles"), str) and "@" in case["smiles"]:
                recorded[case["smiles"]] = case["inchi"]
    assert len(recorded) >= 2, "the corpus stereo identity rows shrank"
    engine = RdkitEngine()
    for smiles, expected in recorded.items():
        molecule = parse_smiles(smiles).nodes[0]
        assert engine.identity(molecule).inchi == expected, smiles
