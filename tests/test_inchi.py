import pytest

from asciichem.inchi import (EngineMissingError, Identity, RdkitEngine,
                             get_engine, identity_for, set_engine)
from asciichem.parser import parse_text


@pytest.fixture
def ethanol():
    return parse_text("C-C-O").nodes[0]


def test_missing_engine_raises_with_guidance(ethanol):
    with pytest.raises(EngineMissingError, match="no InChI engine configured"):
        identity_for(ethanol)


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
