"""The chemomer transport's unit behaviors (asciichem.rinchi): the
opt-in posture (a missing package raises with guidance), the derive
direction's five engine-verbatim lines over the recorded oracle bytes
(the corpus's water-is-water recording), the RDfile edge's
identifier-only export, and the decode direction's role resolution.
The full 36-case byte-exact gate lives in the corpus family
(tests/conformance/test_rinchi_corpus.py), exercised in the engine
posture job."""
import importlib.util

import pytest

from asciichem import rinchi
from asciichem.inchi import EngineMissingError


def _has_chemomer() -> bool:
    return importlib.util.find_spec("chemomer") is not None


requires_engine = pytest.mark.skipif(
    not _has_chemomer(),
    reason="chemomer not installed - the pyo3 reaction-identity engine is opt-in")

# The corpus's water-is-water recording (asciichem-tests, rinchi.json):
# one water drawn each side. The recorded oracle bytes - no mocks.
WATER_RXN = (
    "$RXN\nwater-is-water\n  chemomer recording\n\n  1  1\n"
    "$MOL\n\n  AsciiChem\n\n"
    "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
    "   12.0000   12.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "M  END\n"
    "$MOL\n\n  AsciiChem\n\n"
    "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
    "   12.0000   12.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "M  END\n"
)
WATER_RAUXINFO_GROUP = "0/N:1/rA:1nO/rB:/rC:12,12,0;"
WATER_RAUXINFO = f"RAuxInfo=1.00.1/{WATER_RAUXINFO_GROUP}<>{WATER_RAUXINFO_GROUP}"

# The corpus's equil-water-is-water recording: the same reaction the
# oracle recorded under the forced flag - which the pyo3 surface
# carries no derive entry for (its RXNfile parses directional).
EQUIL_WATER_RXN = WATER_RXN.replace("water-is-water", "equil-water-is-water")
EQUIL_WATER_RINCHI = "RInChI=1.00.1S/H2O/h1H2<>H2O/h1H2/d="

# The corpus's rd-chlorination-no-agents recording: the RDfile edge
# (the RXN block verbatim plus the $DATUM $MFMT agent scan).
RD_CHLORINATION = (
    "$RDFILE 1\n$DATM 2026-11-03 12:00:00\n$RFMT $RIREG 42\n"
    "$RXN\nrd-chlorination-no-agents\n  chemomer recording\n\n  2  2\n"
    "$MOL\n\n  AsciiChem\n\n"
    "  6  6  0  0  0  0  0  0  0  0999 V2000\n"
    "   12.0000   18.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   72.0000   23.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  132.0000   23.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  192.0000   23.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  252.0000   23.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  312.0000   18.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  4  0  0  0  0  0  0  0  0\n"
    "  2  3  4  0  0  0  0  0  0  0  0\n"
    "  3  4  4  0  0  0  0  0  0  0  0\n"
    "  4  5  4  0  0  0  0  0  0  0  0\n"
    "  5  6  4  0  0  0  0  0  0  0  0\n"
    "  1  6  4  0  0  0  0  0  0  0  0\n"
    "M  END\n"
    "$MOL\nchlorine\n  chemomer\n\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "   12.0000   12.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   24.0000   12.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0\n"
    "M  END\n"
    "$MOL\nchlorobenzene\n  chemomer\n\n"
    "  7  7  0  0  0  0  0  0  0  0999 V2000\n"
    "   12.0000   12.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   24.0000   12.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   36.0000   12.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   48.0000   12.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   60.0000   12.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   72.0000   12.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   84.0000   12.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  2  0\n"
    "  2  3  1  0\n"
    "  3  4  2  0\n"
    "  4  5  1  0\n"
    "  5  6  2  0\n"
    "  6  1  1  0\n"
    "  2  7  1  0\n"
    "M  END\n"
    "$MOL\nhydrogen chloride\n  chemomer\n\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "   12.0000   12.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "   24.0000   12.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0\n"
    "M  END\n"
)
RD_CHLORINATION_RINCHI = (
    "RInChI=1.00.1S/C6H5Cl/c7-6-4-2-1-3-5-6/h1-5H!ClH/h1H"
    "<>C6H6/c1-2-4-6-5-3-1/h1-6H!Cl2/c1-2/d-"
)


def test_missing_package_raises_with_guidance():
    if _has_chemomer():
        pytest.skip("chemomer installed; missing-package path not reachable")
    with pytest.raises(EngineMissingError, match="chemomer is not installed"):
        rinchi.derive_from_rxnfile(WATER_RXN)


@requires_engine
def test_an_rxnfile_derives_the_recorded_five_lines():
    record = rinchi.derive_from_rxnfile(WATER_RXN)
    assert record.rinchi == "RInChI=1.00.1S/H2O/h1H2<>H2O/h1H2/d+"
    assert record.rauxinfo == WATER_RAUXINFO
    # The keys are engine-verbatim - the headers included (the corpus
    # records them headerless and the runners strip).
    assert (record.rinchikey ==
            "Short-RInChIKey=SA-FUHFF-XLYOFNOQVP-XLYOFNOQVP-UHFFFADPSC"
            "-NUHFF-NUHFF-NUHFF-ZZZ")
    assert (record.long_key ==
            "Long-RInChIKey=SA-FUHFF-XLYOFNOQVPJJNP-UHFFFAOYSA-N--"
            "XLYOFNOQVPJJNP-UHFFFAOYSA-N")
    assert record.web_key == "Web-RInChIKey=XLYOFNOQVPJJNPIXP-NUHFFFADPSCTJSA"


@requires_engine
def test_an_rdfile_derives_the_identifier_line_only():
    record = rinchi.derive_from_rdfile(RD_CHLORINATION)
    assert record.rinchi == RD_CHLORINATION_RINCHI
    # The pyo3 transport exports the identifier line only for this
    # edge - the remaining fields carry what the transport provides.
    assert record.rauxinfo is None
    assert record.rinchikey is None
    assert record.long_key is None
    assert record.web_key is None


@requires_engine
def test_the_forced_equilibrium_reading_is_not_parsed_from_the_file():
    # The RXNfile carries no direction field - the derive reads it
    # directional (/d+); the corpus's forced reading (/d=) needs the
    # engine's forced flag, which the pyo3 surface carries no entry
    # for - the corpus family gates those four cases through decode.
    record = rinchi.derive_from_rxnfile(EQUIL_WATER_RXN)
    assert record.rinchi == "RInChI=1.00.1S/H2O/h1H2<>H2O/h1H2/d+"
    assert record.rinchi != EQUIL_WATER_RINCHI


@requires_engine
def test_a_rinchi_string_decodes_into_its_roles():
    # A reversed recording (/d-: the products group sorts first): the
    # reader recovers the real roles from the flag.
    reaction = rinchi.decode("RInChI=1.00.1S/<>ClH/h1H!H3N/h1H3/d-")
    assert reaction.directionality == "directional"
    assert [c.inchi for c in reaction.reactants] == ["InChI=1S/ClH/h1H",
                                                     "InChI=1S/H3N/h1H3"]
    assert reaction.products == []
    assert reaction.agents == []

    # The equilibrium flag reads equilibrium.
    assert rinchi.decode(EQUIL_WATER_RINCHI).directionality == "equilibrium"


@requires_engine
def test_decode_with_the_companion_distributes_the_aux_bodies():
    reaction = rinchi.decode("RInChI=1.00.1S/H2O/h1H2<>H2O/h1H2/d+",
                             WATER_RAUXINFO)
    aux = "AuxInfo=1/" + WATER_RAUXINFO_GROUP
    assert reaction.reactants[0].inchi == "InChI=1S/H2O/h1H2"
    assert reaction.reactants[0].aux == aux
    assert reaction.reactants[0].inchikey == "XLYOFNOQVPJJNP-UHFFFAOYSA-N"
    assert reaction.products[0].aux == aux

    # The bare decode (no companion at hand) leaves the aux empty.
    bare = rinchi.decode("RInChI=1.00.1S/H2O/h1H2<>H2O/h1H2/d+")
    assert bare.reactants[0].aux == ""


@requires_engine
def test_garbage_is_rejected_through_the_engine_error():
    with pytest.raises(ValueError, match="rxn"):
        rinchi.decode("RInChI=0.9/ nonsense")
    with pytest.raises(ValueError, match=r"\$RXN"):
        rinchi.derive_from_rxnfile("not an rxnfile")
