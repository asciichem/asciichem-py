"""Reaction identity via RInChI (the chemomer engine's RInChI layer,
mirroring asciichem-rs's ``asciichem::rinchi``).

A reaction (reactants + products + agents, each a molecule) gets its
own identifier assembled from the per-component standard InChIs.
Derived and decoded through the chemomer engine — exactly as the
identity layer (:mod:`asciichem.inchi`) derives per-molecule
identifiers — the same oracle-gated native engine, the same
byte-exact posture.

Two directions, one error vocabulary (the engine's own ``ValueError``
— no second error vocabulary):

- **derive** — a reaction's V2000 RXNfile (or an MDL RDfile, the
  ``$DATUM $MFMT`` agent-scan edge) assembles the identity record
  (:class:`Record` — the five lines: identifier, RAuxInfo, the three
  RInChIKey shapes). Every string is engine-verbatim (key headers
  included); the shared corpus records the keys headerless and the
  runners strip. The RDfile edge exports the identifier line only —
  the remaining fields stay ``None``.
- **decode** — a RInChI string (plus its RAuxInfo companion when
  known) reads back into the reaction it identifies
  (:class:`Reaction`): the components carry their identities straight
  from the string (no structures are rebuilt), and the direction flag
  recovers which group held which role.

The engine is opt-in, exactly the inchi module's posture: the chemomer
package imports lazily inside the functions, and a missing package
raises EngineMissingError with install guidance — no silent fallback.

    from asciichem import rinchi
    rinchi.derive_from_rxnfile(rxnfile)        # Record(rinchi=..., ...)
    rinchi.decode(rinchi_string, rauxinfo)     # Reaction(directionality=..., ...)
"""
from __future__ import annotations

from dataclasses import dataclass

from .inchi import EngineMissingError

INSTALL_GUIDE = ("install the chemomer engine package (pip install "
                 "'asciichem[rinchi]') - the pyo3 binding over the "
                 "native reaction-identity engine (chemomer 1.1.0.0+)")


@dataclass
class Component:
    """One decoded component's identity: the standard InChI string,
    its AuxInfo companion (empty on a bare decode), and the InChIKey."""

    inchi: str
    aux: str
    inchikey: str


@dataclass
class Reaction:
    """The decoded reaction — the decode direction's output: the
    components carry their identities straight from the identifier
    string (no structures are rebuilt), and the directionality
    (``"directional"`` / ``"equilibrium"``) recovers which group held
    which role."""

    directionality: str
    reactants: list[Component]
    products: list[Component]
    agents: list[Component]


@dataclass
class Record:
    """The assembled reaction-identity record — the derive direction's
    output (mirrors asciichem::rinchi's ``RinchiRecord``): the
    identifier, its RAuxInfo companion line, and the three RInChIKey
    shapes (the short hashed one under ``rinchikey``, per the shared
    corpus's field naming). Every string is engine-verbatim — the key
    headers included; the corpus records the keys headerless and the
    runners strip."""

    rinchi: str
    rauxinfo: str | None = None
    rinchikey: str | None = None
    long_key: str | None = None
    web_key: str | None = None


def _engine():
    try:
        import chemomer
    except ImportError as e:
        raise EngineMissingError(f"chemomer is not installed - {INSTALL_GUIDE}") from e
    return chemomer


def derive_from_rxnfile(rxnfile: str) -> Record:
    """Assembles a reaction's identity record from its V2000 RXNfile
    text — the five engine-verbatim lines."""
    chemomer = _engine()
    return Record(
        rinchi=chemomer.rinchi_from_rxn(rxnfile),
        rauxinfo=chemomer.rinchi_aux_from_rxn(rxnfile),
        rinchikey=chemomer.rinchi_short_key_from_rxn(rxnfile),
        long_key=chemomer.rinchi_long_key_from_rxn(rxnfile),
        web_key=chemomer.rinchi_web_key_from_rxn(rxnfile),
    )


def derive_from_rdfile(rdfile: str) -> Record:
    """Assembles the identity record from an MDL RDfile (the RXN block
    verbatim plus the ``$DATUM $MFMT`` agent scan). The pyo3 transport
    exports the identifier line only for this edge (the other four
    ride the RXN exports' assembly under the engine's own gate), so
    the remaining fields stay ``None`` — the record carries what the
    transport provides."""
    return Record(rinchi=_engine().rinchi_from_rd(rdfile))


def decode(rinchi: str, rauxinfo: str = "") -> Reaction:
    """Decodes a RInChI string (plus its RAuxInfo companion when
    known; pass ``""`` when unknown) into the reaction it identifies —
    the components carry their identities (and aux/key content)
    straight from the strings, and the direction flag recovers the
    real roles."""
    native = _engine().rinchi_decode(rinchi, rauxinfo)
    return Reaction(
        directionality=native["directionality"],
        reactants=[Component(**entry) for entry in native["reactants"]],
        products=[Component(**entry) for entry in native["products"]],
        agents=[Component(**entry) for entry in native["agents"]],
    )
