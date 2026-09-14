"""Local identity derivation (TODO.v2 10; TODO.impl 48).

The InChI algorithm is never reimplemented: engines wrap an InChI
implementation — RDKit's (which uses the IUPAC library underneath)
via the `asciichem[inchi]` extra here. Engines are opt-in: without
one, derivation raises EngineMissingError with guidance; no silent
fallback.

    from asciichem.inchi import RdkitEngine, set_engine
    set_engine(RdkitEngine())
    identity_for(molecule)     # Identity(inchi=..., inchikey=...)
"""
from __future__ import annotations

from typing import Optional

from .model import AsciiChemError, Molecule

INSTALL_GUIDE = ("install the InChI engine extra (pip install "
                 "'asciichem[inchi]') and register it via "
                 "asciichem.inchi.set_engine(RdkitEngine())")


class EngineMissingError(AsciiChemError):
    pass


class Identity:
    def __init__(self, inchi: str, inchikey: Optional[str] = None):
        self.inchi = inchi
        self.inchikey = inchikey

    def __eq__(self, other):
        return (isinstance(other, Identity) and
                self.inchi == other.inchi and self.inchikey == other.inchikey)

    def __repr__(self):
        return f"Identity(inchi={self.inchi!r}, inchikey={self.inchikey!r})"


_engine = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def get_engine():
    return _engine


def identity_for(molecule: Molecule, engine=None) -> Identity:
    chosen = engine if engine is not None else _engine
    if chosen is None:
        raise EngineMissingError(f"no InChI engine configured - {INSTALL_GUIDE}")
    return chosen.identity(molecule)


class RdkitEngine:
    """Engine over RDKit's InChI support (asciichem[inchi] extra).
    Pipe: molecule -> molfile -> rdkit -> standard InChI/InChIKey."""

    def identity(self, molecule: Molecule) -> Identity:
        try:
            from rdkit import Chem
        except ImportError as e:
            raise EngineMissingError(f"rdkit is not installed - {INSTALL_GUIDE}") from e

        from .molfile import write_molfile

        molblock = write_molfile(molecule)
        mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
        if mol is None:
            raise AsciiChemError("rdkit could not parse the molfile for this molecule")
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)
        return Identity(inchi=inchi, inchikey=inchikey or None)
