"""Molfile (CTfile V2000) parser + writer - the Python mirror of
AsciiChem::Molfile: coordinates preserved (x2/y2/z2), charges via
M  CHG, isotopes via M  ISO, bond type 4 is aromatic and marks its
atoms, stereo codes 1/6 become wedge/hash."""
from __future__ import annotations

from .model import Atom, Bond, Molecule, ParseError
from .structure import Edge, build_graph, linearize

_BOND_KINDS = {1: "single", 2: "double", 3: "triple", 4: "aromatic"}
_BOND_TYPES = {"single": 1, "double": 2, "triple": 3, "aromatic": 4,
               "wedge": 1, "hash": 1}
_BOND_STEREO = {"wedge": 1, "hash": 6}


def parse_molfile(text: str) -> Molecule:
    lines = text.replace("\r\n", "\n").split("\n")
    if len(lines) < 5:
        raise ParseError("molfile too short")

    def field(line_index, start, length):
        return lines[line_index][start:start + length].strip() if line_index < len(lines) else ""

    atom_field, bond_field = field(3, 0, 3), field(3, 3, 3)
    if not (atom_field.isdigit() and bond_field.isdigit()):
        raise ParseError(f"malformed counts line: {lines[3]!r}")
    atom_count, bond_count = int(atom_field), int(bond_field)

    atoms: list[Atom] = []
    for i in range(1, atom_count + 1):
        line = lines[3 + i] if 3 + i < len(lines) else None
        if line is None:
            raise ParseError(f"truncated atom block (expected {atom_count} atoms)")
        element = line[31:34].strip()
        if not element:
            raise ParseError(f"atom {i} has no element symbol")
        atoms.append(Atom(element=element,
                          x2=float(line[0:10]), y2=float(line[10:20]),
                          z2=float(line[20:30])))

    bonds = []
    for i in range(1, bond_count + 1):
        line = lines[3 + atom_count + i] if 3 + atom_count + i < len(lines) else None
        if line is None:
            raise ParseError(f"truncated bond block (expected {bond_count} bonds)")
        a, b = int(line[0:3]) - 1, int(line[3:6]) - 1
        bond_type, stereo = int(line[6:9]), int(line[9:12])
        if not (0 <= a < atom_count and 0 <= b < atom_count):
            raise ParseError(f"bond {i} has out-of-range atom indexes")
        if stereo == 1:
            kind = "wedge"
        elif stereo == 6:
            kind = "hash"
        elif bond_type in _BOND_KINDS:
            kind = _BOND_KINDS[bond_type]
        else:
            raise ParseError(f"unsupported molfile bond type {bond_type}")
        bonds.append((a, b, kind))

    _apply_properties(lines, atoms)

    # V2000 carries aromaticity on bonds; the model carries it on
    # atoms and bonds, so atoms touching an aromatic bond are marked.
    for a, b, kind in bonds:
        if kind == "aromatic":
            atoms[a].aromatic = True
            atoms[b].aromatic = True

    edges = [Edge(min(a, b), max(a, b), kind) for a, b, kind in bonds]
    return Molecule(nodes=linearize(atoms, edges))


def _apply_properties(lines, atoms):
    for line in lines:
        if line.startswith("M  CHG") or line.startswith("M  ISO"):
            is_charge = line.startswith("M  CHG")
            parts = line[6:].split()
            for i in range(1, len(parts) - 1, 2):
                index, value = int(parts[i]) - 1, int(parts[i + 1])
                if is_charge:
                    sign = "-" if value < 0 else "+"
                    magnitude = abs(value)
                    atoms[index].charge = (sign if magnitude == 1
                                           else f"{magnitude}{sign}")
                else:
                    atoms[index].isotope = str(value)


def write_molfile(molecule: Molecule, name: str = "") -> str:
    atoms, edges = build_graph(molecule)
    if not edges and len(atoms) > 1:
        raise ParseError("molecule has no bonds - a formula is not a structure")

    lines = [name, "  AsciiChem", "",
             f"{len(atoms):3d}{len(edges):3d}  0  0  0  0  0  0  0  0999 V2000"]

    for atom in atoms:
        lines.append(f"{atom.x2 or 0.0:10.4f}{atom.y2 or 0.0:10.4f}"
                     f"{atom.z2 or 0.0:10.4f} {atom.element:<3}"
                     " 0  0  0  0  0  0  0  0  0  0  0  0")

    for edge in edges:
        bond_type = _BOND_TYPES.get(edge.kind)
        if bond_type is None:
            raise ParseError(f"{edge.kind} bonds have no molfile V2000 type")
        stereo = _BOND_STEREO.get(edge.kind, 0)
        lines.append(f"{edge.from_ + 1:3d}{edge.to + 1:3d}{bond_type:3d}"
                     f"{stereo:3d}  0  0  0  0  0  0  0")

    charges = [(i + 1, _charge_value(a.charge))
               for i, a in enumerate(atoms) if a.charge is not None]
    isotopes = [(i + 1, int(a.isotope))
                for i, a in enumerate(atoms) if a.isotope is not None]
    lines.append(_property_line("M  CHG", charges))
    lines.append(_property_line("M  ISO", isotopes))
    lines.append("M  END")
    return "\n".join(l for l in lines if l is not None) + "\n"


def _charge_value(charge: str) -> int:
    magnitude = int(charge[:-1]) if charge[:-1].isdigit() else 1
    return -magnitude if charge.endswith("-") else magnitude


def _property_line(prefix, pairs):
    if not pairs:
        return None
    line = f"{prefix}{len(pairs):3d}"
    for index, value in pairs:
        line += f"{index:4d}{value:4d}"
    return line
