"""Molfile (CTfile V2000) parser + writer - the Python mirror of
AsciiChem::Molfile: coordinates preserved (x2/y2/z2), charges via
M  CHG, isotopes via M  ISO, bond type 4 is aromatic and marks its
atoms, stereo codes 1/6 become wedge/hash."""
from __future__ import annotations

import math

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
    model_atoms, model_edges = build_graph(molecule)
    if not model_edges and len(model_atoms) > 1:
        raise ParseError("molecule has no bonds - a formula is not a structure")

    # The deterministic layout backs authored coordinates: achiral
    # identities are coordinate-blind, but a chiral model's wedge only
    # reads as stereo against non-degenerate drawn positions, so the
    # emission lays the graph out (the same shape the recording
    # pipeline writes).
    from .layout import place_graph, walk_molecule

    positions = place_graph(walk_molecule(molecule))

    # Aromatic carriers lose their bracket H in the identity pipe: the
    # V2000 H-count column is oracle-ignored (the release engine never
    # reads it) and the valence derivation does not hand aromatic atoms
    # their H - a pyrrole's [nH] reads one hydrogen short (the Ruby
    # transport loss, asciichem-ruby#181). The explicit H atom is the
    # transport that survives, the same posture as [2H]. Non-aromatic
    # carriers are untouched: their H derive by valence everywhere, and
    # the recorded corpus molfiles pin their bytes.
    atoms = list(model_atoms)
    edges = list(model_edges)
    h_carriers: dict[int, tuple[int, int]] = {}
    base_atoms = len(model_atoms)
    for index, atom in enumerate(model_atoms):
        if not atom.aromatic or not atom.hydrogens:
            continue
        for nth in range(atom.hydrogens):
            target = len(atoms)
            h_carriers[target] = (index, nth)
            atoms.append(Atom(element="H"))
            edges.append(Edge(from_=index, to=target, kind="single"))

    lines = [name, "  AsciiChem", "",
             f"{len(atoms):3d}{len(edges):3d}  0  0  0  0  0  0  0  0999 V2000"]

    coords: list[tuple[float, float]] = []
    for index, atom in enumerate(atoms):
        carrier = h_carriers.get(index)
        if carrier is not None:
            source = model_atoms[carrier[0]]
            x = (source.x2 or 0.0) + 0.75
            y = (source.y2 or 0.0) + 0.75 * carrier[1]
        else:
            placed = positions[index] if index < len(positions) else None
            x = atom.x2 or (placed.x if placed else 0.0)
            y = atom.y2 or (placed.y if placed else 0.0)
        coords.append((x, y))

    # One wedge per chiral atom, on its first bond, oriented
    # centre -> neighbour: TH1 rises (stereo 1), TH2 falls (6). The
    # V2000 atom-parity field is oracle-ignored for 2D input - the
    # wedge is what carries the tetrahedral layer.
    # Directional bonds draw the E/Z layer: a direction rides a single
    # bond adjacent to a double bond, and the substituent's SIDE of that
    # double bond is the drawn information. The layout's arbitrary
    # placement carries no side semantics (the zigzag draws chains
    # near-collinear), so each directional substituent is placed
    # perpendicular to the double-bond axis, anchored at its bond
    # partner, with the side read against the STRING-ORDERED axis (a to
    # b) - the chemomer lift's invariant, so ``C/C=C/C`` draws trans
    # and ``C/C=C\\C`` cis regardless of which end the axis runs from.
    for edge in edges:
        if edge.direction is None:
            continue
        resolved = _double_bond_axis(edges, edge)
        if resolved is None:
            continue
        axis, p_index, u_index = resolved
        q_index = axis.to if axis.from_ == p_index else axis.from_
        a_index, b_index = min(p_index, q_index), max(p_index, q_index)
        ax, ay = coords[a_index]
        bx, by = coords[b_index]
        px, py = coords[p_index]
        dx, dy = bx - ax, by - ay
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            continue
        nx, ny = -dy / length, dx / length
        side = (1.0 if edge.direction == "up" else -1.0) * \
            (1.0 if u_index > a_index else -1.0)
        coords[u_index] = (px + nx * length * side, py + ny * length * side)

    wedges = {}
    for index, atom in enumerate(atoms):
        if atom.chirality not in ("TH1", "TH2"):
            continue
        # The chiral atom's heavy ligands sit in a trigonal triangle
        # around it (clockwise-ordered list, the chemomer synthesis's
        # corpus-fitted shape), and the wedge code pairs with that
        # handedness: TH1 falls (6), TH2 rises (1). The base layout's
        # zigzag can collide ligand positions, and a degenerate
        # drawing reads no parity at all, so the stereocentre's
        # neighbourhood is synthesized deterministically.
        ligand_bonds = [(i, e) for i, e in enumerate(edges)
                        if e.from_ == index or e.to == index]
        ligands = [e.to if e.from_ == index else e.from_
                   for _, e in ligand_bonds]
        cx, cy = coords[index]
        for slot, ligand in enumerate(ligands[:3]):
            angle = math.radians(90 + 120 * slot)
            coords[ligand] = (cx + 28 * math.cos(angle),
                              cy + 28 * math.sin(angle))
        bond_index = ligand_bonds[0][0]
        edge = edges[bond_index]
        other = edge.to if edge.from_ == index else edge.from_
        wedges[bond_index] = (index, other, 6 if atom.chirality == "TH1" else 1)

    for index, atom in enumerate(atoms):
        x, y = coords[index]
        z = atom.z2 or 0.0
        lines.append(f"{x:10.4f}{y:10.4f}"
                     f"{z:10.4f} {atom.element:<3}"
                     " 0  0  0  0  0  0  0  0  0  0  0  0")

    for bond_index, edge in enumerate(edges):
        bond_type = _BOND_TYPES.get(edge.kind)
        if bond_type is None:
            raise ParseError(f"{edge.kind} bonds have no molfile V2000 type")
        stereo = _BOND_STEREO.get(edge.kind, 0)
        wedge = wedges.get(bond_index)
        if wedge is not None:
            from_, to, stereo = wedge
        else:
            from_, to = edge.from_, edge.to
        lines.append(f"{from_ + 1:3d}{to + 1:3d}{bond_type:3d}"
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


def _double_bond_axis(edges, single):
    """The double-bond axis a directional bond borders: which end of
    the single bond touches a kind-2 bond's atom, with that endpoint,
    the substituent (the other end), and the double-bond edge."""
    for atom, other in ((single.from_, single.to), (single.to, single.from_)):
        for candidate in edges:
            if candidate.kind == "double" and (
                (candidate.from_ == atom and candidate.to != other)
                or (candidate.to == atom and candidate.from_ != other)
            ):
                substituent = single.to if single.from_ == atom else single.from_
                return candidate, atom, substituent
    return None
