"""SMILES ingestion and emission - the Python mirror of
AsciiChem::Smiles. v1 subset deferrals (actionable ParseErrors):
chirality @/@@, E/Z bond directions / \\, wildcards, bonded ring
closures the digit form cannot carry. The writer is deterministic
(DFS tree, single-bond continuations, order-independent token
tie-breaks) - aspirin and naphthalene round-trip exactly."""
from __future__ import annotations

import re

from .model import Atom, Bond, Formula, Molecule, ParseError
from .structure import Edge, build_graph, linearize

ORGANIC = ("Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I")
AROMATIC_TWO = ("se", "as")
AROMATIC_ONE = "bcnops"
BOND_KINDS = {"-": "single", "=": "double", "#": "triple", "$": "quadruple",
              ":": "aromatic"}
LOWERABLE_AROMATIC = {"C", "N", "O", "S", "P", "B", "Se", "As"}
WRITER_ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
CONTINUATION_BONDS = {"single", "aromatic"}


class SmilesParser:
    def __init__(self, source: str):
        self.s = source
        self.pos = 0
        self.atoms: list[Atom] = []
        self.adjacency: list[dict[int, str]] = []
        self.open_rings: dict[str, int] = {}
        self.parent: int | None = None
        self.pending_kind: str | None = None

    def parse(self) -> Formula:
        molecules = [self.component()]
        while self._eat("."):
            molecules.append(self.component())
        if self.pos != len(self.s):
            raise ParseError(
                f"unexpected character {self.s[self.pos]!r} at position "
                f"{self.pos} in {self.s!r}")
        return Formula(molecules)

    # -- component -----------------------------------------------------

    def component(self) -> Molecule:
        self.atoms = []
        self.adjacency = []
        self.open_rings = {}
        self.parent = None
        self.pending_kind = None

        self.chain()

        if self.open_rings:
            digits = ", ".join(sorted(self.open_rings))
            raise ParseError(
                f"unclosed ring bond digit(s): {digits} in {self.s!r}")

        edges = [Edge(a, b, kind)
                 for a, neighbors in enumerate(self.adjacency)
                 for b, kind in neighbors.items() if b > a]
        return Molecule(nodes=linearize(self.atoms, edges))

    def chain(self):
        self.atom_token()
        while not self._eoc():
            ch = self._peek()
            if ch == "(":
                self.branch(None)
            elif self._bond_start():
                kind = self.bond_token()
                if self._peek() == "(":
                    self.branch(kind)
                elif self._digit_start():
                    self.ringbond(kind)
                else:
                    self.pending_kind = kind
                    self.atom_token()
            elif self._digit_start():
                self.ringbond(None)
            else:
                self.pending_kind = None
                self.atom_token()

    def branch(self, kind: str | None):
        self._expect("(")
        if kind is None and self._bond_start():
            kind = self.bond_token()
        saved_parent, saved_pending = self.parent, self.pending_kind
        self.pending_kind = kind
        self.chain()
        self._expect(")")
        self.parent, self.pending_kind = saved_parent, saved_pending

    def ringbond(self, kind: str | None):
        digit = self.ring_digit()
        opener = self.open_rings.pop(digit, None)
        if opener is not None:
            if kind and kind != self._default_kind(opener, self.parent):
                raise ParseError(
                    f"bonded ring closures ({kind}) are not representable "
                    "in the model's ring-closure form")
            self._add_edge(opener, self.parent, kind)
        else:
            self.open_rings[digit] = self.parent

    def atom_token(self):
        atom = self.read_atom()
        index = len(self.atoms)
        self.atoms.append(atom)
        self.adjacency.append({})
        if self.parent is not None:
            self._add_edge(self.parent, index, self.pending_kind)
        self.parent = index
        self.pending_kind = None

    def _add_edge(self, a: int, b: int, explicit: str | None):
        kind = explicit or self._default_kind(a, b)
        self.adjacency[a][b] = kind
        self.adjacency[b][a] = kind

    def _default_kind(self, a: int, b: int) -> str:
        both = self.atoms[a].aromatic and self.atoms[b].aromatic
        return "aromatic" if both else "single"

    # -- tokens ----------------------------------------------------------

    def read_atom(self) -> Atom:
        if self._peek() == "[":
            return self.bracket_atom()
        two = self.s[self.pos:self.pos + 2]
        for symbol in ORGANIC:
            if two.startswith(symbol):
                self.pos += len(symbol)
                return Atom(element=symbol)
        for symbol in AROMATIC_TWO:
            if two.startswith(symbol):
                self.pos += 2
                return Atom(element=_capitalize(symbol), aromatic=True)
        peeked = self._peek()
        if peeked is not None and peeked in AROMATIC_ONE:
            symbol = self._peek()
            self.pos += 1
            return Atom(element=_capitalize(symbol), aromatic=True)
        raise ParseError(
            f"unexpected character {self._peek()!r} at position {self.pos} "
            f"in {self.s!r}")

    def bracket_atom(self) -> Atom:
        self._expect("[")
        isotope = self._digits()
        symbol = self.bracket_symbol()
        self.reject_chirality()
        hydrogens = self.hcount()
        charge = self.bracket_charge()
        self.skip_class()
        self._expect("]")
        return Atom(element=_capitalize(symbol), isotope=isotope,
                    charge=charge,
                    aromatic=True if re.match(r"[a-z]", symbol) else None,
                    hydrogens=hydrogens)

    def bracket_symbol(self) -> str:
        two = self.s[self.pos:self.pos + 2]
        if two in AROMATIC_TWO:
            self.pos += 2
            return two
        current = self._peek()
        if current is not None and current in AROMATIC_ONE:
            self.pos += 1
            return current
        if current is not None and current.isupper():
            symbol = current
            if self.pos + 1 < len(self.s) and self.s[self.pos + 1].islower():
                symbol += self.s[self.pos + 1]
            self.pos += len(symbol)
            return symbol
        raise ParseError(
            f"expected an element symbol at position {self.pos} in {self.s!r}")

    def reject_chirality(self):
        if self._peek() != "@":
            return
        token = "@@" if self.s[self.pos:self.pos + 2] == "@@" else "@"
        raise ParseError(f"chirality '{token}' is not supported in the v1 subset")

    def hcount(self):
        if self._peek() != "H":
            return None
        self.pos += 1
        count = self._digits()
        return int(count) if count else 1

    def bracket_charge(self):
        sign = self._peek()
        if sign is None or sign not in "+-":
            return None
        self.pos += 1
        second = self._peek() or ""
        if second == sign:
            self.pos += 1
            count = 2
        elif second.isdigit():
            count = int(self._digits())
        else:
            count = None
        return f"{count}{sign}" if count and count > 0 else sign

    def skip_class(self):
        if self._peek() != ":":
            return
        self.pos += 1
        self._digits()

    def bond_token(self) -> str:
        ch = self._peek()
        kind = BOND_KINDS.get(ch)
        if not kind:
            if ch in "/\\":
                raise ParseError(
                    f"bond direction {ch!r} (E/Z stereo) is not supported "
                    "in the v1 subset")
            raise ParseError(
                f"expected a bond or atom at position {self.pos} in {self.s!r}")
        self.pos += 1
        return kind

    def ring_digit(self) -> str:
        if self._peek() == "%":
            self.pos += 1
            digit = self.s[self.pos:self.pos + 2]
            if not re.fullmatch(r"\d\d", digit):
                raise ParseError("malformed %nn ring closure")
            self.pos += 2
            return digit
        d = self._peek()
        if not (d and d.isdigit()):
            raise ParseError(f"expected ring digit at position {self.pos}")
        self.pos += 1
        return d

    # -- helpers -----------------------------------------------------------

    def _peek(self):
        return self.s[self.pos] if self.pos < len(self.s) else None

    def _digits(self):
        m = re.match(r"\d+", self.s[self.pos:])
        if not m:
            return None
        self.pos += m.end()
        return m.group()

    def _bond_start(self) -> bool:
        ch = self._peek()
        return ch is not None and ch in "-=#$:/\\"

    def _digit_start(self) -> bool:
        ch = self._peek()
        return ch is not None and (ch.isdigit() or ch == "%")

    def _eoc(self) -> bool:
        ch = self._peek()
        return ch is None or ch in ".)"

    def _eat(self, token: str) -> bool:
        if self.s.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def _expect(self, token: str):
        if not self._eat(token):
            raise ParseError(
                f"expected {token!r} at position {self.pos} in {self.s!r}")


def _capitalize(s: str) -> str:
    return s[0].upper() + s[1:]


def parse_smiles(smiles: str) -> Formula:
    return SmilesParser(smiles).parse()


# -- deterministic writer ---------------------------------------------------

BOND_TOKENS = {"single": "-", "double": "=", "triple": "#",
               "quadruple": "$", "aromatic": ":"}


def write_smiles(node) -> str:
    if isinstance(node, Formula):
        return ".".join(_write_molecule(m) for m in node.nodes
                        if isinstance(m, Molecule))
    return _write_molecule(node)


def _write_molecule(molecule: Molecule) -> str:
    atoms, edges = build_graph(molecule)
    if not edges and len(atoms) > 1:
        raise ParseError("molecule has no bonds - a formula is not a structure")

    adjacency: list[dict[int, str]] = [{} for _ in atoms]
    for edge in edges:
        adjacency[edge.from_][edge.to] = edge.kind
        adjacency[edge.to][edge.from_] = edge.kind
    return _DeterministicWriter(atoms, adjacency).write()


class _DeterministicWriter:
    def __init__(self, atoms, adjacency):
        self.atoms = atoms
        self.adjacency = adjacency
        self.visited: set[int] = set()
        self.digits: list[list[str]] = [[] for _ in atoms]
        self.digit_by_edge: dict[tuple[int, int], str] = {}
        self.next_digit = 1
        self.tree_children: list[list[int]] = [[] for _ in atoms]
        self.closures: list[list[int]] = [[] for _ in atoms]
        self.tree_size: list[int] = [1] * len(atoms)

    def write(self) -> str:
        if not self.atoms:
            return ""
        self._build_tree(0, None)
        self._compute_tree_size(0)
        return "".join(self._emit(0, None, None))

    def _build_tree(self, index: int, parent):
        self.visited.add(index)
        for nb in sorted(self.adjacency[index]):
            if nb != parent and nb in self.visited:
                self.closures[index].append(nb)
            elif nb not in self.visited:
                self.tree_children[index].append(nb)
                self._build_tree(nb, index)

    def _compute_tree_size(self, index: int):
        size = 1
        for child in self.tree_children[index]:
            self._compute_tree_size(child)
            size += self.tree_size[child]
        self.tree_size[index] = size

    def _emit(self, index, parent, incoming_kind):
        out = []
        if parent is not None:
            out.append(("literal", self._bond_token(incoming_kind, parent, index)))
        out.append(("atom", index))
        # Ring digits attach to per-atom lists (both endpoints); no
        # literal units - they would render twice.
        for nb in self.closures[index]:
            self._ring_digit_for(index, nb)

        children = self.tree_children[index]
        if not children:
            return self._render(out)
        ordered = sorted(children,
                         key=lambda c: self._continuation_rank(index, c))
        continuation, branches = ordered[0], ordered[1:]
        rendered = self._render(out)
        for child in branches:
            rendered += "(" + self._render(self._emit(
                child, index, self.adjacency[index][child])) + ")"
        return rendered + self._render(
            self._emit(continuation, index, self.adjacency[index][continuation]))

    def _render(self, parts) -> str:
        out = ""
        for part in parts:
            if isinstance(part, str):
                out += part
            else:
                kind, value = part
                if kind == "atom":
                    out += self._atom_token(value) + "".join(self.digits[value])
                else:
                    out += value
        return out

    def _continuation_rank(self, index: int, child: int):
        kind = self.adjacency[index][child]
        return (0 if kind in CONTINUATION_BONDS else 1,
                -self.tree_size[child], self._atom_token(child))

    def _ring_digit_for(self, a: int, b: int) -> str:
        key = (min(a, b), max(a, b))
        digit = self.digit_by_edge.get(key)
        if digit:
            return digit
        if self.next_digit > 9:
            raise ParseError("too many ring closures for SMILES output")
        digit = str(self.next_digit)
        self.next_digit += 1
        self.digit_by_edge[key] = digit
        self.digits[key[0]].append(digit)
        self.digits[key[1]].append(digit)
        return digit

    def _bond_token(self, kind, from_, to):
        token = BOND_TOKENS.get(kind)
        if not token:
            raise ParseError(f"{kind} bonds have no SMILES form (v1 subset)")
        if kind not in ("single", "aromatic"):
            return token
        if kind == "single":
            return token if self._aromatic(from_) and self._aromatic(to) else ""
        return "" if self._aromatic(from_) and self._aromatic(to) else token

    def _atom_token(self, index: int) -> str:
        atom = self.atoms[index]
        aromatic = atom.aromatic is True
        lowercase = aromatic and atom.element in LOWERABLE_AROMATIC
        needs_bracket = (atom.charge is not None or atom.isotope is not None
                         or atom.hydrogens is not None
                         or (atom.element not in WRITER_ORGANIC and not lowercase)
                         or (aromatic and not lowercase))
        if not needs_bracket:
            return atom.element.lower() if lowercase else atom.element
        symbol = atom.element.lower() if lowercase else atom.element
        token = "["
        if atom.isotope is not None:
            token += atom.isotope
        token += symbol
        if atom.hydrogens is not None:
            token += "H" + (str(atom.hydrogens) if atom.hydrogens > 1 else "")
        if atom.charge is not None:
            token += _charge_suffix(atom.charge)
        return token + "]"

    def _aromatic(self, index: int) -> bool:
        return self.atoms[index].aromatic is True


def _charge_suffix(charge: str) -> str:
    m = re.match(r"\d+", charge)
    count = int(m.group()) if m else 1
    sign = charge[-1]
    return f"{sign}{count}" if count > 1 else sign
