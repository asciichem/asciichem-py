"""CML (Chemical Markup Language) round-trip - the Python port of
asciichem-rs's `cml` module (rs#23), bidirectional conversion between
the AsciiChem model and the CML XML wire:

    model nodes  --emit-->  canonical wire (molecule/atom/bond/
                            reaction + the aci: extension channels)
                 --DOM-->   leptris  --serialize-->  XML string

    XML string  --leptris SAX-->  namespace-resolved tree
                --read-->         canonical wire  --restore-->  model nodes

# The leptris directive

Every byte of XML parsing and serialization goes through the leptris
engine (the mandated serialization layer for the family - never
hand-rolled XML strings). The parse half rides `leptris.sax`
(namespace-resolved events); the construction half rides the
programmatic builder API (`Document.create` / `create_element` /
`create_child` / `set` / `.text`) and its serializer - escaping,
well-formedness, and the emitted bytes all come from libleptris.

# The aci: extension namespace

CML's standard wire covers element, isotope, charge, count, and bond
order - not oxidation state, Lewis markers, electron configuration,
embedded math, free-form text, group structure, the beyond-formula
constructs, or reaction conditions. AsciiChem carries these through an
`aci:` (AsciiChem extension) namespace, four channels, one per scope,
each a frozen registry:

- **Atom attributes** - `aci:<wire_name>` attributes on `<atom>`
  elements (`oxidationState`, `lonePairs`, `radicalElectrons`,
  `ringClosures`).
- **Top-level elements** - `<aci:<name> position="N">text</...>`
  children of `<cml>` for `ElectronConfiguration`, `EmbeddedMath`,
  `Text`, `Crystal`, `Spectrum`, `Calculation`, `ZMatrix`,
  `Mechanism`; `position` preserves node ordering in the formula. The
  payload is the construct's canonical text render - round-trip-safe
  by construction (the Text formatter is the canonicaliser), the
  reference's TopLevel handler contract.
- **Molecule groups** - `<aci:group multiplicity="N" bracket="X"
  atomRefs="a1 a2"/>` children of `<molecule>`, preserving AsciiChem
  group structure through the canonical flattening.
- **Reaction conditions** - `aci:conditionsAbove` /
  `aci:conditionsBelow` attributes on `<reaction>` (the reference's
  ConditionsExtensions attribute shape), read back per reaction
  element in document (= canonical) order.

The `xmlns:aci` declaration appears only when at least one extension
is present; plain atoms and molecules produce clean CML. Unknown
elements and attributes are ignored on read; foreign namespaces can
never be mistaken for a channel (classification is by resolved
namespace URI, not by written prefix).

Constructs with neither wire representation nor channel raise
`CmlError` on emit rather than dropping silently (bare atoms/bonds
outside a molecule, reaction participants that are not molecules,
molecule annotations)."""
from __future__ import annotations

import dataclasses
from typing import Optional

from leptris import Document, tostring
from leptris import ParseError as LeptrisParseError
from leptris import sax

from .model import (Atom, AsciiChemError, Bond, Formula, Group, Molecule,
                    Node, Reaction, ReactionCascade, ReactionConditions,
                    diagnostic_label)
from . import text as text_renderer
from .engine import parse_text

# The CML core namespace (the standard wire).
CML_NAMESPACE = "http://www.xml-cml.org/schema"
# The AsciiChem extension namespace (the `aci:` channels).
ACI_NAMESPACE = "https://asciichem.org/cml-ext"
# The written prefix for ACI_NAMESPACE on emit.
ACI_PREFIX = "aci"


class CmlError(AsciiChemError):
    """An error surfaced by CML emit or read: the XML engine rejected
    the input, the wire is not the shape this module reads, or the
    model construct has no wire representation in this slice."""


# ---------------------------------------------------------------------------
# Frozen channel registries (the aci: extension surface). Adding a
# channel member is one entry here - no other code changes.
# ---------------------------------------------------------------------------

# Atom-attribute channel: model field name -> wire attribute name
# (without the `aci:` prefix), in wire order.
ATOM_FIELDS: list[tuple[str, str]] = [
    ("oxidation_state", "oxidationState"),
    ("lone_pairs", "lonePairs"),
    ("radical_electrons", "radicalElectrons"),
    ("ring_closures", "ringClosures"),
]

# Top-level channel element names (the registry order follows the
# reference's TopLevel::HANDLERS).
TOP_LEVEL_ELEMENTS = ("electronConfiguration", "embeddedMath", "text",
                      "crystal", "spectrum", "calculation", "zmatrix",
                      "mechanism")

# The per-reaction conditions channel: `aci:conditionsAbove` /
# `aci:conditionsBelow` attributes on `<reaction>` (the reference's
# ConditionsExtensions attribute shape), read back per reaction
# element in document (= canonical) order.
CONDITION_ATTRS = (("above", "conditionsAbove"), ("below", "conditionsBelow"))

# Group brackets on the wire (the model stores the same names).
BRACKETS = ("paren", "square", "brace")

_ORDER_CODES = {"double": "D", "triple": "T", "quadruple": "Q", "wedge": "W",
                "hash": "H", "dative": "A", "wavy": "V"}

_KIND_FROM_ORDER = {"D": "double", "T": "triple", "Q": "quadruple",
                    "W": "wedge", "H": "hash", "A": "dative", "V": "wavy"}


def _order_code(kind: str) -> str:
    """Bond kind -> CML order code (the reference's CML_ORDER_CODES)."""
    return _ORDER_CODES.get(kind, "S")


def _kind_from_order(order: str) -> str:
    """CML order code -> bond kind (the reference's KIND_BY_CML_ORDER)."""
    return _KIND_FROM_ORDER.get(order, "single")


def _charge_value(charge: str) -> Optional[int]:
    """Parse an atom charge ("2+", "+2", "-", "2-") into a signed int."""
    first, last = charge[:1], charge[-1:]
    if last in ("+", "-") and first.isdigit():
        digits, sign = charge[:-1], last
    elif first in ("+", "-") and last.isdigit():
        digits, sign = charge[1:], first
    elif first in ("+", "-") and len(charge) == 1:
        digits, sign = "", first
    else:
        return None
    if digits and not digits.isdigit():
        return None
    magnitude = int(digits) if digits else 1
    return magnitude if sign == "+" else -magnitude


def _leading_integer(raw) -> int:
    """The leading integer of a string (Ruby's String#to_i semantics:
    leading digits, 0 when none) - the reference's lenient XML integer
    cast for wire values."""
    digits = ""
    for char in str(raw or ""):
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# Canonical IDs - a/b/m/r/g counters shared across one translation
# pass, so re-running on equivalent input yields the same output.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Ids:
    atoms: int = 0
    bonds: int = 0
    molecules: int = 0
    reactions: int = 0
    groups: int = 0

    def next_atom(self) -> str:
        self.atoms += 1
        return f"a{self.atoms}"

    def next_bond(self) -> str:
        self.bonds += 1
        return f"b{self.bonds}"

    def next_molecule(self) -> str:
        self.molecules += 1
        return f"m{self.molecules}"

    def next_reaction(self) -> str:
        self.reactions += 1
        return f"r{self.reactions}"

    def next_group(self) -> str:
        self.groups += 1
        return f"g{self.groups}"


# ---------------------------------------------------------------------------
# Wire records - the flat molecule data the walk produces and the
# serializer writes.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _WireAtom:
    id: str
    element: str
    isotope: Optional[str]
    formal_charge: Optional[str]
    count: Optional[str]
    # The aci: channel values, keyed by model field name.
    extensions: dict


@dataclasses.dataclass
class _WireBond:
    id: str
    from_: str
    to: str
    order: str


@dataclasses.dataclass
class _WireGroup:
    multiplicity: Optional[str]
    bracket: str
    atom_ids: list


@dataclasses.dataclass
class _MoleculeRecord:
    coefficient: Optional[str]
    atoms: list
    bonds: list
    groups: list


@dataclasses.dataclass
class _ReactionRecord:
    arrow: str
    reactants: list
    products: list
    conditions: Optional[ReactionConditions] = None


# ---------------------------------------------------------------------------
# Emit: model -> wire records -> DOM -> XML.
# ---------------------------------------------------------------------------

def to_cml(node) -> str:
    """Serializes formula nodes to a CML XML string. Accepts a
    `Formula`, a node list, or a single node; `parse_cml(to_cml(nodes))`
    rebuilds the model for the gated surface - the three-way
    round-trip contract."""
    if isinstance(node, Formula):
        nodes = list(node.nodes)
    elif isinstance(node, (list, tuple)):
        nodes = list(node)
    else:
        nodes = [node]
    ids = _Ids()

    # Walk every formula node, per-category, like the reference
    # adapter: molecules, reactions, cascades regroup; top-level aci:
    # entries keep their formula position.
    molecules: list[_MoleculeRecord] = []
    reactions: list[_ReactionRecord] = []
    reaction_lists: list[list[_ReactionRecord]] = []
    top_level: list[tuple[int, str, str]] = []
    needs_aci = False

    for position, inner in enumerate(nodes):
        if isinstance(inner, Molecule):
            record, extensions = _walk_molecule(ids, inner)
            needs_aci = needs_aci or extensions
            molecules.append(record)
        elif isinstance(inner, Reaction):
            record, extensions = _walk_reaction(ids, inner)
            needs_aci = needs_aci or extensions
            reactions.append(record)
        elif isinstance(inner, ReactionCascade):
            steps = []
            for step in inner.steps:
                record, extensions = _walk_reaction(ids, step)
                needs_aci = needs_aci or extensions
                steps.append(record)
            reaction_lists.append(steps)
        elif isinstance(inner, (Atom, Bond)):
            label = _node_label(inner)
            raise CmlError(f"construct has no CML wire representation: "
                           f"bare {label} outside a molecule")
        else:
            element, content = _top_level_payload(inner)
            needs_aci = True
            top_level.append((position, element, content))

    document = Document.create()
    root = document.create_element("cml")
    root.set("xmlns", CML_NAMESPACE)
    if needs_aci:
        root.set("xmlns:aci", ACI_NAMESPACE)
    for position, element, content in top_level:
        entry = root.create_child(f"{ACI_PREFIX}:{element}")
        entry.set("position", str(position))
        entry.text = content
    for record in molecules:
        _emit_molecule(root, ids, record)
    for record in reactions:
        _emit_reaction(root, ids, record)
    for steps in reaction_lists:
        element = root.create_child("reactionList")
        for step in steps:
            _emit_reaction(element, ids, step)
    document.set_root(root)
    return tostring(document, encoding="unicode")


def _node_label(node) -> str:
    return diagnostic_label(node)


def _top_level_payload(node) -> tuple[str, str]:
    """The top-level channel payload: wire element name + text content.
    Text-shaped constructs serialize as their canonical text render
    (round-trip-safe by construction); embedded math carries the raw
    source (the backticks are re-added on read)."""
    from .model import (Calculation, Crystal, ElectronConfiguration,
                        EmbeddedMath, Mechanism, Spectrum, Text, ZMatrix)
    if isinstance(node, ElectronConfiguration):
        return "electronConfiguration", text_renderer.render(node)
    if isinstance(node, EmbeddedMath):
        return "embeddedMath", node.source
    if isinstance(node, Text):
        return "text", f'"{node.content}"'
    if isinstance(node, Crystal):
        return "crystal", text_renderer.render(node)
    if isinstance(node, Spectrum):
        return "spectrum", text_renderer.render(node)
    if isinstance(node, Calculation):
        return "calculation", text_renderer.render(node)
    if isinstance(node, ZMatrix):
        return "zmatrix", text_renderer.render(node)
    if isinstance(node, Mechanism):
        return "mechanism", text_renderer.render(node)
    raise CmlError(f"construct has no CML wire representation: "
                   f"{_node_label(node)}")


def _walk_molecule(ids: _Ids, molecule: Molecule) -> tuple[_MoleculeRecord, bool]:
    """Walks a molecule's nodes, producing the ordered atom list, the
    bond list with atomRefs, and the group records - the single source
    of atom ID assignment. Mirrors the reference's MoleculeWalker."""
    walker = _MoleculeWalker(ids)
    walker.walk_nodes(molecule.nodes, None)
    walker.emit_ring_bonds()
    extensions = bool(walker.groups) or any(
        atom.extensions for atom in walker.atoms)
    record = _MoleculeRecord(coefficient=molecule.coefficient,
                             atoms=walker.atoms, bonds=walker.bonds,
                             groups=walker.groups)
    return record, extensions


class _MoleculeWalker:
    """Walks a molecule's nodes, assigning atom/bond/group IDs and
    producing the wire records (the reference's MoleculeWalker)."""

    def __init__(self, ids: _Ids):
        self.ids = ids
        self.atoms: list[_WireAtom] = []
        self.bonds: list[_WireBond] = []
        self.groups: list[_WireGroup] = []
        self.group_stack: list[int] = []
        self.pending_bond: Optional[str] = None
        self.last_atom: Optional[str] = None
        self.open_rings: list[tuple[str, str]] = []
        self.ring_pairs: list[tuple[str, str]] = []

    def walk_nodes(self, nodes, multiplier: Optional[int]) -> None:
        for node in nodes:
            if isinstance(node, Atom):
                self.emit_atom(node, multiplier)
            elif isinstance(node, Bond):
                self.pending_bond = node.kind
            elif isinstance(node, Group):
                self.enter_group(node, multiplier)
            elif isinstance(node, Molecule):
                inner = _combine_multiplier(multiplier, node.coefficient)
                self.walk_nodes(node.nodes, inner)

    def enter_group(self, group: Group, multiplier: Optional[int]) -> None:
        multiplicity = group.multiplicity
        inner = _combine_multiplier(multiplier, multiplicity)
        bracket = group.bracket if group.bracket in BRACKETS else "paren"
        # The canonical walk assigns group IDs (`g1`, ...) at the same
        # point the reference's GroupRecord does, keeping the ID stream
        # identical; the wire itself carries groups by atomRefs only.
        self.ids.next_group()
        self.groups.append(_WireGroup(multiplicity=multiplicity,
                                      bracket=bracket, atom_ids=[]))
        self.group_stack.append(len(self.groups) - 1)
        self.walk_nodes(group.nodes, inner)
        self.group_stack.pop()

    def emit_atom(self, atom: Atom, multiplier: Optional[int]) -> None:
        id_ = self.ids.next_atom()
        for index in self.group_stack:
            self.groups[index].atom_ids.append(id_)
        if atom.ring_closures is not None:
            for digit in atom.ring_closures:
                matched = None
                for i, (open_digit, _) in enumerate(self.open_rings):
                    if open_digit == digit:
                        matched = i
                        break
                if matched is not None:
                    _, from_ = self.open_rings.pop(matched)
                    self.ring_pairs.append((from_, id_))
                else:
                    self.open_rings.append((digit, id_))
        if multiplier is not None:
            combined = _combine_multiplier(multiplier, atom.subscript)
            count = str(combined if combined is not None else multiplier)
        else:
            count = atom.subscript
        if self.pending_bond is not None and self.last_atom is not None:
            kind = self.pending_bond
            self.pending_bond = None
            self.bonds.append(_WireBond(id=self.ids.next_bond(),
                                        from_=self.last_atom, to=id_,
                                        order=_order_code(kind)))
        extensions = {}
        for field, wire in ATOM_FIELDS:
            value = getattr(atom, field)
            if value is None:
                continue
            extensions[field] = value if isinstance(value, str) else str(value)
        self.atoms.append(_WireAtom(id=id_, element=atom.element,
                                    isotope=atom.isotope,
                                    formal_charge=atom.charge,
                                    count=count, extensions=extensions))
        self.last_atom = id_

    def emit_ring_bonds(self) -> None:
        """Emits canonical bonds for each ring-closure pair, skipping
        pairs a positional bond already covers (the degenerate
        `C1-C1`)."""
        pairs, self.ring_pairs = self.ring_pairs, []
        for from_, to in pairs:
            duplicate = any((bond.from_ == from_ and bond.to == to)
                            or (bond.from_ == to and bond.to == from_)
                            for bond in self.bonds)
            if not duplicate:
                self.bonds.append(_WireBond(id=self.ids.next_bond(),
                                            from_=from_, to=to, order="S"))


def _combine_multiplier(left: Optional[int], right) -> Optional[int]:
    """The reference walker's multiplier combine: numeric multiply,
    None when neither side carries a usable integer, the usable side
    otherwise. None right with a numeric left yields the multiplier
    (a group without multiplicity inside a `2`-coefficient molecule
    still doubles its atom counts). Only a full integer literal
    combines ("2a" stays a raw string subscript)."""
    if isinstance(right, str):
        parsed = int(right) if right.isdigit() else None
    elif isinstance(right, int):
        parsed = right
    else:
        parsed = None
    if left is None and parsed is None:
        return None
    return (left or 1) * (parsed or 1)


def _total_formal_charge(atoms: list[_WireAtom]) -> Optional[str]:
    total = 0
    charged = False
    for atom in atoms:
        value = (_charge_value(atom.formal_charge)
                 if atom.formal_charge is not None else None)
        if value is not None:
            charged = True
            total += value
    if not charged:
        return None
    return f"+{total}" if total > 0 else str(total)


def _emit_molecule(parent, ids: _Ids, record: _MoleculeRecord) -> None:
    element = parent.create_child("molecule")
    element.set("id", ids.next_molecule())
    if record.coefficient is not None:
        element.set("count", record.coefficient)
    total = _total_formal_charge(record.atoms)
    if total is not None:
        element.set("formalCharge", total)
    if record.atoms:
        array = element.create_child("atomArray")
        for atom in record.atoms:
            el = array.create_child("atom")
            el.set("id", atom.id)
            el.set("elementType", atom.element)
            if atom.isotope is not None:
                el.set("isotope", atom.isotope)
            if atom.formal_charge is not None:
                el.set("formalCharge", atom.formal_charge)
            if atom.count is not None:
                el.set("count", atom.count)
            for field, wire in ATOM_FIELDS:
                if field in atom.extensions:
                    el.set(f"{ACI_PREFIX}:{wire}", atom.extensions[field])
    if record.bonds:
        array = element.create_child("bondArray")
        for bond in record.bonds:
            el = array.create_child("bond")
            el.set("id", bond.id)
            el.set("atomRefs2", f"{bond.from_} {bond.to}")
            el.set("order", bond.order)
    if record.groups:
        for group in record.groups:
            el = element.create_child(f"{ACI_PREFIX}:group")
            if group.multiplicity is not None:
                el.set("multiplicity", group.multiplicity)
            el.set("bracket", group.bracket)
            el.set("atomRefs", " ".join(group.atom_ids))


def _walk_reaction(ids: _Ids,
                   reaction: Reaction) -> tuple[_ReactionRecord, bool]:
    conditions = reaction.conditions
    has_conditions = (conditions is not None
                      and (conditions.above is not None
                           or conditions.below is not None))
    record = _ReactionRecord(arrow=reaction.arrow, reactants=[], products=[],
                             conditions=conditions if has_conditions else None)
    extensions = has_conditions
    for molecule in reaction.reactants:
        participant, has_extensions = _walk_participant(ids, molecule)
        record.reactants.append(participant)
        extensions = extensions or has_extensions
    for molecule in reaction.products:
        participant, has_extensions = _walk_participant(ids, molecule)
        record.products.append(participant)
        extensions = extensions or has_extensions
    return record, extensions


def _walk_participant(ids: _Ids, node) -> tuple[_MoleculeRecord, bool]:
    if not isinstance(node, Molecule):
        raise CmlError(f"construct has no CML wire representation: "
                       f"reaction participant {_node_label(node)}")
    return _walk_molecule(ids, node)


def _emit_reaction(parent, ids: _Ids, record: _ReactionRecord) -> None:
    element = parent.create_child("reaction")
    element.set("id", ids.next_reaction())
    element.set("title", record.arrow)
    element.set("type", record.arrow)
    if record.conditions is not None:
        # The conditions channel: above/below ride `aci:` attributes on
        # the reaction element (the reference's ConditionsExtensions
        # attribute shape; the bracket text stays verbatim, so the
        # typed entries re-derive on the restored model).
        for side, attr in CONDITION_ATTRS:
            value = getattr(record.conditions, side)
            if value is not None:
                element.set(f"{ACI_PREFIX}:{attr}", value)
    reactant_list = element.create_child("reactantList")
    for molecule in record.reactants:
        _emit_participant(reactant_list, ids, "reactant", molecule)
    product_list = element.create_child("productList")
    for molecule in record.products:
        _emit_participant(product_list, ids, "product", molecule)


def _emit_participant(parent, ids: _Ids, role: str,
                      molecule: _MoleculeRecord) -> None:
    participant = parent.create_child(role)
    substance = participant.create_child("substance")
    substance.set("role", role)
    _emit_molecule(substance, ids, molecule)


# ---------------------------------------------------------------------------
# Read: XML -> namespace-resolved tree -> model. The tree is built from
# the engine's SAX events - the namespace URIs are resolved through the
# declared prefix mappings, so channel classification is exact
# regardless of the written prefix or where declarations appear.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _XAttr:
    ns: Optional[str]
    local: str
    value: str


@dataclasses.dataclass
class _XNode:
    ns: Optional[str]
    local: str
    attrs: list
    children: list
    text: str

    def attr(self, local: str) -> Optional[str]:
        for a in self.attrs:
            if a.ns is None and a.local == local:
                return a.value
        return None

    def attr_ns(self, ns: str, local: str) -> Optional[str]:
        for a in self.attrs:
            if a.ns == ns and a.local == local:
                return a.value
        return None

    def children_named(self, local: str) -> list:
        return [c for c in self.children if c.local == local]

    def child(self, local: str) -> Optional["_XNode"]:
        for c in self.children:
            if c.local == local:
                return c
        return None

    def is_core(self) -> bool:
        """True when this node rides the core CML namespace (or no
        namespace - sloppy-but-common undeclared CML)."""
        return self.ns is None or self.ns == CML_NAMESPACE


def _resolve_prefix(scopes: list[tuple[str, str]], prefix: str) -> Optional[str]:
    """Innermost binding for `prefix` ("" = the default namespace)."""
    for bound, uri in reversed(scopes):
        if bound == prefix:
            return uri
    return None


def _qname_split(name: str) -> tuple[str, str]:
    prefix, separator, local = name.partition(":")
    if separator:
        return prefix, local
    return "", name


class _TreeBuilder(sax.SAXHandler):
    """Builds the namespace-resolved tree from the engine's SAX
    events (the port of the reference's TreeBuilder). The recovered
    diagnostic is stored as `error_message` - an attribute named
    `error` would shadow the SAXHandler error callback with a None
    value and break the engine's dispatch."""

    def __init__(self):
        self.scopes: list[tuple[str, str]] = []
        self.stack: list[_XNode] = []
        self.root: Optional[_XNode] = None
        self.error_message: Optional[str] = None

    def start_prefix_mapping(self, prefix: str, uri: str) -> None:
        self.scopes.append((prefix, uri))

    def end_prefix_mapping(self, prefix: str) -> None:
        for index in range(len(self.scopes) - 1, -1, -1):
            if self.scopes[index][0] == prefix:
                self.scopes.pop(index)
                return

    def start_element(self, name: str, attributes) -> None:
        prefix, local = _qname_split(name)
        ns = _resolve_prefix(self.scopes, prefix)
        parsed = []
        for attr_name, value in attributes.items():
            attr_prefix, _ = _qname_split(attr_name)
            # xmlns declarations ride the prefix-mapping events; and
            # unprefixed attributes have no namespace, per XML
            # Namespaces - the default declaration never applies.
            if attr_prefix == "xmlns" or attr_name == "xmlns":
                continue
            attr_ns = (None if attr_prefix == ""
                       else _resolve_prefix(self.scopes, attr_prefix))
            parsed.append(_XAttr(ns=attr_ns, local=_qname_split(attr_name)[1],
                                 value=value))
        self.stack.append(_XNode(ns=ns, local=local, attrs=parsed,
                                 children=[], text=""))

    def characters(self, text: str) -> None:
        if self.stack:
            self.stack[-1].text += text

    def cdata(self, text: str) -> None:
        if self.stack:
            self.stack[-1].text += text

    def end_element(self, name: str) -> None:
        if not self.stack:
            return
        done = self.stack.pop()
        if self.stack:
            self.stack[-1].children.append(done)
        else:
            self.root = done

    def error(self, message: str, line: int, column: int) -> None:
        if self.error_message is None:
            self.error_message = message


def build_tree(xml: str) -> _XNode:
    builder = _TreeBuilder()
    try:
        sax.parse(xml, builder)
    except LeptrisParseError as exc:
        raise CmlError(f"CML XML engine error: {exc}") from exc
    if builder.error_message is not None:
        raise CmlError(f"CML XML engine error: {builder.error_message}")
    if builder.root is None:
        raise CmlError("CML XML engine error: empty document")
    return builder.root


def parse_cml(xml: str) -> Formula:
    """Parses a CML XML string back into a Formula (the node list)."""
    root = build_tree(xml)
    if root.local != "cml" or not root.is_core():
        raise CmlError(f"CML wire violation: expected a <cml> root, "
                       f"got <{root.local}>")

    # Restore order mirrors the reference translator: core wire (atoms
    # carry the atom-attribute channel inline), then group rebuild,
    # then top-level insertion at preserved positions.
    nodes: list[Node] = []
    top_level: list[tuple[int, Node]] = []
    for child in root.children:
        if child.ns == ACI_NAMESPACE:
            if child.local in TOP_LEVEL_ELEMENTS:
                top_level.append(_read_top_level(child))
            continue
        if not child.is_core():
            continue
        if child.local == "molecule":
            nodes.append(_read_molecule(child))
        elif child.local == "reaction":
            nodes.append(_read_reaction(child))
        elif child.local == "reactionList":
            steps = [_read_reaction(reaction)
                     for reaction in child.children_named("reaction")]
            nodes.append(ReactionCascade(steps=steps))

    # Ascending position order, like the reference restore: earlier
    # inserts never shift the later positions.
    top_level.sort(key=lambda entry: entry[0])
    for position, node in top_level:
        nodes.insert(min(position, len(nodes)), node)
    return Formula(nodes=nodes)


def _read_top_level(element: _XNode) -> tuple[int, Node]:
    raw = element.attr("position")
    position = max(_leading_integer(raw), 0) if raw is not None else 0
    content = element.text
    # The wire carries the construct's canonical text render; wrapped
    # constructs get their delimiters back before re-parsing.
    payload = (f"`{content}`" if element.local == "embeddedMath"
               else content)
    parsed = parse_text(payload)
    if not parsed.nodes:
        raise CmlError(f"CML wire violation: aci:{element.local} "
                       "content is empty")
    return position, parsed.nodes[0]


def _read_molecule(element: _XNode) -> Molecule:
    coefficient = element.attr("count")
    wire_atoms = []
    wire_bonds = []
    groups = []
    for child in element.children:
        if child.ns == ACI_NAMESPACE:
            if child.local == "group":
                groups.append(_read_group(child))
            continue
        if child.local == "atomArray":
            wire_atoms.extend(_read_atom(atom)
                              for atom in child.children_named("atom"))
        elif child.local == "bondArray":
            wire_bonds.extend(_read_bond(bond)
                              for bond in child.children_named("bond"))
    molecule = Molecule(nodes=[], coefficient=coefficient)
    _rebuild_atoms_and_bonds(molecule, wire_atoms, wire_bonds)
    _rebuild_groups(molecule, groups)
    return molecule


@dataclasses.dataclass
class _RestoredAtom:
    id: Optional[str]
    atom: Atom


@dataclasses.dataclass
class _RestoredBond:
    refs: list
    order: str


@dataclasses.dataclass
class _GroupRead:
    multiplicity: Optional[str]
    bracket: str
    atom_ids: list


def _read_atom(element: _XNode) -> _RestoredAtom:
    element_type = element.attr("elementType")
    if element_type is None:
        raise CmlError("CML wire violation: atom without elementType")
    count = element.attr("count")
    subscript = None if count is None or count == "1" else count
    lone_pairs = (element.attr_ns(ACI_NAMESPACE, "lonePairs"))
    radical = element.attr_ns(ACI_NAMESPACE, "radicalElectrons")
    return _RestoredAtom(
        id=element.attr("id"),
        atom=Atom(
            element=element_type,
            isotope=element.attr("isotope"),
            subscript=subscript,
            charge=element.attr("formalCharge"),
            oxidation_state=element.attr_ns(ACI_NAMESPACE, "oxidationState"),
            lone_pairs=_lewis_count(lone_pairs) if lone_pairs is not None else None,
            radical_electrons=(_lewis_count(radical)
                               if radical is not None else None),
            ring_closures=element.attr_ns(ACI_NAMESPACE, "ringClosures"),
        ))


def _lewis_count(value: str) -> int:
    """The reference's lenient XML integer cast (Ruby's String#to_i)."""
    return max(_leading_integer(value), 0)


def _read_bond(element: _XNode) -> _RestoredBond:
    raw = element.attr("atomRefs2") or ""
    refs = raw.split()
    if len(refs) != 2:
        raise CmlError("CML wire violation: bond atomRefs2 must reference "
                       f"two atoms, got {raw.split()!r}")
    return _RestoredBond(refs=refs, order=element.attr("order") or "S")


def _read_group(element: _XNode) -> _GroupRead:
    bracket = element.attr("bracket")
    if bracket not in BRACKETS:
        bracket = "paren"
    return _GroupRead(multiplicity=element.attr("multiplicity"),
                      bracket=bracket,
                      atom_ids=(element.attr("atomRefs") or "").split())


def _rebuild_atoms_and_bonds(molecule: Molecule, wire_atoms: list,
                             wire_bonds: list) -> None:
    """Rebuilds the molecule's node list: atoms in wire order, bonds
    inserted just before their later endpoint (the linear-spelling
    position), ring bonds (spanning atoms) left to the atom-attribute
    channel's ring-closure digits."""
    ids = [restored.id for restored in wire_atoms]
    nodes: list[Node] = [_atom_of(restored) for restored in wire_atoms]
    if not wire_bonds:
        molecule.nodes = nodes
        return

    def position(id_: str) -> Optional[int]:
        return ids.index(id_) if id_ in ids else None

    placed = []
    for bond in wire_bonds:
        positions = sorted(p for p in (position(ref) for ref in bond.refs)
                           if p is not None)
        later = positions[-1] if positions else 0
        span = (positions[-1] - positions[0]) if len(positions) >= 2 else 0
        placed.append((bond, later, span))
    # Descending later position so earlier insertions don't shift the
    # pending ones.
    placed.sort(key=lambda entry: entry[1], reverse=True)
    for bond, later, span in placed:
        # A ring bond connects non-adjacent atoms; its endpoints span
        # a gap, and the aci:ringClosures digits re-spell it.
        if span > 1:
            continue
        nodes.insert(later, Bond(kind=_kind_from_order(bond.order),
                                 direction=None))
    molecule.nodes = nodes


def _atom_of(restored: _RestoredAtom) -> Atom:
    return restored.atom


def _rebuild_groups(molecule: Molecule, groups: list[_GroupRead]) -> None:
    """Rebuilds Group nodes from the aci:group records. Records are
    processed in reverse order so inner groups rebuild before their
    outer parents (the walker records outer groups first)."""
    for record in reversed(groups):
        _rebuild_one_group(molecule, record)


def _rebuild_one_group(molecule: Molecule, record: _GroupRead) -> None:
    targets = []
    for id_ in record.atom_ids:
        if id_.startswith("a") and id_[1:].isdigit():
            targets.append(int(id_[1:]))
    if not targets:
        return
    span = _find_target_span(molecule.nodes, targets)
    if span is None:
        return
    first, last = span
    _apply_group_multiplicity(molecule.nodes, first, last,
                              record.multiplicity)
    _splice_group(molecule, first, last, record)


def _find_target_span(nodes, targets) -> Optional[tuple[int, int]]:
    """Walks the molecule's nodes in canonical-walk order, tracking
    which atom id each position covers; the group spans the
    contiguous run covering any referenced atom (bonds included,
    preserving the chain inside the group)."""
    first = None
    last = None
    canonical = 0
    for index, node in enumerate(nodes):
        count = _count_atoms(node)
        covered = any(canonical + offset in targets
                      for offset in range(1, count + 1))
        if covered:
            first = index if first is None else first
            last = index
        canonical += count
    if first is None:
        return None
    return first, last if last is not None else first


def _count_atoms(node) -> int:
    """Counts the atoms transitively contained in a node."""
    if isinstance(node, Atom):
        return 1
    if isinstance(node, (Group, Molecule)):
        return sum(_count_atoms(child) for child in node.nodes)
    return 0


def _apply_group_multiplicity(nodes, first: int, last: int,
                              multiplicity: Optional[str]) -> None:
    """Divides each covered atom's subscript by the group multiplicity
    (the emit side multiplied it in). Inner groups were already
    divided when they rebuilt, so only bare atoms in this slice move."""
    if multiplicity is None:
        return
    divisor = _leading_integer(multiplicity)
    if divisor <= 0:
        return
    for node in nodes[first:last + 1]:
        if not isinstance(node, Atom):
            continue
        if node.subscript is None:
            continue
        divided = _leading_integer(node.subscript) // divisor
        node.subscript = None if divided == 1 else str(divided)


def _splice_group(molecule: Molecule, first: int, last: int,
                  record: _GroupRead) -> None:
    """Replaces the target run with a Group node. A bond immediately
    before the group's first node is its attachment bond (the grammar
    keeps it inside the group), so it moves in rather than re-spelling
    as a stray prefix marker."""
    inner = molecule.nodes[first:last + 1]
    molecule.nodes[first:last + 1] = []
    start = first
    if start > 0 and isinstance(molecule.nodes[start - 1], Bond):
        bond = molecule.nodes.pop(start - 1)
        inner.insert(0, bond)
        start -= 1
    molecule.nodes.insert(start, Group(nodes=inner,
                                       multiplicity=record.multiplicity,
                                       bracket=record.bracket))


def _read_reaction(element: _XNode) -> Reaction:
    arrow = element.attr("type") or element.attr("title")
    if arrow not in ("forward", "reverse", "equilibrium", "resonance"):
        arrow = "forward"
    reactants: list[Node] = []
    products: list[Node] = []
    for child in element.children:
        if not child.is_core():
            continue
        if child.local == "reactantList":
            reactants.extend(_read_participant(p)
                             for p in child.children_named("reactant"))
        elif child.local == "productList":
            products.extend(_read_participant(p)
                            for p in child.children_named("product"))
    conditions = None
    above = element.attr_ns(ACI_NAMESPACE, "conditionsAbove")
    below = element.attr_ns(ACI_NAMESPACE, "conditionsBelow")
    if above is not None or below is not None:
        conditions = ReactionConditions(above=above, below=below)
    return Reaction(reactants=reactants, products=products, arrow=arrow,
                    conditions=conditions)


def _read_participant(element: _XNode) -> Node:
    substance = element.child("substance")
    if substance is None:
        raise CmlError(f"CML wire violation: {element.local} "
                       "without substance")
    molecule = substance.child("molecule")
    if molecule is None:
        raise CmlError("CML wire violation: substance without molecule")
    return _read_molecule(molecule)
