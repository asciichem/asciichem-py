"""The linter - the Python mirror of `AsciiChem::Linter`. The parser
is total: it accepts any syntactically valid input without judging
chemistry correctness. The linter is a separate opt-in pass that
walks the model and reports chemistry errors - unbalanced reactions,
valence violations, isotope nonsense, broken identifier annotations,
structural identity drift.

# Registry

REGISTRY holds the checks as (name, function) pairs in the reference's
registration order; `run()` executes it over a parsed formula,
returning diagnostics sorted errors-first (then message, for stable
output). The registry is open for extension: `register(name, check)`
adds a check; a caller may also hand `run()` a composed dict to run a
subset.

# Checks (the reference's thirteen)

balance, bracket balance, charge balance, crystal sanity, element
validation, identifier consistency, identifier format, the
engine-scaled identity cross-check, isotope sanity, spectrum peaks,
unclosed rings, valence, and z-matrix references - the reference's
full set, ported (the shared corpus's `linter-expectations` family in
`tests/conformance/` gates the engine-independent slice and this
implementation's engine posture).

# Totality

Every check reports, none raises. Unknown constructs are skipped
(conservative None), the identity cross-check turns engine rejection
into a warning, and diagnostics carry severity + message + the
node's diagnostic label."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import identifiers, inchi, periodic_table
from .model import (Atom, AsciiChemError, Crystal, Formula, Group, Molecule,
                    Reaction, ReactionCascade, Spectrum, ZMatrix, ZRow,
                    diagnostic_label as node_label)

SEVERITIES = ("error", "warning", "info")
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Diagnostic:
    """One lint finding. `label` is the offending node's diagnostic
    label (the reference's `Diagnostic#to_s` context - e.g.
    `Atom(C)`); `details` carries optional structured data behind the
    message (e.g. balance's per-element [reactants, products] counts)
    for consumers that want the numbers, not the prose. Human-facing
    output stays the message."""
    severity: str
    message: str
    node: object = field(default=None, compare=False, repr=False)
    details: Optional[dict] = field(default=None, compare=False)

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")

    @property
    def label(self) -> Optional[str]:
        """The node's diagnostic label (the corpus's `node` field) -
        None when the diagnostic has no node context."""
        return node_label(self.node) if self.node is not None else None

    def to_s(self) -> str:
        context = self.label
        if context:
            return f"[{self.severity}] {context}: {self.message}"
        return f"[{self.severity}] {self.message}"

    def __str__(self) -> str:
        return self.to_s()


def error(message: str, node=None) -> Diagnostic:
    return Diagnostic("error", message, node)


def warning(message: str, node=None) -> Diagnostic:
    return Diagnostic("warning", message, node)


def info(message: str, node=None) -> Diagnostic:
    return Diagnostic("info", message, node)


# -- shared walk ---------------------------------------------------------

def walk(formula: Formula):
    """Depth-first walk over every node in the formula: containers
    recurse into their children (groups, molecules, reactions' sides,
    cascade steps' sides). Adding a new container class means
    extending this branch list - no check edits."""
    stack = list(formula.nodes)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (Group, Molecule)):
            stack.extend(node.nodes)
        elif isinstance(node, Reaction):
            stack.extend(node.reactants)
            stack.extend(node.products)
        elif isinstance(node, ReactionCascade):
            for step in node.steps:
                stack.extend(step.reactants)
                stack.extend(step.products)


def collect_nodes(formula: Formula, kind) -> list:
    """Every node of `kind` in the tree, outermost first."""
    return [n for n in walk(formula) if isinstance(n, kind)]


# -- balance -------------------------------------------------------------

def _enumerate(node, multiplier) -> Optional[dict[str, int]]:
    """Recursive atom enumeration with a current multiplier. None when
    the node holds atoms the check cannot enumerate (conservative
    skip, never a false positive)."""
    if isinstance(node, Atom):
        raw = node.subscript
        sub = int(raw) if raw and raw.isdigit() else 1
        return {node.element: (sub if sub > 0 else 1) * multiplier}
    if isinstance(node, Molecule):
        return _enumerate_many(node.nodes, multiplier)
    if isinstance(node, Group):
        raw = node.multiplicity
        mult = int(raw) if raw and raw.isdigit() else 1
        return _enumerate_many(node.nodes, (mult if mult > 0 else 1) * multiplier)
    return None


def _enumerate_many(nodes, multiplier) -> Optional[dict[str, int]]:
    result: dict[str, int] = {}
    for node in nodes:
        sub = _enumerate(node, multiplier)
        if sub is None:
            return None
        for element, n in sub.items():
            result[element] = result.get(element, 0) + n
    return result


def _count_side(molecules) -> Optional[dict[str, int]]:
    """Element => count over one reaction side, or None if any
    molecule couldn't be enumerated."""
    counts: dict[str, int] = {}
    for molecule in molecules:
        mol_counts = _enumerate(molecule, 1)
        if mol_counts is None:
            return None
        coefficient = molecule.stoichiometric_coefficient()
        for element, n in mol_counts.items():
            counts[element] = counts.get(element, 0) + n * coefficient
    return counts


def _element_diffs(reactants, products) -> str:
    parts = []
    for element in sorted(set(reactants) | set(products)):
        r = reactants.get(element, 0)
        p = products.get(element, 0)
        if r != p:
            parts.append(f"{element}: {r} vs {p}")
    return ", ".join(parts)


def _element_deltas(reactants, products) -> dict[str, list[int]]:
    """Element => [reactant_count, product_count] for every element
    whose counts differ - the numbers behind the message."""
    deltas: dict[str, list[int]] = {}
    for element in sorted(set(reactants) | set(products)):
        r = reactants.get(element, 0)
        p = products.get(element, 0)
        if r != p:
            deltas[element] = [r, p]
    return deltas


def balance(formula: Formula) -> list[Diagnostic]:
    """Verifies that a reaction is stoichiometrically balanced - the
    atom counts on the reactants side equal those on the products
    side. Coefficients and group multiplicities are applied. The
    check ignores electrons (`e^-`), conditions, and the arrow kind.
    It treats unknown constructs conservatively: any molecule it
    can't enumerate atoms for is skipped, not flagged."""
    diagnostics = []
    for reaction in collect_nodes(formula, Reaction):
        reactants = _count_side(reaction.reactants)
        products = _count_side(reaction.products)
        if reactants is None or products is None:
            continue
        if reactants == products:
            continue
        diagnostics.append(Diagnostic(
            "error",
            f"Reaction is not balanced: {_element_diffs(reactants, products)}",
            reaction, details=_element_deltas(reactants, products)))
    return diagnostics


# -- bracket balance -----------------------------------------------------

def bracket_balance(formula: Formula) -> list[Diagnostic]:
    """Verifies group brackets are consistent. The grammar already
    enforces this at parse time, but the linter adds defence in depth
    in case future grammar changes or direct model construction
    produce malformed trees. (The open/close characters derive from
    the bracket kind, so the one property that can drift is an
    unknown kind - that is what the check guards.)"""
    return [
        error(f"Group bracket kind {group.bracket!r} is not one of "
              "paren, square, brace", node=group)
        for group in collect_nodes(formula, Group)
        if group.bracket not in ("paren", "square", "brace")
    ]


# -- charge balance ------------------------------------------------------

_CHARGE_RE = re.compile(r"^(?P<lead>\d*)(?P<sign>[+-])$|^(?P<sign2>[+-])(?P<tail>\d*)$")


def _parse_charge_value(charge: Optional[str]) -> int:
    """Parse a charge string like "2+", "+", "-1", "2-" into an
    integer. 0 for None/empty (atom has no charge)."""
    if not charge:
        return 0
    match = _CHARGE_RE.match(charge)
    if match is None:
        return 0
    if match.group("sign") is not None:
        digits, sign = match.group("lead"), match.group("sign")
    else:
        digits, sign = match.group("tail"), match.group("sign2")
    n = int(digits) if digits else 1
    return n if sign == "+" else -n


def _format_charge(value: int) -> str:
    if value == 0:
        return "0"
    return f"+{value}" if value > 0 else str(value)


def _enumerate_charge(node) -> Optional[int]:
    """Recursively sum atom charges, applying group multiplicities.
    None when the charge of any atom cannot be enumerated."""
    if isinstance(node, Atom):
        return _parse_charge_value(node.charge)
    if isinstance(node, Group):
        raw = node.multiplicity
        mult = int(raw) if raw and raw.isdigit() else 1
        inner = _sum_nodes_charge(node.nodes)
        return None if inner is None else inner * (mult if mult > 0 else 1)
    if isinstance(node, Molecule):
        return _sum_nodes_charge(node.nodes)
    return 0


def _sum_nodes_charge(nodes) -> Optional[int]:
    total = 0
    for node in nodes:
        charge = _enumerate_charge(node)
        if charge is None:
            return None
        total += charge
    return total


def _total_charge(molecules) -> Optional[int]:
    """Sum formal charges on one side of the reaction. None if any
    molecule couldn't be enumerated (mirrors balance's conservative
    skip behaviour)."""
    total = 0
    for molecule in molecules:
        coefficient = int(molecule.coefficient) if molecule.coefficient else None
        coeff = coefficient if coefficient and coefficient > 0 else 1
        inner = _enumerate_charge(molecule)
        if inner is None:
            return None
        total += coeff * inner
    return total


def charge_balance(formula: Formula) -> list[Diagnostic]:
    """Verifies charge conservation in reactions: total formal charge
    on the reactants side equals total on the products side.

    Parallel to balance (which validates atom conservation). Both are
    needed because a reaction can be stoichiometrically balanced but
    charge-imbalanced (e.g. `H+ + OH- -> H_2O+`)."""
    diagnostics = []
    for reaction in collect_nodes(formula, Reaction):
        reactants = _total_charge(reaction.reactants)
        products = _total_charge(reaction.products)
        if reactants is None or products is None:
            continue
        if reactants == products:
            continue
        diagnostics.append(error(
            f"Reaction charge is not balanced: reactants "
            f"{_format_charge(reactants)} vs products "
            f"{_format_charge(products)}", node=reaction))
    return diagnostics


# -- element validation --------------------------------------------------

def element_validation(formula: Formula) -> list[Diagnostic]:
    """Validates that each Atom's element symbol is in the periodic
    table. Catches typos like `Hx`, `Cy`, `Oq` that would otherwise
    silently parse and produce nonsense output.

    Severity is `warning`, not `error` - the parser is total and
    might intentionally accept placeholder elements (e.g. `X` for
    "unknown halogen" in teaching contexts). The linter flags them
    for the user to confirm."""
    return [
        warning(f"Unknown element symbol {atom.element!r}; "
                "not in the periodic table", node=atom)
        for atom in collect_nodes(formula, Atom)
        if not periodic_table.known(atom.element)
    ]


# -- isotope sanity ------------------------------------------------------

def isotope_sanity(formula: Formula) -> list[Diagnostic]:
    """Sanity-checks isotope mass numbers against the element's atomic
    number. The isotope's mass must be >= the atomic number (a
    nucleus can't have fewer protons than its mass number suggests).
    Unknown elements produce an info diagnostic, not an error."""
    diagnostics = []
    for atom in collect_nodes(formula, Atom):
        if not atom.isotope:
            continue
        z = periodic_table.atomic_number(atom.element)
        if z is None:
            diagnostics.append(info(
                f"Element {atom.element!r} not in periodic table; "
                "skipping isotope check", node=atom))
            continue
        mass = int(atom.isotope)
        if mass >= z:
            continue
        diagnostics.append(error(
            f"Isotope mass {mass} is less than atomic number {z} "
            f"for {atom.element}", node=atom))
    return diagnostics


# -- unclosed rings ------------------------------------------------------

def _walk_atoms(node):
    """Atoms of a molecule in source order, recursing through groups
    and nested molecules (the reference's RingBonds.walk_atoms)."""
    if isinstance(node, Atom):
        yield node
    elif isinstance(node, (Group, Molecule)):
        for child in node.nodes:
            yield from _walk_atoms(child)


def _unclosed_atoms(molecule: Molecule) -> list[Atom]:
    """Atoms whose ring_closures digits have no matching partner (the
    reference's RingBonds.unclosed_atoms, same walk and same
    open/close bookkeeping)."""
    open_rings: dict[str, Atom] = {}
    for atom in _walk_atoms(molecule):
        for digit in atom.ring_closures or "":
            if digit in open_rings:
                del open_rings[digit]
            else:
                open_rings[digit] = atom
    return list(open_rings.values())


def unclosed_ring(formula: Formula) -> list[Diagnostic]:
    """Flags ring closure digits that have no matching partner. Catches
    typos like `C1-C-C` (digit 1 opened but never closed)."""
    diagnostics = []
    for molecule in collect_nodes(formula, Molecule):
        for atom in _unclosed_atoms(molecule):
            digits = atom.ring_closures or ""
            diagnostics.append(error(
                f"Atom has unmatched ring closure digit(s) "
                f"{_ruby_inspect_string(digits)}", node=atom))
    return diagnostics


def _ruby_inspect_string(value: str) -> str:
    """Ruby String#inspect shape (the reference's message): the
    string in double quotes."""
    return f'"{value}"'


# -- valence -------------------------------------------------------------

def valence(formula: Formula) -> list[Diagnostic]:
    """Verifies that each atom's total bond order does not exceed the
    element's typical valence. Catches typos like `CH_5` (carbon with
    five bonds). Runs over the perception pass's graph view: every
    atom's bond order counts ALL its bonds (rings and branches
    included - the linear chain approximation missed both)."""
    from .perception import bond_order_sum
    from .structure import build_graph

    diagnostics = []
    for molecule in collect_nodes(formula, Molecule):
        atoms, edges = build_graph(molecule)
        for index, atom in enumerate(atoms):
            load = bond_order_sum(index, edges)
            max_v = periodic_table.max_valence(atom.element)
            if max_v is None:
                diagnostics.append(info(
                    f"Element {atom.element!r} not in valence table; skipping",
                    node=atom))
                continue
            if load <= max_v:
                continue
            diagnostics.append(error(
                f"Atom {atom.element} has bond order {load}; "
                f"max valence is {max_v}", node=atom))
    return diagnostics


# -- identifier format ---------------------------------------------------

def identifier_format(formula: Formula) -> list[Diagnostic]:
    """Validates the *format* of molecule identifier annotations
    (`@cas("...")`, `@inchi("...")`, `@smiles("...")`) whose
    convention has a registered offline validator
    (`asciichem.identifiers`). Lexical/checksum only - checking that
    an identifier is actually *assigned* to a substance is resolution
    (TODO.v2 07 Layer 2), not linting. Conventions without a
    validator (e.g. `iupac`, `cid`) are skipped silently."""
    diagnostics = []
    for molecule in collect_nodes(formula, Molecule):
        for identifier in molecule.identifiers:
            message = identifiers.diagnostic(identifier.convention,
                                             identifier.value)
            if message is None:
                continue
            hill = molecule.hill_formula()
            diagnostics.append(error(
                f"{message} ({identifier.convention} identifier on "
                f"{hill or 'molecule'})", node=molecule))
    return diagnostics


# -- identifier consistency ---------------------------------------------

def _formula_string(counts: dict[str, int]) -> str:
    return "".join(f"{e}{c}" if c != 1 else e
                   for e, c in sorted(counts.items()))


def _inspect_list(items: list[str]) -> str:
    """Ruby-style array inspect (the reference's message shape): each
    string double-quoted - `["C"]`, `[]`."""
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


def _composition_diffs(molecule_counts, inchi_counts) -> str:
    parts = []
    for element in sorted(set(molecule_counts) | set(inchi_counts)):
        actual = molecule_counts.get(element, 0)
        expected = inchi_counts.get(element, 0)
        if actual != expected:
            parts.append(f"{element}: {actual} vs {expected}")
    return ", ".join(parts)


def _smiles_mismatch(counts: dict[str, int], value: str) -> Optional[str]:
    """None when the SMILES element set agrees with the molecule
    (implicit hydrogens tolerated); a message otherwise."""
    smiles_elements = identifiers.Smiles.element_set(value)
    if smiles_elements is None:
        return None  # unanalysable - the format check reports it
    elements = list(counts)
    extra = sorted(set(smiles_elements) - set(elements))
    missing = sorted(set(elements) - set(smiles_elements) - {"H"})
    if not extra and not missing:
        return None
    return (f"SMILES elements {_inspect_list(smiles_elements)} do not match "
            f"molecule {_formula_string(counts)}: "
            f"unexpected {_inspect_list(extra)}, missing {_inspect_list(missing)}")


def identifier_consistency(formula: Formula) -> list[Diagnostic]:
    """Cross-checks identifier values against the molecule they
    annotate, where checkable offline:

    - InChI: the formula layer's element counts must match the
      molecule's composition (error - the formula layer is
      authoritative).
    - SMILES: the element set must agree with the molecule's elements
      (warning - implicit hydrogens are tolerated, so only set
      membership is checkable without a full parser).

    Structural equality (stereo, isotopes, tautomers) needs the InChI
    engine (TODO.v2 10) or resolution (TODO.v2 07 Layer 2) and is out
    of scope here. Identifiers describe the *substance*, not the
    stoichiometric multiplier, so comparisons use the molecule's
    composition without its coefficient."""
    diagnostics = []
    for molecule in collect_nodes(formula, Molecule):
        counts = molecule.element_counts(with_coefficient=False)
        for identifier in molecule.identifiers:
            if identifier.convention == "inchi":
                inchi_counts = identifiers.Inchi.formula_counts(identifier.value)
                if inchi_counts is None:
                    continue  # malformed - the format check reports it
                diffs = _composition_diffs(counts, inchi_counts)
                if not diffs:
                    continue
                diagnostics.append(error(
                    f"InChI formula {_formula_string(inchi_counts)} does not "
                    f"match molecule {_formula_string(counts)}: {diffs}",
                    node=molecule))
            elif identifier.convention == "smiles":
                message = _smiles_mismatch(counts, identifier.value)
                if message is not None:
                    diagnostics.append(warning(message, node=molecule))
    return diagnostics


# -- identity cross-check ------------------------------------------------

_IDENTITY_CONVENTIONS = ("inchi", "inchikey")


def identity_cross_check(formula: Formula) -> list[Diagnostic]:
    """Structural cross-check of identity annotations (TODO.v2 10).
    Where the identifier consistency check compares offline-analysable
    layers (formula composition), this check derives the InChI of the
    drawn structure via the configured engine and compares it to the
    annotated `@inchi`/`@inchikey` - catching transcription errors
    that composition checks cannot (wrong connectivity, stereo,
    isotopes).

    Engine-scaled, mirroring the resolver's opt-in sources: with no
    engine configured the check emits one warning explaining how to
    enable it (protection off is worth saying); with an engine,
    mismatches are errors. Molecules without identity annotations are
    never checked."""
    annotated = [m for m in collect_nodes(formula, Molecule)
                 if any(i.convention in _IDENTITY_CONVENTIONS
                        for i in m.identifiers)]
    if not annotated:
        return []
    if inchi.get_engine() is None:
        return [warning(_skip_message(len(annotated)))]
    diagnostics = []
    for molecule in annotated:
        try:
            identity = inchi.identity_for(molecule)
        except AsciiChemError as exc:
            # Not derivable (e.g. a formula is not a structure: no
            # bonds) or the engine rejected it - report, never raise:
            # the linter stays total.
            diagnostics.append(warning(
                f"cannot derive InChI for cross-check: {exc}", node=molecule))
            continue
        for identifier in molecule.identifiers:
            mismatch = _identity_mismatch(identifier, identity, molecule)
            if mismatch is not None:
                diagnostics.append(mismatch)
    return diagnostics


def _identity_mismatch(identifier, identity, molecule) -> Optional[Diagnostic]:
    if identifier.convention == "inchi":
        if identifier.value == identity.inchi:
            return None
        return error(
            "InChI annotation does not match the drawn structure: "
            f"annotated {identifier.value}, computed {identity.inchi}",
            node=molecule)
    if identifier.convention == "inchikey":
        if identifier.value == identity.inchikey:
            return None
        return error(
            "InChIKey annotation does not match the drawn structure: "
            f"annotated {identifier.value}, computed {identity.inchikey}",
            node=molecule)
    return None


def _skip_message(count: int) -> str:
    return (f"structural identity cross-check skipped for {count} annotated "
            f"molecule(s): no InChI engine configured - {inchi.INSTALL_GUIDE}")


# -- crystal sanity ------------------------------------------------------

def _format_number(value) -> str:
    """Render a cell parameter the way the reference prints the raw
    authored value: integral floats without the trailing `.0`."""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def crystal_sanity(formula: Formula) -> list[Diagnostic]:
    """Validates unit-cell parameters on a Crystal node:

    - lengths (a, b, c) must be positive
    - angles (alpha, beta, gamma) must be in (0, 180] degrees
    - fractional coordinates on atoms must be in [0, 1)

    Each cell parameter is optional in the model (a partial crystal
    description is allowed), so missing values are skipped, not
    flagged."""
    diagnostics = []
    for crystal in collect_nodes(formula, Crystal):
        for dim in Crystal.LENGTH_FIELDS:
            diagnostics.extend(
                _cell_param(crystal, dim, lambda num: num > 0,
                            f"Crystal {dim} must be positive, got "))
        for dim in Crystal.ANGLE_FIELDS:
            diagnostics.extend(
                _cell_param(crystal, dim, lambda num: 0 < num <= 180,
                            f"Crystal {dim} must be in (0, 180], got "))
        for atom in crystal.atoms:
            for attr in ("x_fract", "y_fract", "z_fract"):
                value = getattr(atom, attr)
                if value is None:
                    continue
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= num < 1:
                    continue
                diagnostics.append(warning(
                    f"Fractional coordinate {attr}={_format_number(value)} on "
                    f"{atom.element} is outside [0, 1)", node=crystal))
    return diagnostics


def _cell_param(crystal, dim, ok, prefix: str) -> list[Diagnostic]:
    value = getattr(crystal, dim)
    if value is None:
        return []
    try:
        num = float(value)
    except (TypeError, ValueError):
        return []
    if ok(num):
        return []
    return [error(f"{prefix}{_format_number(value)}", node=crystal)]


# -- spectrum peaks ------------------------------------------------------

_LEADING_NUMBER = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


def _leading_number(raw) -> Optional[float]:
    """Extract a leading signed number from a string like "-3H" ->
    -3.0, "100%" -> 100.0, "abc" -> None. Tolerant of unit suffixes
    which are common in peak data (H, %, coulombs, etc.)."""
    if raw is None:
        return None
    match = _LEADING_NUMBER.match(str(raw))
    return float(match.group(1)) if match else None


def spectrum_peaks(formula: Formula) -> list[Diagnostic]:
    """Validates spectrum peak data:

    - peak positions must be parseable numbers (warn on non-numeric)
    - intensities, when numeric, must be non-negative

    Does NOT enforce domain-specific ranges (e.g. 0-12 ppm for 1H
    NMR, 0-220 ppm for 13C NMR) - those depend on instrument and
    solvent and would over-fit. The structural sanity check is
    enough."""
    diagnostics = []
    for spectrum in collect_nodes(formula, Spectrum):
        for peak in spectrum.peaks:
            position = _leading_number(peak.position)
            if peak.position is not None:
                if position is None:
                    diagnostics.append(warning(
                        f"Spectrum peak position {peak.position!r} "
                        "is not numeric", node=spectrum))
                elif position < 0:
                    diagnostics.append(warning(
                        f"Spectrum peak position {_format_number(position)} "
                        "is negative", node=spectrum))
            intensity = _leading_number(peak.intensity)
            if intensity is not None and intensity < 0:
                diagnostics.append(warning(
                    f"Spectrum peak intensity {_format_number(intensity)} "
                    "is negative", node=spectrum))
    return diagnostics


# -- z-matrix references -------------------------------------------------

def zmatrix_references(formula: Formula) -> list[Diagnostic]:
    """Validates Z-Matrix references: each ref1/ref2/ref3 must name an
    atom that appears in an earlier row. Catches ordering mistakes
    that would otherwise produce dangling references downstream.

    Also checks geometric sanity: distances must be positive, angles
    in (0, 180], dihedrals in [-180, 180]."""
    diagnostics = []
    for zmatrix in collect_nodes(formula, ZMatrix):
        seen: set[str] = set()
        for row in zmatrix.rows:
            seen.add(row.atom)
            for ref in (row.ref1, row.ref2, row.ref3):
                if ref is None or ref in seen:
                    continue
                diagnostics.append(error(
                    f"ZMatrix row references {_ruby_inspect_string(ref)} "
                    "before it is defined", node=zmatrix))
        for row in zmatrix.rows:
            diagnostics.extend(_zmatrix_geometry(row, zmatrix))
    return diagnostics


def _zmatrix_geometry(row: ZRow, zmatrix: ZMatrix) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for attr, ok, text in (
            ("distance", lambda v: v > 0,
             "ZMatrix distance must be positive, got "),
            ("angle", lambda v: 0 < v <= 180,
             "ZMatrix angle must be in (0, 180], got "),
            ("dihedral", lambda v: -180 <= v <= 180,
             "ZMatrix dihedral must be in [-180, 180], got ")):
        raw = getattr(row, attr)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if ok(value):
            continue
        out.append(error(f"{text}{_format_number(raw)}", node=zmatrix))
    return out


# -- registry ------------------------------------------------------------

# Check registry. Each entry maps a name to its check function;
# iteration order is insertion order (the reference's registration
# order). Extending: `register("name", check)` - the registry is
# open, the runner is closed.
REGISTRY: dict = {}


def register(name: str, check) -> None:
    if name not in REGISTRY:
        REGISTRY[name] = check


def builtin() -> dict:
    """The built-in registry (a copy): every check this module ports,
    in the reference's registration order. Names mirror the
    reference's registry entries."""
    return dict(REGISTRY)


def run(formula: Formula, registry: Optional[dict] = None) -> list[Diagnostic]:
    """Run every registered check against the model. Returns a list of
    Diagnostic objects (empty if no issues), sorted by severity
    (errors first) then by message for stable output, deduplicated on
    (message, node identity) - the reference's sort/uniq."""
    checks = registry if registry is not None else REGISTRY
    diagnostics: list[Diagnostic] = []
    for check in checks.values():
        diagnostics.extend(check(formula))
    diagnostics.sort(key=lambda d: (SEVERITY_ORDER.get(d.severity, 99),
                                    d.message))
    seen: set[tuple] = set()
    unique: list[Diagnostic] = []
    for d in diagnostics:
        key = (d.message, id(d.node) if d.node is not None else None)
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def errors(formula: Formula) -> bool:
    """Convenience: true if any check produced an error."""
    return any(d.severity == "error" for d in run(formula))


# Built-in registration (the reference's `register :name` calls, in
# registration order).
register("balance", balance)
register("bracket_balance", bracket_balance)
register("charge_balance", charge_balance)
register("crystal_sanity", crystal_sanity)
register("element_validation", element_validation)
register("identifier_consistency", identifier_consistency)
register("identifier_format", identifier_format)
register("identity_cross_check", identity_cross_check)
register("isotope_sanity", isotope_sanity)
register("spectrum_peaks", spectrum_peaks)
register("unclosed_ring", unclosed_ring)
register("valence", valence)
register("zmatrix_references", zmatrix_references)
