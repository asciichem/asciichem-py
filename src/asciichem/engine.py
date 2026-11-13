"""The AsciiChem parse engine: the PARG artifact engine, period
(TODO.v2/17 slice 9; the maintainer's PARG-only directive). The
hand-ported recursive-descent layer is deleted — the grammar is NOT
reimplemented in Python. This module consumes the compiled AsciiChem
PARG artifact (vendored at ``parg/asciichem.json``, sourced from the
asciichem-grammar contract; the same artifact asciichem-rs embeds and the
Ruby reference vendors), verifies its checksum through ``parsanol``
(the Python runtime for PARG artifacts, driving ``libparsanol`` — the
same portable engine every runtime ships), and adapts the shaped tree
into the model the rest of the package renders.

The adaptation mirrors the Ruby reference's PargEngine: TreeAdapter
unwraps the artifact's captured structural rules into flat shapes,
then the builder walks capture-for-capture into the model — the same
capture names TransformRules consumes on the Ruby side. Builder
helpers carried over from the removed rd layer (AtomBuilder charge/
oxidation disambiguation, the crystal/spectrum/calculation/mechanism
body parsers) operate on captured strings only; the grammar lives in
the artifact.

Parse rejections surface as ``ParseError``; a match that stops short
of the (trimmed) input is an error too — never a silently partial
parse. Deeply nested input can exceed a default thread stack (the
corpus fuzz cases), so oversized parses run on a dedicated worker
thread with a generous stack, mirroring the Rust core.
"""
from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

from parsanol import ParseFailed, PargArtifact

from .model import (Atom, Bond, Calculation, CalculationProperty, Crystal,
                    ElectronConfiguration, EmbeddedMath, Formula, Group,
                    Identifier, Mechanism, MechanismFlow, MechanismStep,
                    Molecule, Name, ParseError, Reaction, ReactionCascade,
                    ReactionConditions, Spectrum, SpectrumPeak, Text, ZMatrix,
                    ZRow)

# The artifact entry holding the whole AsciiChem grammar.
ENTRY = "formula"

ARTIFACT_PATH = Path(__file__).parent / "parg" / "asciichem.json"

# Version the vendored artifact was built from; the load fails loudly
# on drift so the engine never runs a stale grammar.
PARG_GRAMMAR_VERSION = "0.4.0"

ARROW_TOKENS = {"<=>": "equilibrium", "<->": "resonance", "->": "forward", "<-": "reverse"}
STEREO = {"alpha": "alpha", "beta": "beta", "R": "R", "S": "S", "E": "E", "Z": "Z",
          "α": "alpha", "β": "beta", "a": "alpha", "b": "beta"}
IDENTIFIER_TYPES = {"inchi", "smiles", "cas", "iupac", "cid", "chebi"}
BOND_KINDS = {"direction_up": ("single", "up"), "direction_down": ("single", "down"),
              "quadruple": ("quadruple", None), "wedge": ("wedge", None),
              "hash": ("hash", None), "dative": ("dative", None),
              "wavy": ("wavy", None), "triple": ("triple", None),
              "double": ("double", None), "single": ("single", None)}
BRACKETS = {"(": "paren", "[": "square", "{": "brace"}

# Slice wire objects ({offset, length, line, column, value}) — the
# native engine's captured-text form.
_SLICE_KEYS = {"offset", "length", "line", "column", "value"}


def is_slice(node) -> bool:
    return isinstance(node, dict) and set(node) == _SLICE_KEYS


# -- artifact -----------------------------------------------------------

_ARTIFACT: PargArtifact | None = None


def _artifact() -> PargArtifact:
    global _ARTIFACT
    if _ARTIFACT is None:
        artifact = PargArtifact.from_path(ARTIFACT_PATH)
        version = artifact.envelope.get("version")
        if version != PARG_GRAMMAR_VERSION:
            raise ValueError(
                f"vendored PARG artifact version {version!r} does not match "
                f"the engine's pinned grammar {PARG_GRAMMAR_VERSION!r} — "
                "re-vendor from asciichem-grammar")
        _ARTIFACT = artifact
    return _ARTIFACT


# -- deep-stack worker ---------------------------------------------------
#
# Stack reserved for the recursive PEG engine. Deeply nested input (the
# corpus fuzz cases) exceeds a default thread stack, so oversized parses
# run on a dedicated thread with a generous stack — reserved virtually,
# committed only as used (mirrors the Rust core).

PARSE_STACK_BYTES = 64 * 1024 * 1024

# Inputs at or below this length parse on the CALLER's thread: PEG
# recursion depth is bounded by input length, so a short input cannot
# exhaust a default 8 MB stack — and skipping the worker hop is worth
# hundreds of µs per parse. Longer inputs go to the worker.
INLINE_PARSE_MAX = 16 * 1024

_WORKER: "_DeepStackWorker | None" = None
_WORKER_GUARD = threading.Lock()


def _worker() -> "_DeepStackWorker":
    global _WORKER
    if _WORKER is None:
        with _WORKER_GUARD:
            if _WORKER is None:
                worker = _DeepStackWorker()
                previous = threading.stack_size()
                try:
                    threading.stack_size(PARSE_STACK_BYTES)
                    worker.thread = threading.Thread(
                        target=worker._loop, daemon=True, name="asciichem-parse")
                    worker.thread.start()
                finally:
                    threading.stack_size(previous)
                _WORKER = worker
    return _WORKER


class _DeepStackWorker:
    """A persistent parse worker: one thread with the deep stack,
    spawned once, reused for every oversized parse."""

    def __init__(self):
        self.thread: threading.Thread | None = None
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()

    @staticmethod
    def _loop(jobs: queue.SimpleQueue) -> None:
        while True:
            fn, result = jobs.get()
            try:
                result.put((True, fn()))
            except BaseException as error:  # delivered to the caller
                result.put((False, error))

    def run(self, fn):
        if self.thread is not None and threading.current_thread() is self.thread:
            return fn()  # never enqueue from the worker itself (self-deadlock)
        result: queue.SimpleQueue = queue.SimpleQueue()
        self.jobs.put((fn, result))
        ok, value = result.get()
        if not ok:
            raise value
        return value


# The native engine's parse context is process-wide; calls are
# serialised under this lock (the Rust core runs one worker thread for
# the same reason). Reentrant so body re-parses (crystal atoms) nest
# safely.
_NATIVE_LOCK = threading.RLock()

def _parse_native(artifact: PargArtifact, text: str):
    """One native parse call, serialised and stack-sited."""
    if len(text) <= INLINE_PARSE_MAX:
        with _NATIVE_LOCK:
            return artifact.parse(text, ENTRY)
    worker = _worker()
    return worker.run(lambda: _parse_native(artifact, text))


# -- tree adaptation ------------------------------------------------------
#
# Unwraps the artifact tree's captured structural rules into the flat
# shapes the builder matches on (port of the Ruby PargEngine::TreeAdapter).

# Sequence captures whose merged array form collapses to the single
# hash the builder carries.
ARRAY_MERGE_KEYS = ("cascade", "molecule", "annotated_molecule")

# Captured structural rules whose single-key hash collapses to the
# shape the builder matches on.
WRAPPERS = ("molecule", "reaction", "annotated_molecule", "segment",
            "plus_term", "annotation_item", "formula_tail", "condition",
            "embedded_math")


def consumed_length(node) -> int:
    """How much input the tree covers (offset+length over slices,
    summed across hash siblings, max across array items)."""
    if is_slice(node):
        return node["offset"] + node["length"]
    if isinstance(node, dict):
        return sum(consumed_length(value) for value in node.values())
    if isinstance(node, list):
        return max((consumed_length(member) for member in node), default=0)
    return 0


def adapt(node):
    if is_slice(node):
        return node["value"]
    if isinstance(node, dict):
        return normalise({key: adapt(value) for key, value in node.items()})
    if isinstance(node, list):
        return [adapt(member) for member in node]
    return node


def normalise(hash_in: dict):
    if len(hash_in) == 1:
        return unwrap_single_key(hash_in)
    out = dict(hash_in)
    for key in ARRAY_MERGE_KEYS:
        if key in out and isinstance(out[key], list):
            out[key] = merge_items(out[key])
    return out


def unwrap_single_key(hash_in: dict):
    (key, value), = hash_in.items()
    if key == "embedded_math":
        return {"math_source": math_source_of(value)}
    if key in WRAPPERS:
        return value
    if key == "group":
        return merge_multiplicity(merge_items(value))
    return hash_in


def capture_text(value, default: str = "") -> str:
    """A captured string. parslet's zero-match form (an empty array)
    normalises to `default` — the rd layer rendered empty captures as
    the empty string (e.g. `` empty math, `@name("")`)."""
    if value is None:
        return default
    if isinstance(value, list) and not value:
        return default
    return value


def math_source_of(inner):
    if isinstance(inner, dict):
        return capture_text(inner["math_source"])
    return capture_text(inner)


def merge_multiplicity(merged: dict):
    if not merged.get("multiplicity"):
        merged["multiplicity"] = None
    return merged


def merge_items(items):
    """Captured sequences with embedded repetitions yield arrays of
    per-child capture hashes; merge them into the single hash the
    builder carries (later duplicate keys append, matching parslet's
    repeated-capture behavior)."""
    if not isinstance(items, list):
        return items
    merged: dict = {}
    for item in items:
        for key, new in item.items():
            old = merged.get(key)
            merged[key] = old + (new if isinstance(new, list) else [new]) \
                if isinstance(old, list) else new
    return merged


# -- builders ----------------------------------------------------------
#
# Each builder turns captured strings into typed model fields. The
# AtomBuilder charge/oxidation disambiguation is carried over verbatim
# from the rd layer (sign-first charges canonicalise to
# number-then-sign).

def strip_marker(value, marker=None):
    if value is None:
        return None
    s = value
    if marker and s.startswith(marker):
        s = s[1:]
    if s[:1] in "^_":
        s = s[1:]
    return s or None


def detect_charge(s):
    if not s:
        return None
    m = re.fullmatch(r"(\d*)([+-])", s)
    if m:
        return (m.group(1) or "") + m.group(2) if m.group(1) else m.group(2)
    m = re.fullmatch(r"([+-])(\d*)", s)
    if m:
        # Sign-first canonicalises to number-then-sign ("+2" -> "2+").
        return (m.group(2) or "") + m.group(1) if m.group(2) else m.group(1)
    return None


def detect_oxidation(s):
    if not s:
        return None
    m = re.fullmatch(r"\(([IVXLCDM]+)\)", s)
    return m.group(1) if m else None


def _float_or_none(value):
    return float(value) if value is not None else None


def build_atom(attrs: dict) -> Atom:
    sup = strip_marker(attrs.get("superscript"))
    charge = detect_charge(sup)
    oxidation = detect_oxidation(sup)
    return Atom(
        element=attrs.get("element") or "",
        isotope=strip_marker(attrs.get("isotope")),
        subscript=strip_marker(attrs.get("subscript"), "_"),
        superscript=None if (charge or oxidation) else sup,
        charge=charge,
        oxidation_state=oxidation,
        lone_pairs=len(attrs["lone_pairs"]) if attrs.get("lone_pairs") else None,
        radical_electrons=len(attrs["radical_electrons"]) if attrs.get("radical_electrons") else None,
        ring_closures=attrs.get("ring_closures") or None,
        x2=_float_or_none(attrs.get("x2")),
        y2=_float_or_none(attrs.get("y2")),
        z2=_float_or_none(attrs.get("z2")),
        atom_parity=attrs.get("atom_parity"),
        chirality=attrs.get("chirality"),
        spin_multiplicity=attrs.get("spin_multiplicity"),
        atom_title=attrs.get("atom_title"),
        x_fract=_float_or_none(attrs.get("x_fract")),
        y_fract=_float_or_none(attrs.get("y_fract")),
        z_fract=_float_or_none(attrs.get("z_fract")),
    )


def build_unit(unit: dict):
    if "atom" in unit:
        return build_atom(unit["atom"])
    if "open_bracket" in unit:
        return build_group(unit)
    for key, (kind, direction) in BOND_KINDS.items():
        if key in unit:
            return Bond(kind, direction=direction)
    raise ParseError(f"unrecognised molecule unit shape: {sorted(unit)}")


def build_molecule(mol: dict) -> Molecule:
    stereo = STEREO.get(mol["stereo"]) if "stereo" in mol else None
    coefficient = mol["coefficient"].get("value") if "coefficient" in mol else None
    units = [build_unit(unit) for unit in mol.get("units") or []]
    return Molecule(nodes=units, coefficient=coefficient, stereo=stereo)


def build_group(group: dict) -> Group:
    nodes = [build_node(node) for node in group.get("group_nodes") or []]
    multiplicity = group.get("multiplicity")
    if isinstance(multiplicity, str) and multiplicity.startswith("_"):
        multiplicity = multiplicity[1:]
    return Group(nodes=nodes, multiplicity=multiplicity,
                 bracket=BRACKETS.get(group.get("open_bracket"), "paren"))


def condition_text(node):
    """The bracket text of an arrow condition. An empty `[]` capture
    arrives as parslet's zero-match form (an empty array) — absent,
    never an empty string."""
    if node is None:
        return None
    text = node.get("text") if isinstance(node, dict) else node
    if text is None or (isinstance(text, list) and not text):
        return None
    return text


def as_terms(side):
    if isinstance(side, list):
        return side
    if side is None:
        return []
    return [side]


def _reaction_from(reactants: list, products: list, arrow: dict) -> Reaction:
    kind = ARROW_TOKENS.get(arrow.get("kind"), "forward")
    above = condition_text(arrow.get("above"))
    below = condition_text(arrow.get("below"))
    conditions = ReactionConditions(above, below) if (above or below) else None
    return Reaction(reactants, products, arrow=kind, conditions=conditions)


def build_reaction(reaction: dict) -> Reaction:
    reactants = [build_molecule(term) for term in as_terms(reaction.get("reactants"))]
    products = [build_molecule(term) for term in as_terms(reaction.get("products"))]
    return _reaction_from(reactants, products, reaction.get("arrow") or {})


def build_cascade(data) -> ReactionCascade:
    if isinstance(data, list):
        data = next((item for item in data
                     if isinstance(item, dict) and "first" in item), None)
    first = (data or {}).get("first")
    segments = (data or {}).get("segments")
    if segments is None:
        segments = []
    elif not isinstance(segments, list):
        segments = [segments]
    steps = [build_reaction(first)]
    for segment in segments:
        products = [build_molecule(term) for term in as_terms(segment.get("products"))]
        steps.append(_reaction_from(steps[-1].products, products,
                                    segment.get("arrow") or {}))
    return ReactionCascade(steps)


def strip_quotes(s: str) -> str:
    return s[1:-1] if s.startswith('"') and s.endswith('"') else s


def build_node(node: dict):
    """One formula/group node, dispatched on the adapted capture keys."""
    if "mol" in node:
        molecule = build_molecule(node["mol"])
        for ann in node.get("annotations") or []:
            apply_annotation(molecule, annotation_tuple(ann))
        return molecule
    if "cascade" in node:
        return build_cascade(node["cascade"])
    if "reactants" in node or "products" in node or "arrow" in node:
        return build_reaction(node)
    if "electron_config" in node:
        pairs = [(pair["orbital"], pair["occupancy"])
                 for pair in node["electron_config"]]
        return ElectronConfiguration(pairs)
    if "crystal_node" in node:
        data = node["crystal_node"]
        if isinstance(data, dict):
            return build_crystal(data.get("crystal_name"),
                                 data.get("crystal_params"),
                                 data.get("crystal_body"))
        return build_crystal(None, None, None)  # bare "crystal"
    if "spectrum_node" in node:
        data = node["spectrum_node"]
        if isinstance(data, dict):
            return build_spectrum(data.get("spectrum_type"),
                                  data.get("spectrum_params"),
                                  data.get("spectrum_body"))
        return build_spectrum(None, None, None)  # bare "spectrum"
    if "calc_node" in node:
        data = node["calc_node"]
        if isinstance(data, dict):
            return build_calculation(data.get("calc_params"),
                                     data.get("calc_body"))
        return build_calculation(None, None)  # bare "calc"
    if "zmatrix_node" in node:
        data = node["zmatrix_node"]
        return build_zmatrix(data.get("zmatrix_body") if isinstance(data, dict)
                             else None)  # bare "zmatrix"
    if "mechanism_node" in node:
        data = node["mechanism_node"]
        return build_mechanism(data.get("mechanism_body") if isinstance(data, dict)
                               else None)  # bare "mechanism"
    if "open_bracket" in node:
        return build_group(node)
    if "units" in node or "stereo" in node or "coefficient" in node:
        return build_molecule(node)
    if "math_source" in node:
        return EmbeddedMath(node["math_source"])
    if "text_run" in node:
        return Text(strip_quotes(node["text_run"]))
    if "group_text_run" in node:
        return Text(strip_quotes(node["group_text_run"]))
    raise ParseError(f"unrecognised node shape: {sorted(node)}")


def annotation_tuple(ann: dict):
    if "meta_key" in ann:
        return ("meta", capture_text(ann["meta_key"]), capture_text(ann["meta_value"]))
    return ("ann", ann.get("ann_type"), capture_text(ann.get("ann_value")))


def apply_annotation(molecule: Molecule, ann):
    kind = ann[0]
    if kind == "meta":
        molecule.metadata.append({"name": ann[1], "content": ann[2]})
        return
    _, ann_type, value = ann
    if ann_type == "name":
        molecule.names.append(Name(value))
    elif ann_type == "title":
        molecule.title = value
    elif ann_type == "formula":
        molecule.formulas.append({"concise": value})
    elif ann_type == "label":
        molecule.labels.append({"value": value})
    elif ann_type in IDENTIFIER_TYPES:
        molecule.identifiers.append(Identifier(value, ann_type))
    else:
        molecule.properties.append({"title": ann_type, "value": value})


# -- body languages -----------------------------------------------------
#
# The artifact captures crystal/spectrum/calculation/zmatrix/mechanism
# bodies as raw strings; these builders parse the body text (carried
# over from the rd layer, semantics unchanged).

def parse_params(raw):
    params = {}
    if not raw:
        return params
    for pair in raw.split(","):
        key, _, value = pair.partition("=")
        if key:
            params[key.strip()] = value.strip()
    return params


def validate_colon_lines(body, what):
    for i, line in enumerate(body.split("\n")):
        stripped = line.strip()
        if stripped and ":" not in stripped:
            raise ParseError(
                f"{what} on line {i + 1} is missing ':' separator: {stripped!r}")


def split_mechanism_step(label: str, value: str):
    """A step value is its reaction, optionally followed by
    `| flow: SOURCE > TARGET` clauses (semicolon-separated). Unknown
    clause kinds are rejected, never dropped."""
    parts = [part.strip() for part in value.split("|")]
    reaction, clauses = parts[0], parts[1:]
    flows = []
    for clause in clauses:
        kind, colon, payload = clause.partition(":")
        if not colon or kind.strip() != "flow":
            raise ParseError(
                f"mechanism step {label!r} has an unknown clause: {clause!r} "
                '(supported: "flow: SOURCE > TARGET")')
        for arrow in payload.split(";"):
            arrow = arrow.strip()
            if not arrow:
                continue
            source, gt, target = arrow.partition(">")
            source, target = source.strip(), target.strip()
            if not gt or not source or not target:
                raise ParseError(
                    f"mechanism step {label!r} has a malformed flow: {arrow!r} "
                    '(expected "SOURCE > TARGET")')
            flows.append(MechanismFlow(source, target))
    return reaction, flows


def build_crystal(name, params_raw, body) -> Crystal:
    params = parse_params(params_raw)
    atoms = []
    if body:
        inner = parse_text(body)
        for node in inner.nodes:
            if isinstance(node, Molecule):
                atoms.extend(n for n in node.nodes if isinstance(n, Atom))

    def num(key):
        return float(params[key]) if key in params else None

    return Crystal(name=name or None, a=num("a"), b=num("b"), c=num("c"),
                   alpha=num("alpha"), beta=num("beta"), gamma=num("gamma"),
                   spacegroup=params.get("sg"), atoms=atoms)


def build_spectrum(technique, params_raw, body) -> Spectrum:
    peaks = []
    if body:
        validate_colon_lines(body, "spectrum peak")
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            assignment = None
            m = re.search(r'"([^"]*)"', line)
            if m:
                assignment = m.group(1)
                line = re.sub(r'"[^"]*"', "", line).strip()
            position, _, rest = line.partition(":")
            tokens = rest.split()
            peaks.append(SpectrumPeak(position.strip() or None,
                                      tokens[0] if tokens else None,
                                      tokens[1] if len(tokens) > 1 else None,
                                      assignment))
    return Spectrum(technique=technique or None, params=parse_params(params_raw),
                    peaks=peaks)


def build_calculation(params_raw, body) -> Calculation:
    method = basis = None
    if params_raw:
        parts = params_raw.split("/", 1)
        method = parts[0].strip() or None
        basis = parts[1].strip() if len(parts) > 1 else None
    properties = []
    if body:
        validate_colon_lines(body, "calculation property")
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            key, _, rest = line.partition(":")
            if not key:
                continue
            tokens = rest.split()
            properties.append(CalculationProperty(
                key.strip(), tokens[0] if tokens else "",
                tokens[1] if len(tokens) > 1 else None))
    return Calculation(method, basis, properties)


def build_zmatrix(body) -> ZMatrix:
    rows = []
    if body:
        for line in body.split("\n"):
            tokens = line.split()
            if not tokens:
                continue
            rows.append(ZRow(
                atom=tokens[0],
                ref1=tokens[1] if len(tokens) > 1 else None,
                distance=tokens[2] if len(tokens) > 2 else None,
                ref2=tokens[3] if len(tokens) > 3 else None,
                angle=tokens[4] if len(tokens) > 4 else None,
                ref3=tokens[5] if len(tokens) > 5 else None,
                dihedral=tokens[6] if len(tokens) > 6 else None))
    return ZMatrix(rows)


def build_mechanism(body) -> Mechanism:
    steps, spectators = [], []
    if body:
        validate_colon_lines(body, "mechanism entry")
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            key, _, value = line.partition(":")
            if not value:
                continue
            if key.strip() == "spectator":
                spectators.extend(value.split())
            else:
                reaction, flows = split_mechanism_step(key.strip(),
                                                       value.strip())
                steps.append(MechanismStep(key.strip(), reaction, flows))
    return Mechanism(steps, spectators)


# -- entry ---------------------------------------------------------------

def _parse_tree(source: str):
    """The strict shaped tree for the trimmed input, or ParseError."""
    text = source.strip()
    try:
        return _parse_native(_artifact(), text)
    except ParseFailed as failure:
        wire = failure.wire or {}
        offset = wire.get("offset")
        expected = wire.get("expected") or []
        what = ", ".join(str(item) for item in expected[:4]) if expected else "no further input"
        position = f"at char {offset + 1}" if isinstance(offset, int) else "of input"
        raise ParseError(f"Parse error {position}: expected {what}") from None


def parse_text(source: str) -> Formula:
    raw = _parse_tree(source)
    text = source.strip()
    covered = consumed_length(raw)
    if covered < len(text):
        raise ParseError(
            f"input not fully captured: consumed {covered} of {len(text)} bytes")
    formula = adapt(raw).get("formula")
    if formula is None:
        nodes = []
    elif isinstance(formula, list):
        nodes = formula
    else:
        nodes = [formula]
    return Formula([build_node(node) for node in nodes])
