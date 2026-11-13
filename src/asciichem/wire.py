"""Canonical wire form (asciichem-model v1 JSON). Emission carries
the same fuzz-junk guards as the reference implementations
(non-digit subscript/isotope/coefficient values are omitted); the
core-set ingestion mirrors them too."""
from __future__ import annotations

import json
import re
from typing import Optional

from .model import (Atom, Bond, Calculation, Crystal, ElectronConfiguration,
                    EmbeddedMath, Formula, Group, Mechanism, Molecule, Name,
                    ParseError, Reaction, ReactionCascade, ReactionConditions,
                    Spectrum, Text, ZMatrix)

DIGITS = re.compile(r"^\d+$")


def _digits_or_none(s: Optional[str]) -> Optional[str]:
    return s if s is not None and DIGITS.match(s) else None


def to_model_json(node, indent=None) -> str:
    return json.dumps(to_wire(node), indent=indent)


def to_wire(node) -> dict:
    if isinstance(node, Formula):
        return {"type": "formula", "nodes": [to_wire(n) for n in node.nodes]}
    if isinstance(node, Molecule):
        wire = {"type": "molecule", "nodes": [to_wire(n) for n in node.nodes]}
        if (coefficient := _digits_or_none(node.coefficient)) is not None:
            wire["coefficient"] = coefficient
        if node.stereo is not None:
            wire["stereo"] = node.stereo
        if node.identifiers:
            wire["identifiers"] = [
                {"type": "identifier", "value": i.value, "convention": i.convention}
                for i in node.identifiers]
        return wire
    if isinstance(node, Atom):
        wire = {"type": "atom", "element": node.element}
        if (isotope := _digits_or_none(node.isotope)) is not None:
            wire["isotope"] = isotope
        if node.charge is not None:
            wire["charge"] = node.charge
        if (subscript := _digits_or_none(node.subscript)) is not None:
            wire["subscript"] = subscript
        if node.oxidation_state is not None:
            wire["oxidationState"] = node.oxidation_state
        if node.lone_pairs is not None:
            wire["lonePairs"] = node.lone_pairs
        if node.radical_electrons is not None:
            wire["radicalElectrons"] = node.radical_electrons
        if (ring := _digits_or_none(node.ring_closures)) is not None:
            wire["ringClosures"] = ring
        if node.aromatic is not None:
            wire["aromatic"] = node.aromatic
        if node.chirality is not None:
            wire["chirality"] = node.chirality
        if node.hydrogens is not None:
            wire["hydrogens"] = node.hydrogens
        return wire
    if isinstance(node, Bond):
        # kind is always emitted: the reference implementation does,
        # and the differential gate compares wire forms byte-wise.
        wire = {"type": "bond", "kind": node.kind}
        if node.direction is not None:
            wire["direction"] = node.direction
        return wire
    if isinstance(node, Group):
        wire = {"type": "group", "nodes": [to_wire(n) for n in node.nodes]}
        if (multiplicity := _digits_or_none(node.multiplicity)) is not None:
            wire["multiplicity"] = multiplicity
        if node.bracket != "paren":
            wire["bracket"] = node.bracket
        return wire
    if isinstance(node, Reaction):
        wire = {"type": "reaction",
                "reactants": [to_wire(m) for m in node.reactants],
                "products": [to_wire(m) for m in node.products]}
        if node.arrow != "forward":
            wire["arrow"] = node.arrow
        if node.conditions and (node.conditions.above or node.conditions.below):
            conditions = {}
            if node.conditions.above is not None:
                conditions["above"] = node.conditions.above
            if node.conditions.below is not None:
                conditions["below"] = node.conditions.below
            wire["conditions"] = conditions
        return wire
    if isinstance(node, ReactionCascade):
        return {"type": "reaction-cascade",
                "steps": [to_wire(s) for s in node.steps]}
    if isinstance(node, ElectronConfiguration):
        wire = {"type": "electron-configuration",
                "orbitals": [{"orbital": o, "occupancy": occ} for o, occ in node.orbitals]}
        if node.term_symbol:
            m, letter, j = node.term_symbol
            term = {}
            if m is not None:
                term["multiplicity"] = m
            if letter is not None:
                term["letter"] = letter
            if j is not None:
                term["jValue"] = j
            wire["termSymbol"] = term
        return wire
    if isinstance(node, EmbeddedMath):
        return {"type": "embedded-math", "source": node.source}
    if isinstance(node, Text):
        return {"type": "text", "content": node.content}
    if isinstance(node, Crystal):
        wire = {"type": "crystal"}
        if node.name is not None:
            wire["name"] = node.name
        for key in ("a", "b", "c", "alpha", "beta", "gamma"):
            value = getattr(node, key)
            if value is not None:
                # The cell parameters carry NUMBERS on the wire (the
                # shared v1 crystal schema's type:number) - the model
                # keeps the authored strings and the emit casts. A
                # non-numeric capture is a malformed record, not a
                # silent null (the reference's cast).
                try:
                    wire[key] = float(value)
                except ValueError:
                    raise ParseError(
                        f"crystal cell param {value!r} is not a number")
        if node.spacegroup is not None:
            wire["spacegroup"] = node.spacegroup
        if node.atoms:
            wire["atoms"] = [to_wire(a) for a in node.atoms]
        return wire
    if isinstance(node, Spectrum):
        wire = {"type": "spectrum"}
        if node.technique is not None:
            wire["technique"] = node.technique
        if node.params:
            wire["params"] = dict(node.params)
        if node.peaks:
            peaks = []
            for p in node.peaks:
                peak = {"position": p.position or ""}
                if p.intensity is not None:
                    peak["intensity"] = p.intensity
                if p.multiplicity is not None:
                    peak["multiplicity"] = p.multiplicity
                if p.assignment is not None:
                    peak["assignment"] = p.assignment
                peaks.append(peak)
            wire["peaks"] = peaks
        return wire
    if isinstance(node, Calculation):
        wire = {"type": "calculation"}
        if node.method is not None:
            wire["method"] = node.method
        if node.basis is not None:
            wire["basis"] = node.basis
        if node.properties:
            wire["properties"] = [
                {"title": p.title, "value": p.value,
                 **({"units": p.units} if p.units is not None else {})}
                for p in node.properties]
        return wire
    if isinstance(node, ZMatrix):
        wire = {"type": "zmatrix"}
        if node.rows:
            rows = []
            for r in node.rows:
                row = {"atom": r.atom}
                for key in ("ref1", "distance", "ref2", "angle", "ref3", "dihedral"):
                    value = getattr(r, key)
                    if value is not None:
                        row[key] = value
                rows.append(row)
            wire["rows"] = rows
        return wire
    if isinstance(node, Mechanism):
        from .engine import parse_text
        wire = {"type": "mechanism"}
        if node.steps:
            steps = []
            for s in node.steps:
                inner = parse_text(s.reaction)
                if len(inner.nodes) != 1 or not isinstance(inner.nodes[0], Reaction):
                    raise ParseError(
                        f"mechanism step body is not a reaction: {s.reaction!r}")
                step = {"label": s.label,
                        "reaction": to_wire(inner.nodes[0])}
                if s.flows:
                    step["flows"] = [{"source": f.source, "target": f.target}
                                     for f in s.flows]
                steps.append(step)
            wire["steps"] = steps
        if node.spectators:
            spectators = []
            for sp in node.spectators:
                inner = parse_text(sp)
                if len(inner.nodes) != 1 or not isinstance(inner.nodes[0], Molecule):
                    raise ParseError(
                        f"mechanism spectator is not a molecule: {sp!r}")
                spectators.append(to_wire(inner.nodes[0]))
            wire["spectators"] = spectators
        return wire
    if isinstance(node, Name):
        wire = {"type": "name", "content": node.content}
        return wire
    raise ParseError(f"no wire form for {type(node).__name__}")


_CORE_INGEST = {"formula", "atom", "molecule", "group", "bond", "reaction",
                "reaction-cascade", "electron-configuration", "embedded-math", "text",
                "mechanism", "crystal"}


def _number_text(value) -> str:
    """A wire number as model text: the reference's f64 Display shape -
    integral values drop the decimal (90 renders "90", 5.64 "5.64"),
    and scientific notation never appears (Python's repr switches to
    exponents below 1e-4; the reference spells them out)."""
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    text = repr(value)
    if "e" in text or "E" in text:
        text = f"{value:.20f}".rstrip("0").rstrip(".")
    return text


def from_model_json(data) -> Formula:
    if isinstance(data, str):
        data = json.loads(data)
    if data.get("type") != "formula":
        raise ParseError(f"expected a formula node, got {data.get('type')!r}")
    return _from_node(data)


# -- substance records (the resolver's wire form) -------------------------
#
# The resolved view of a substance (asciichem-model v1
# substance-record): every identifier and property carries its own
# provenance. Field names follow the schema exactly (camelCase); keys
# whose value is None are omitted (additionalProperties: false).

def substance_to_wire(substance) -> dict:
    """The canonical wire form (asciichem-model v1 substance-record)
    of a resolved Substance."""
    from .resolver.substance import Substance, Provenance
    if not isinstance(substance, Substance):
        raise TypeError(f"expected a resolver Substance, got {type(substance).__name__}")
    wire: dict = {"type": "substance-record"}
    if substance.preferred_name is not None:
        wire["preferredName"] = substance.preferred_name
    if substance.synonyms:
        wire["synonyms"] = list(substance.synonyms)
    if substance.formula is not None:
        wire["formula"] = substance.formula
    if substance.molecular_weight is not None:
        wire["molecularWeight"] = substance.molecular_weight
    identifiers_wire = []
    for i in substance.identifiers:
        entry = {"identifier": {"type": "identifier", "value": i.value,
                                "convention": i.convention}}
        provenance = _provenance_wire(i.provenance)
        if provenance is not None:
            entry["provenance"] = provenance
        identifiers_wire.append(entry)
    wire["identifiers"] = identifiers_wire
    if substance.properties:
        wire["properties"] = [
            _merge({"name": p.name, "value": p.value,
                    "units": p.units, "provenance": p.provenance},
                   _provenance_wire(p.provenance))
            for p in substance.properties]
    return wire


def _provenance_wire(provenance):
    if provenance is None:
        return None
    out = {"type": "provenance", "source": provenance.source}
    for wire_key, value in (("retrievedAt", provenance.retrieved_at),
                            ("sourceVersion", provenance.source_version),
                            ("attribution", provenance.attribution)):
        if value is not None:
            out[wire_key] = value
    return out


def _merge(base: dict, provenance: dict | None) -> dict:
    out = {k: v for k, v in base.items()
           if v is not None and k != "provenance"}
    if provenance is not None:
        out["provenance"] = provenance
    return out


def substance_from_wire(data) -> object:
    """Rebuilds a resolver Substance from the wire form (the cache
    path). Accepts a dict or a JSON string."""
    from .resolver.substance import (Identifier, Property, Provenance,
                                     Substance)
    if isinstance(data, str):
        data = json.loads(data)
    if data.get("type") != "substance-record":
        raise ParseError(
            f"expected a substance-record node, got {data.get('type')!r}")
    return Substance(
        preferred_name=data.get("preferredName"),
        synonyms=list(data.get("synonyms") or []),
        formula=data.get("formula"),
        molecular_weight=data.get("molecularWeight"),
        identifiers=[
            Identifier(
                value=pi["identifier"]["value"],
                convention=pi["identifier"]["convention"],
                provenance=_provenance_domain(pi.get("provenance")))
            for pi in data.get("identifiers", [])],
        properties=[
            Property(
                name=p["name"], value=p["value"],
                units=p.get("units"),
                provenance=_provenance_domain(p.get("provenance")))
            for p in data.get("properties", [])])


def _provenance_domain(wire: dict | None):
    if wire is None:
        return None
    from .resolver.substance import Provenance
    return Provenance(source=wire["source"],
                      retrieved_at=wire.get("retrievedAt"),
                      source_version=wire.get("sourceVersion"),
                      attribution=wire.get("attribution"))


def _from_node(wire) -> object:
    kind = wire.get("type")
    if kind not in _CORE_INGEST:
        raise ParseError(
            f'"{kind}" is not in the ingestible core set (emission covers it; '
            "round-trip acceptance lands with its corpus level)")
    if kind == "formula":
        return Formula([_from_node(n) for n in wire.get("nodes", [])])
    if kind == "atom":
        return Atom(element=wire["element"], isotope=wire.get("isotope"),
                    subscript=wire.get("subscript"),
                    superscript=wire.get("superscript"), charge=wire.get("charge"),
                    oxidation_state=wire.get("oxidationState"),
                    lone_pairs=wire.get("lonePairs"),
                    radical_electrons=wire.get("radicalElectrons"),
                    ring_closures=wire.get("ringClosures"),
                    aromatic=wire.get("aromatic"), hydrogens=wire.get("hydrogens"),
                    chirality=wire.get("chirality"))
    if kind == "bond":
        return Bond(wire.get("kind", "single"), direction=wire.get("direction"))
    if kind == "group":
        return Group([_from_node(n) for n in wire.get("nodes", [])],
                     multiplicity=wire.get("multiplicity"),
                     bracket=wire.get("bracket", "paren"))
    if kind == "molecule":
        molecule = Molecule([_from_node(n) for n in wire.get("nodes", [])],
                            coefficient=wire.get("coefficient"),
                            stereo=wire.get("stereo"))
        from .model import Identifier
        for i in wire.get("identifiers", []):
            molecule.identifiers.append(Identifier(i["value"], i["convention"]))
        return molecule
    if kind == "reaction":
        conditions = wire.get("conditions")
        return Reaction([_from_node(m) for m in wire["reactants"]],
                        [_from_node(m) for m in wire["products"]],
                        arrow=wire.get("arrow", "forward"),
                        conditions=ReactionConditions(
                            conditions.get("above"), conditions.get("below"))
                        if conditions else None)
    if kind == "reaction-cascade":
        return ReactionCascade([_from_node(s) for s in wire["steps"]])
    if kind == "electron-configuration":
        term = wire.get("termSymbol")
        return ElectronConfiguration(
            [(o["orbital"], o["occupancy"]) for o in wire["orbitals"]],
            (term.get("multiplicity"), term.get("letter"), term.get("jValue"))
            if term else None)
    if kind == "embedded-math":
        return EmbeddedMath(wire["source"])
    if kind == "text":
        return Text(wire["content"])
    if kind == "mechanism":
        # The model keeps a step's reaction as TEXT; the wire carries
        # the typed node, so ingestion renders it back through the
        # canonical text form.
        from .model import Mechanism, MechanismFlow, MechanismStep
        from .text import render

        steps = []
        for s in wire.get("steps", []):
            reaction = render(_from_node(s["reaction"]))
            flows = [MechanismFlow(f["source"], f["target"])
                     for f in s.get("flows", [])]
            steps.append(MechanismStep(s["label"], reaction, flows))
        spectators = [render(_from_node(sp))
                      for sp in wire.get("spectators", [])]
        return Mechanism(steps, spectators)
    if kind == "crystal":
        from .model import Crystal
        # The wire's numbers cast back to the model's authored-string
        # form (the reference's f64 Display: 90 -> "90", 5.64 -> "5.64").
        cells = {key: _number_text(wire[key])
                 for key in ("a", "b", "c", "alpha", "beta", "gamma")
                 if wire.get(key) is not None}
        return Crystal(name=wire.get("name"), spacegroup=wire.get("spacegroup"),
                       atoms=[_from_node(a) for a in wire.get("atoms", [])],
                       **cells)
    raise ParseError(f"unhandled wire type: {kind}")
