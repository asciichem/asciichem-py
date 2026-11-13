"""The mechanism step graph (TODO.v2/13 item 4; the port of the Ruby
reference's ``AsciiChem::MechanismGraph``). ``mechanism{}`` entries
stay verbatim (label + reaction text) in the model; this service
derives the graph the text implies - each step's reaction parsed with
the same grammar, species that appear on both sides of a step
recognised as catalysts, and steps linked where a product species of
an earlier step is a reactant of a later one.

Everything is computed; nothing mutates. A step whose value is not a
reaction (prose, "proton transfer") keeps a ``None`` reaction and takes
part in no links - never an error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .model import Mechanism, ParseError, Reaction, ReactionCascade
from .text import render as render_text
from .engine import parse_text


@dataclass
class GraphStep:
    label: str
    # The parsed reaction (or cascade); None when the step's value is
    # not a reaction - prose takes part in no links.
    reaction: Reaction | ReactionCascade | None
    # Coefficient-free species names on both sides of the step.
    catalysts: list = field(default_factory=list)
    flows: list = field(default_factory=list)


@dataclass
class GraphLink:
    from_: str
    to: str
    # The shared species names (coefficient-free, in the earlier
    # step's product order).
    species: list


@dataclass
class Graph:
    steps: list
    links: list


def for_mechanism(mechanism: Mechanism) -> Graph:
    """Derives the step graph of a parsed ``Mechanism`` node."""
    steps = [graph_step(step) for step in mechanism.steps]
    return Graph(steps=steps, links=links(steps))


def graph_step(step) -> GraphStep:
    reaction = parse_reaction(step.reaction)
    sides = sides_of(reaction) if reaction is not None else None
    catalysts = ([name for name in sides["reactants"]
                  if name in sides["products"]] if sides else [])
    return GraphStep(label=step.label, reaction=reaction, catalysts=catalysts,
                     flows=list(step.flows))


def links(steps: list[GraphStep]) -> list[GraphLink]:
    """Links each step to every later step sharing a species: the
    earlier step's product names against the later step's reactant
    names."""
    out: list[GraphLink] = []
    for index, source in enumerate(steps):
        if source.reaction is None:
            continue
        from_products = sides_of(source.reaction)["products"]
        for target in steps[index + 1:]:
            if target.reaction is None:
                continue
            shared = [name for name in from_products
                      if name in sides_of(target.reaction)["reactants"]]
            if shared:
                out.append(GraphLink(from_=source.label, to=target.label,
                                     species=shared))
    return out


def parse_reaction(text: str):
    """The step's value as a Reaction/ReactionCascade, or None - prose
    and non-reaction nodes keep the graph quiet, never an error."""
    try:
        node = parse_text(text).nodes[0]
    except ParseError:
        return None
    if isinstance(node, (Reaction, ReactionCascade)):
        return node
    return None


def sides_of(reaction) -> dict:
    """Coefficient-free species names per side, in side order; a
    cascade's entry side is its head and its exit side is its last
    step (the reference's ``sides_of``)."""
    if isinstance(reaction, ReactionCascade):
        steps = reaction.steps
        if not steps:
            return {"reactants": {}, "products": {}}
        return {"reactants": side_species(steps[0].reactants),
                "products": side_species(steps[-1].products)}
    return {"reactants": side_species(reaction.reactants),
            "products": side_species(reaction.products)}


def side_species(molecules) -> dict:
    """Insertion-ordered species name -> total coefficient. The NAME is
    the molecule's canonical text with its leading integer stripped -
    the matching key, never the wire form."""
    out: dict[str, int] = {}
    for molecule in molecules:
        name = re.sub(r"^\d+", "", render_text(molecule))
        out[name] = out.get(name, 0) + molecule.stoichiometric_coefficient()
    return out
