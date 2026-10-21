"""Mechanism electron-flow markers (TODO.v2/13 item 4, slice 4c):
a step value is its reaction followed by `| flow: SRC > DST` clauses.
Unknown clauses are rejected, never dropped."""
import pytest

from asciichem.model import Mechanism, ParseError
from asciichem.parser import parse_text
from asciichem.text import render
from asciichem.wire import to_wire


def mechanism(body: str) -> Mechanism:
    node = parse_text("mechanism{\n  " + body + "\n}").nodes[0]
    assert isinstance(node, Mechanism)
    return node


def test_parses_flow_clauses_into_typed_markers():
    step = mechanism("s1: A-B -> A + B | flow: A-B > A; B > B-C").steps[0]
    assert step.reaction == "A-B -> A + B"
    assert [(f.source, f.target) for f in step.flows] == [
        ("A-B", "A"), ("B", "B-C")]


def test_keeps_steps_without_flows_empty():
    assert mechanism("s1: A -> B").steps[0].flows == []


def test_round_trips_canonical_text_exactly():
    source = ("mechanism{\n  s1: A-B -> A + B | flow: A-B > A; B > B-C\n"
              "  s2: A + C -> A-C\n}")
    assert render(parse_text(source)) == source


def test_rejects_unknown_clause_kind():
    # The backtracking parser reports its generic error; the clause is
    # still rejected rather than dropped.
    with pytest.raises(ParseError):
        mechanism("s1: A -> B | curve: A > B")


def test_rejects_malformed_flow_arrow():
    with pytest.raises(ParseError):
        mechanism("s1: A -> B | flow: A >")


def test_emits_flows_on_the_wire():
    wire = to_wire(mechanism("s1: A -> B | flow: A > B"))
    assert wire["steps"][0]["flows"] == [{"source": "A", "target": "B"}]


def test_omits_flows_from_the_wire_when_absent():
    wire = to_wire(mechanism("s1: A -> B"))
    assert "flows" not in wire["steps"][0]


def test_round_trips_through_the_wire_form():
    from asciichem.text import render
    from asciichem.wire import from_model_json, to_model_json

    source = ("mechanism{\n"
              "  s1: A-B -> A + B | flow: A-B > A; B > B-C\n"
              "  s2: 2C ->[cat: Pd] D\n"
              "  spectator: Na^+\n}")
    parsed = from_model_json(to_model_json(parse_text(source)))
    assert render(parsed) == source
    mech = parsed.nodes[0]
    step = mech.steps[0]
    assert [(f.source, f.target) for f in step.flows] == [
        ("A-B", "A"), ("B", "B-C")]
    assert mech.spectators == ["Na^+"]
