"""The mechanism step graph (TODO.v2/13 item 4; the port of the Ruby
reference's mechanism_graph_spec): derived from the verbatim
``mechanism{}`` entries - steps, catalysts (species on both sides of a
step), and links (a product species reused as a reactant by a later
step). Computed; the text and wire forms are unchanged."""
from asciichem import parse_text
from asciichem.mechanism_graph import for_mechanism
from asciichem.model import Mechanism, Reaction
from asciichem.text import render


def graph_for(*lines):
    source = "mechanism{\n  " + "\n  ".join(lines) + "\n}"
    parsed = parse_text(source).nodes[0]
    assert isinstance(parsed, Mechanism)
    return for_mechanism(parsed)


def test_parses_each_step_value_into_a_reaction():
    graph = graph_for("step1: A + B -> C")
    assert len(graph.steps) == 1
    assert graph.steps[0].label == "step1"
    assert isinstance(graph.steps[0].reaction, Reaction)


def test_links_a_later_step_that_consumes_an_earlier_product():
    graph = graph_for("s1: A + B -> C", "s2: C -> D")
    assert [(l.from_, l.to, l.species) for l in graph.links] == [
        ("s1", "s2", ["C"])]


def test_links_chains_step_to_step_not_across_gaps():
    graph = graph_for("s1: A -> B", "s2: B -> C", "s3: C -> D")
    assert [(l.from_, l.to, l.species) for l in graph.links] == [
        ("s1", "s2", ["B"]), ("s2", "s3", ["C"])]


def test_does_not_link_steps_with_no_shared_species():
    graph = graph_for("s1: A -> B", "s2: X -> Y")
    assert graph.links == []


def test_recognises_a_species_on_both_sides_of_one_step_as_a_catalyst():
    graph = graph_for("s1: A + Pd -> B + Pd")
    assert graph.steps[0].catalysts == ["Pd"]


def test_links_through_a_shared_catalyst_species_as_well():
    graph = graph_for("s1: A + Pd -> B + Pd", "s2: Pd + C -> D")
    assert [l.species for l in graph.links] == [["Pd"]]


def test_keeps_prose_entries_as_none_reactions_with_no_links():
    graph = graph_for("s1: proton transfer", "s2: A -> B")
    assert graph.steps[0].reaction is None
    assert isinstance(graph.steps[1].reaction, Reaction)
    assert graph.links == []


def test_uses_a_cascade_entry_side_and_exit_side_for_linking():
    graph = graph_for("s1: A -> B -> C", "s2: C -> D")
    assert [(l.from_, l.to, l.species) for l in graph.links] == [
        ("s1", "s2", ["C"])]


def test_ignores_coefficients_when_matching_species():
    graph = graph_for("s1: 2A -> 3B", "s2: 2B -> C")
    assert [(l.from_, l.to, l.species) for l in graph.links] == [
        ("s1", "s2", ["B"])]


def test_carries_flows_into_the_step_graph():
    graph = graph_for("s1: A -> B | flow: A > B")
    assert [f.source for f in graph.steps[0].flows] == ["A"]


def test_leaves_the_canonical_text_and_model_unchanged():
    source = "mechanism{\n  s1: A -> B\n  s2: B -> C\n}"
    assert render(parse_text(source)) == source
