"""The depiction-hint override (asciichem.github.io#125 step 3): the
authored x2/y2 coordinates (the CML round-trip, molfile-carried) take
precedence over every computed layout — the ruby reference's
pre-positioned rule (layout.rb all_positioned?)."""
from asciichem.layout import layout_molecule
from asciichem.model import Atom, Bond, Molecule


def _authored_molecule():
    return Molecule(
        nodes=[Atom("C", x2=10.0, y2=20.0), Bond(), Atom("O", x2=66.0, y2=13.0)]
    )


def test_authored_coordinates_override_the_computed_layout():
    graph, positions = layout_molecule(_authored_molecule())
    assert [(p.x, p.y) for p in positions] == [(10.0, 20.0), (66.0, 13.0)]
    assert [a.element for a in graph.atoms] == ["C", "O"]


def test_partial_authoring_layouts_normally():
    molecule = Molecule(
        nodes=[Atom("C", x2=10.0, y2=20.0), Bond(), Atom("O")]
    )
    _, positions = layout_molecule(molecule)
    # The computed layout ignores the partial authoring: the first
    # atom lands where the placer puts it, not at (10, 20).
    assert (positions[0].x, positions[0].y) != (10.0, 20.0)
