# asciichem (Python)

Python implementation of [AsciiChem](https://asciichem.org) — parse
ASCII chemistry into a **semantic model** (atoms, bonds, isotopes,
charges, reactions, electron configurations), then render canonical
text or the canonical wire JSON. The defining fix over AsciiMath:
`^14C` binds the isotope to the atom, never to a phantom `{}` carrier.

Strategy A of the ecosystem's [language-binding
ADR](https://github.com/asciichem/asciichem-ts/blob/main/docs/adr/0001-language-bindings.adoc):
a native implementation conforming to the shared contracts —
`asciichem-model` (v1 wire form) and `asciichem-tests` (the corpus
every implementation passes). This package, like the TypeScript one,
must agree with the reference on every corpus case.

Parsing runs on the **parsanol family** (TODO.v2/17 slice 9): the
grammar is NOT reimplemented here — `asciichem.engine` consumes the
checksummed AsciiChem PARG artifact (vendored at
`src/asciichem/parg/asciichem.json`, the same artifact the Rust core
embeds and the Ruby reference vendors) through
[`parsanol`](https://pypi.org/project/parsanol/), the one runtime
dependency (its per-platform wheels vendor the native engine).

## Usage

```python
from asciichem import parse_text
from asciichem.text import render
from asciichem.wire import to_model_json, from_model_json

from asciichem.smiles import parse_smiles, write_smiles

formula = parse_text("2H_2 + O_2 ->[heat] 2H_2O")
aspirin = parse_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # same model, same renderers
write_smiles(aspirin)  # deterministic writer; round-trips exactly
render(formula)          # canonical AsciiChem text (round-trip contract)
to_model_json(formula)   # asciichem-model v1 wire JSON
back = from_model_json(to_model_json(formula))
```

## Conformance claim

Verified in CI against the [asciichem-tests](https://github.com/asciichem/asciichem-tests)
corpus and [asciichem-model](https://github.com/asciichem/asciichem-model)
schemas (both cloned as siblings):

| Level | Claim |
|---|---|
| parse/reject | 100% — every corpus case parses or raises `ParseError` exactly as marked |
| L0 | 100% — emission validates against the v1 JSON Schemas |
| L1 | 100% — `render(parse(s)) == s` for every `roundTrip` case |
| L3 | 100% — `@`/`@@` chirality, directional bonds; every stereo fixture |
| SMILES | 100% — ingestion + deterministic emission, every `structure/smiles/*` fixture (asciichem-tests v0.8.0) |
| molfile | 100% — V2000 ingestion + emission, every `structure/molfile/*` fixture |
| L2 (MathML goldens) | 100% — byte-identical against the corpus goldens |
| Resolver / InChI | not claimed — `asciichem.inchi` provides an engine seam only; no bundled engine |

### Conformance levels

Mapped onto the ecosystem's formal levels — see
[the levels document](https://www.asciichem.org/guides/conformance-levels/):

| Level | Claim |
|---|---|
| L0 notation | **yes** — parse/reject and canonical Text round-trip at 100% |
| L1 semantics | **yes** — wire emission validates against the v1 JSON Schemas (corpus `L0` gate at 100%) |
| L2 render | **yes** — canonical text, wire, MathML goldens, SVG byte-exact with the TypeScript implementation |
| L3 stereo | **yes** — `@`/`@@` chirality, directional bonds |
| L4 identity | not claimed — engine seam only; no bundled engine |

## Development

```sh
pip install -e ".[dev]"
python -m pytest           # unit + corpus conformance (siblings cloned)
```

## License

BSD-2-Clause — same as the rest of the AsciiChem ecosystem.
