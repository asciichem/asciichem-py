# Vendored PARG artifact

`asciichem.json` is the compiled PARG grammar of the AsciiChem syntax,
vendored from the contract repository
[asciichem-grammar](https://github.com/asciichem/asciichem-grammar) — the same
artifact `asciichem-rs` embeds (`artifacts/asciichem.json`) and the Ruby
reference vendors (`lib/asciichem/engine/parg/asciichem.json`).

- Grammar version pinned by `engine.py`: **0.4.0** (`PARG_GRAMMAR_VERSION`)
- Envelope checksum (verified by `parsanol` on every load, over the
  canonicalized envelope): `sha256:2891ce4d…`
- File sha256 at vendoring time:
  `b5c1f5289f0487e974091e451f5b2703c12c861347c09dd8332bdcfd1822c34f`

PARG artifacts are checksummed; `parsanol` refuses a mismatched envelope,
so a stale copy fails loudly rather than parsing a different language.
