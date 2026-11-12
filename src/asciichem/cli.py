"""The `asciichem` command-line interface: the offline command set the
reference Thor CLI (`asciichem-ruby/lib/asciichem/cli.rb`) ships, on
ONE dispatch so the contract - stdout shapes and exit codes - is
testable offline. `run()` is the test seam: the corpus runner invokes
it in-process (argv after the program name) and asserts on the
captured stdout/stderr/exit; `main()` is the console-script entry
point over the same dispatch.

Contract: mirror the Ruby CLI - parse errors exit 1, unknown `convert`
output formats 2, resolve errors 3, cite errors 4, identity errors 5;
`lint` exits 1 on error-severity findings, `roundtrip` 1 on a
mismatch, `validate` 1 on a known-but-invalid identifier; usage errors
exit 1 (the Thor posture). `tree` prints the command list in the
reference's Thor shape (`ascii_chem:cli` root, connector columns,
alphabetical, help and tree themselves excluded)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import __version__
from . import citation, cml, identifiers, linter, resolver, wire
from .model import (AsciiChemError, Formula, Molecule, ParseError,
                    Reaction, ReactionCascade)
from . import text as text_renderer
from .molfile import parse_molfile, write_molfile
from .smiles import parse_smiles, write_smiles

# The convert output formats (the help line's list - the reference's
# desc text, pinned by the corpus `tree` case).
CONVERT_FORMATS = ("mathml|text|html|latex|svg|structural-svg|"
                   "model-json|cml|smiles|molfile")

# The command registry - the single source for the dispatch and for
# `tree` (one source of truth for names and help texts). Order is the
# reference's Thor tree order (alphabetical; help and tree themselves
# excluded from the listing).
COMMANDS = [
    ("cite", "Resolve a substance and emit a dataset-type Relaton "
             "bibitem (XML)"),
    ("convert", f"Convert INPUT to FORMAT ({CONVERT_FORMATS})"),
    ("identity", "Derive InChI/InChIKey locally from the structure "
                 "(offline; requires an InChI engine)"),
    ("lint", "Run chemistry checks; exit 1 on error, 0 if clean"),
    ("parse-cml", "Parse CML XML and emit AsciiChem text"),
    ("resolve", "Resolve a substance from a source (network; cached)"),
    ("roundtrip", "Parse and re-emit; exit non-zero if not equal"),
    ("validate", "Offline identifier validation"),
    ("version", "Print the asciichem version"),
]


class _Fail(Exception):
    """A command failure: the stderr line (already prefixed -
    "Parse error: ", "Resolve error: ", ...) and the exit code the
    reference maps it to. `line` is empty when the command already
    wrote its own stderr block (roundtrip's mismatch report)."""

    def __init__(self, code: int, line: str = ""):
        super().__init__(line)
        self.code = code
        self.line = line


class FormatError(AsciiChemError):
    """An unknown (or unshipped) convert output format - the exit-2
    family, the reference's AsciiChem::FormatError."""


def _failure(code: int, prefix: str, message) -> _Fail:
    return _Fail(code, f"{prefix}{message}")


def _silent(code: int) -> _Fail:
    return _Fail(code)


# Ruby `puts` semantics: exactly one trailing newline, never two
# (the molfile writer's output already ends with one).
def _emit(out, text: str) -> None:
    out.write(text)
    if not text.endswith("\n"):
        out.write("\n")


# -- parser --------------------------------------------------------------

class _Parser(argparse.ArgumentParser):
    """Usage errors ride the Thor posture: exit 1 (the reference's
    Thor raises a usage error with exit status 1), the message on
    stderr."""

    def error(self, message: str):  # noqa: D401 - argparse contract
        raise _Fail(1, f"usage: {self.prog} {message}")


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="asciichem", add_help=True,
                     description="AsciiChem: ASCII syntax for chemistry "
                                 "parsed into a semantic model")
    # `--version`/`-v` are handled in `run` before argparse sees them
    # (the reference maps them to the version command; argparse's
    # version action would bypass the captured-stream seam).
    parser.add_argument("--version", "-v", action="store_true",
                        help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = False

    convert = subparsers.add_parser(
        "convert", help=COMMANDS[1][1],
        description="Convert INPUT to FORMAT")
    convert.add_argument("-i", "--input", help="Source text (or '-' for stdin)")
    convert.add_argument("-f", "--file", help="Read source from a file")
    convert.add_argument("--from", dest="from_grammar", default="asciichem",
                         help="Input grammar: asciichem|smiles|molfile")
    convert.add_argument("-t", "--format", default="mathml",
                         help="Output format")

    identity = subparsers.add_parser("identity", help=COMMANDS[2][1],
                                     description=COMMANDS[2][1])
    identity.add_argument("-i", "--input")
    identity.add_argument("-f", "--file")
    identity.add_argument("--from", dest="from_grammar", default="asciichem",
                          help="Input grammar: asciichem|smiles|molfile")
    identity.add_argument("--engine-bin", dest="engine_bin",
                          help="Path to the inchi-1 binary (not supported: "
                               "the Python engine derives in-process)")

    lint = subparsers.add_parser("lint", help=COMMANDS[3][1],
                                 description=COMMANDS[3][1])
    lint.add_argument("-i", "--input", required=True,
                      help="AsciiChem source text")
    lint.add_argument("-f", "--format", default="text",
                      help="Output format: text or json")

    parse_cml = subparsers.add_parser(
        "parse-cml", aliases=["parse_cml"], help=COMMANDS[4][1],
        description=COMMANDS[4][1])
    parse_cml.add_argument("-i", "--input", required=True,
                           help="CML XML source")

    resolve = subparsers.add_parser("resolve", help=COMMANDS[5][1],
                                    description=COMMANDS[5][1])
    _add_resolution_options(resolve)
    resolve.add_argument("-t", "--format", default="model-json",
                         help="Output: model-json | text | smiles")

    roundtrip = subparsers.add_parser("roundtrip", help=COMMANDS[6][1],
                                      description=COMMANDS[6][1])
    roundtrip.add_argument("-i", "--input", required=True)

    validate = subparsers.add_parser("validate", help=COMMANDS[7][1],
                                     description=COMMANDS[7][1])
    validate.add_argument("-i", "--input", required=True)

    subparsers.add_parser("version", help=COMMANDS[8][1],
                          description=COMMANDS[8][1])

    # Invocable but excluded from Thor's own tree listing - the same
    # posture the reference's built-in tree task has.
    subparsers.add_parser("tree", help="Print a tree of all available "
                                       "commands",
                          description="Print a tree of all available "
                                      "commands")

    cite = subparsers.add_parser("cite", help=COMMANDS[0][1],
                                 description=COMMANDS[0][1])
    _add_resolution_options(cite)
    return parser


def _add_resolution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cas", help="CAS registry number")
    parser.add_argument("--name", help="Substance name")
    parser.add_argument("--cid", help="PubChem CID")
    parser.add_argument("--inchikey", help="InChIKey")
    parser.add_argument("--smiles", help="SMILES")
    parser.add_argument("--source", default="pubchem",
                        help="Resolver source")
    parser.add_argument("--refresh", action="store_true",
                        help="Bypass the cache")


# -- dispatch ------------------------------------------------------------

def run(arguments: list[str], out=None, err=None) -> int:
    """Runs the CLI. The returned int is the process exit code. The
    argv is the list AFTER the program name (the corpus runner's
    invocation shape); `out`/`err` default to the process streams."""
    if out is None:
        out = sys.stdout
    if err is None:
        err = sys.stderr
    # The reference's `map %w[--version -v] => :version`: the flags
    # print the version line and exit 0 (stdout, not stderr).
    arguments = list(arguments)
    if "--version" in arguments or "-v" in arguments:
        _emit(out, f"asciichem {__version__}")
        return 0
    parser = _build_parser()
    try:
        args = parser.parse_args(list(arguments))
    except _Fail as failed:
        if failed.line:
            err.write(failed.line + "\n")
        return failed.code
    if args.command is None:
        # Bare `asciichem`: the command list, exit 0.
        parser.print_help(out)
        return 0
    # argparse reports the alias actually used (parse_cml) - normalize
    # to the canonical dispatch name.
    command = _ALIASES.get(args.command, args.command)
    try:
        handler = _HANDLERS[command]
        handler(args, out, err)
        return 0
    except _Fail as failed:
        if failed.line:
            err.write(failed.line + "\n")
        return failed.code


# -- source ingestion ----------------------------------------------------

def _source_text(input_: Optional[str], file: Optional[str]) -> str:
    """The source text: -f wins over -i, `-` reads stdin (the
    reference `read_source` order)."""
    if file is not None:
        with open(file, encoding="utf-8") as handle:
            return handle.read()
    if input_ is None:
        raise _Fail(1, "Parse error: provide -i INPUT or -f FILE")
    if input_ == "-":
        return sys.stdin.read()
    return input_


def _ingest(source: str, from_grammar: str) -> Formula:
    """One ingestion point per input grammar (TODO.v2 09): every
    grammar funnels into the same model, so every output format works
    regardless of the input language. Molfile input may be a path."""
    import os

    if from_grammar == "asciichem":
        from .engine import parse_text
        return parse_text(source)
    if from_grammar == "smiles":
        return parse_smiles(source)
    if from_grammar == "molfile":
        text = (open(source, encoding="utf-8").read()
                if os.path.isfile(source) else source)
        return Formula(nodes=[parse_molfile(text)])
    raise ParseError(f"unknown --from grammar: {from_grammar}")


def _molecules_in(formula: Formula) -> list:
    """Molecules anywhere at the top level: direct formula nodes plus
    reaction terms. Identity derivation needs exactly one molecule."""
    molecules: list[Molecule] = []
    for node in formula.nodes:
        if isinstance(node, Reaction):
            candidates = list(node.reactants) + list(node.products)
        elif isinstance(node, ReactionCascade):
            candidates = [m for step in node.steps
                          for m in list(step.reactants) + list(step.products)]
        else:
            candidates = [node]
        molecules.extend(m for m in candidates if isinstance(m, Molecule))
    return molecules


# -- convert -------------------------------------------------------------

def _cmd_convert(args, out, err) -> None:
    source = _source_text(args.input, args.file)
    try:
        formula = _ingest(source, args.from_grammar)
        rendered = _render(formula, args.format)
    except FormatError as exc:
        raise _failure(2, "Format error: ", exc) from exc
    except ParseError as exc:
        raise _failure(1, "Parse error: ", exc) from exc
    except AsciiChemError as exc:
        # Emit-side rejections (an unbonded formula's SMILES, a wire
        # construct outside the subset) ride the parse family - the
        # reference's convert maps every non-format failure to exit 1.
        raise _failure(1, "Parse error: ", exc) from exc
    _emit(out, rendered)


def _render(formula: Formula, format: str) -> str:
    from . import html, latex, mathml, svg

    if format == "mathml":
        return mathml.render_mathml(formula)
    if format == "text":
        return text_renderer.render(formula)
    if format == "svg":
        return svg.render_svg(formula)
    if format == "structural-svg":
        return svg.render_structural_svg(formula)
    if format == "model-json":
        return wire.to_model_json(formula)
    if format == "cml":
        return cml.to_cml(formula)
    if format == "smiles":
        return write_smiles(formula)
    if format == "molfile":
        for node in formula.nodes:
            if isinstance(node, Molecule):
                return write_molfile(node, "")
        raise ParseError("molfile output needs a molecule")
    if format == "html":
        return html.render(formula)
    if format == "latex":
        return latex.render(formula)
    # An unknown or unshipped formatter is the format family (exit 2),
    # the reference's `unknown formatter` shape.
    raise FormatError(f"unknown formatter :{format}")


def _cmd_parse_cml(args, out, err) -> None:
    try:
        formula = cml.parse_cml(args.input)
    except AsciiChemError as exc:
        raise _failure(1, "CML parse error: ", exc) from exc
    _emit(out, text_renderer.render(formula))


def _cmd_roundtrip(args, out, err) -> None:
    from .engine import parse_text
    try:
        rendered = text_renderer.render(parse_text(args.input))
    except ParseError as exc:
        raise _failure(1, "Parse error: ", exc) from exc
    if rendered == args.input:
        _emit(out, rendered)
        return
    err.write(f"round-trip mismatch:\n"
              f"  input:    {args.input!r}\n"
              f"  rendered: {rendered!r}\n")
    raise _silent(1)


# -- lint ----------------------------------------------------------------

def _cmd_lint(args, out, err) -> None:
    from .engine import parse_text
    try:
        formula = parse_text(args.input)
    except ParseError as exc:
        raise _failure(1, "Parse error: ", exc) from exc
    diagnostics = linter.run(formula)
    if args.format == "json":
        payload = [{"severity": d.severity, "message": d.message,
                    "node": d.label} for d in diagnostics]
        _emit(out, json.dumps(payload, indent=2))
    else:
        for diagnostic in diagnostics:
            _emit(out, str(diagnostic))
    if any(d.severity == "error" for d in diagnostics):
        raise _silent(1)


# -- validate ------------------------------------------------------------

def _cmd_validate(args, out, err) -> None:
    from .engine import parse_text
    try:
        formula = parse_text(args.input)
    except ParseError as exc:
        raise _failure(1, "Parse error: ", exc) from exc
    # Top-level molecules only - annotations live on the formula's
    # molecule nodes (the reference's grep(Molecule)).
    annotations = [i for node in formula.nodes
                   if isinstance(node, Molecule)
                   for i in node.identifiers]
    if not annotations:
        _emit(out, "no identifier annotations found")
        return
    invalid = False
    for identifier in annotations:
        known = identifiers.known(identifier.convention)
        valid = known and identifiers.valid(identifier.convention,
                                            identifier.value)
        status = ("ok" if valid
                  else "INVALID" if known else "unknown convention")
        _emit(out, f"{identifier.convention:<12} {identifier.value:<40} "
                   f"{status}")
        if known and not valid:
            invalid = True
    if invalid:
        raise _silent(1)


# -- identity ------------------------------------------------------------

def _cmd_identity(args, out, err) -> None:
    from . import inchi

    if args.engine_bin is not None:
        raise _failure(
            5, "Identity error: ",
            f"the Python engine derives identities in-process through "
            f"rdkit; --engine-bin ({args.engine_bin}) is not supported")
    # The reference parses whatever read_source gives - no input means
    # the empty string, whose parse failure is an identity error (5).
    try:
        source = _source_text(args.input, args.file)
    except _Fail as failed:
        if "provide -i" in failed.line:
            source = ""
        else:
            raise
    try:
        formula = _ingest(source, args.from_grammar)
        molecules = _molecules_in(formula)
        if len(molecules) != 1:
            raise AsciiChemError(
                f"identity derivation needs exactly one molecule, "
                f"got {len(molecules)}")
        derived = inchi.identity_for(molecules[0])
    except AsciiChemError as exc:
        raise _failure(5, "Identity error: ", exc) from exc
    _emit(out, derived.inchi)
    if derived.inchikey:
        _emit(out, derived.inchikey)


# -- resolve / cite ------------------------------------------------------

def _selected_identifier(args) -> tuple[str, str]:
    """The selected identifier: the first set of
    --cas/--name/--cid/--inchikey/--smiles wins (the reference's
    filter_map order); `--cid` answers under the `pubchem-cid`
    convention."""
    for flag, convention in (("--cas", "cas"), ("--name", "name"),
                             ("--cid", "pubchem-cid"),
                             ("--inchikey", "inchikey"),
                             ("--smiles", "smiles")):
        value = getattr(args, flag[2:])
        if value is not None:
            return convention, value
    raise ValueError("no identifier flag given")


def _cmd_resolve(args, out, err) -> None:
    prefix = "Resolve error: "
    try:
        convention, value = _selected_identifier(args)
    except ValueError:
        raise _failure(3, prefix,
                       "give one of --cas/--name/--cid/--inchikey/--smiles")
    substance = _resolve_substance(args, value, convention, prefix)
    if args.format == "text":
        text = substance.preferred_name or ""
    elif args.format == "smiles":
        text = substance.identifier_value("canonical-smiles") or ""
    else:
        # The reference's else-branch: anything but text/smiles is the
        # model JSON.
        text = json.dumps(wire.substance_to_wire(substance))
    _emit(out, text)


def _cmd_cite(args, out, err) -> None:
    prefix = "Cite error: "
    try:
        convention, value = _selected_identifier(args)
    except ValueError:
        raise _failure(4, prefix,
                       "give one of --cas/--name/--cid/--inchikey/--smiles")
    substance = _resolve_substance(args, value, convention, prefix)
    try:
        xml = citation.to_xml(substance)
    except AsciiChemError as exc:
        raise _failure(4, prefix, exc) from exc
    _emit(out, xml)


def _resolve_substance(args, value: str, convention: str, prefix: str):
    """One resolution path for resolve/cite: the unknown-source lookup
    errors immediately (the reference's `Resolver[source]` raises
    before any request); an unanswered lookup is "does not know"."""
    if args.source not in resolver.adapters:
        known = ", ".join(sorted(resolver.adapters))
        raise _failure(
            3 if prefix.startswith("Resolve") else 4, prefix,
            f"unknown resolver source {args.source!r} (registered: {known})")
    substance = resolver.resolve(value=value, convention=convention,
                                 sources=[args.source],
                                 refresh=args.refresh)
    if substance is None:
        code = 3 if prefix.startswith("Resolve") else 4
        raise _failure(code, prefix,
                       f"{args.source} does not know {value!r}")
    return substance


# -- tree / version ------------------------------------------------------

def _cmd_version(args, out, err) -> None:
    _emit(out, f"asciichem {__version__}")


def _cmd_tree(args, out, err) -> None:
    # Thor's built-in `tree` task, mirrored: the root label and the
    # connector shape are the reference's; the entries are this
    # dispatch's own help texts (help and tree excluded, as Thor's
    # tree lists). The COMMANDS registry is the one source of truth.
    lines = ["ascii_chem:cli"]
    last = len(COMMANDS) - 1
    for index, (name, help_text) in enumerate(COMMANDS):
        connector = "└─" if index == last else "├─"
        lines.append(f"  {connector} {name.replace('-', '_')} ({help_text})")
    _emit(out, "\n".join(lines))


_HANDLERS = {
    "cite": _cmd_cite,
    "convert": _cmd_convert,
    "identity": _cmd_identity,
    "lint": _cmd_lint,
    "parse-cml": _cmd_parse_cml,
    "resolve": _cmd_resolve,
    "roundtrip": _cmd_roundtrip,
    "validate": _cmd_validate,
    "version": _cmd_version,
    "tree": _cmd_tree,
}

# argparse reports the alias actually used; normalize before dispatch.
_ALIASES = {"parse_cml": "parse-cml"}


def main(argv: Optional[list[str]] = None) -> int:
    """The console-script entry point: run the dispatch over the
    process argv and exit with its code."""
    code = run(sys.argv[1:] if argv is None else list(argv))
    raise SystemExit(code)
