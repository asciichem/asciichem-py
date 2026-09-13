"""Recursive-descent parser for the AsciiChem grammar - a rule-by-rule
port of the reference parslet grammar (ordered choice preserved):
coefficient lookahead, the hydrogen bare-subscript special case, the
restricted stereo set, cascade-before-reaction, keyword constructs
before molecules. The AtomBuilder charge/oxidation disambiguation is
ported verbatim (sign-first charges canonicalise to number-then-sign)."""
from __future__ import annotations

import re
from typing import Optional

from .model import (Atom, Bond, Calculation, CalculationProperty, Crystal,
                    ElectronConfiguration, EmbeddedMath, Formula, Group,
                    Identifier, Mechanism, MechanismStep, Molecule, Name,
                    ParseError, Reaction, ReactionCascade, ReactionConditions,
                    Spectrum, SpectrumPeak, Text, ZMatrix, ZRow)

WS = " \t\r\n\f\v"
ARROWS = [("<=>", "equilibrium"), ("<->", "resonance"), ("->", "forward"), ("<-", "reverse")]
BOND_KINDS = [("##", "quadruple"), (">-", "wedge"), ("-<", "hash"),
              ("~>", "dative"), ("~~", "wavy"), ("#", "triple"),
              ("=", "double"), ("-", "single")]
STEREO = {"alpha": "alpha", "beta": "beta", "R": "R", "S": "S", "E": "E", "Z": "Z",
          "α": "alpha", "β": "beta", "a": "alpha", "b": "beta"}
IDENTIFIER_TYPES = {"inchi", "smiles", "cas", "iupac", "cid", "chebi"}


class Parser:
    def __init__(self, source: str):
        self.s = source
        self.pos = 0

    # -- entry ------------------------------------------------------------

    def parse(self) -> Formula:
        self.skip_ws()
        nodes = [self.node()]
        while True:
            save = self.pos
            self.skip_ws()
            try:
                nodes.append(self.node())
            except ParseError:
                self.pos = save
                break
        self.skip_ws()
        if self.pos != len(self.s):
            self.fail("valid AsciiChem syntax")
        return Formula(nodes)

    def node(self):
        for rule in (self.reaction_cascade, self.reaction, self.electron_config,
                     self.crystal, self.spectrum, self.calculation, self.zmatrix,
                     self.mechanism, self.annotated_molecule, self.molecule,
                     self.embedded_math, self.text_run):
            result = self.try_rule(rule)
            if result is not None:
                return result
        self.fail("valid AsciiChem syntax")

    def try_rule(self, rule):
        save = self.pos
        try:
            return rule()
        except ParseError:
            self.pos = save
            return None

    # -- reactions --------------------------------------------------------

    def reaction_cascade(self):
        first = self.reaction()
        tail = []
        while True:
            save = self.pos
            arrow = self.try_rule(self.arrow)
            if arrow is None:
                self.pos = save
                break
            self.skip_ws()
            try:
                tail.append((arrow, self.terms()))
            except ParseError:
                self.pos = save
                break
        if not tail:
            raise ParseError("cascade needs a second step")
        steps = [first]
        for arrow, products in tail:
            steps.append(Reaction(steps[-1].products, products,
                                  arrow=ARROW_TOKENS[arrow[0]],
                                  conditions=arrow[1]))
        return ReactionCascade(steps)

    def reaction(self):
        reactants = self.terms()
        arrow = self.arrow()
        self.skip_ws()
        products = self.terms()
        return Reaction(reactants, products, arrow=ARROW_TOKENS[arrow[0]],
                        conditions=arrow[1])

    def terms(self):
        molecules = [self.molecule()]
        while True:
            save = self.pos
            self.skip_ws()
            if not self.eat("+"):
                self.pos = save
                break
            self.skip_ws()
            try:
                molecules.append(self.molecule())
            except ParseError:
                self.pos = save
                break
        return molecules

    def arrow(self):
        self.skip_ws()
        kind = None
        for token, _ in ARROWS:
            if self.s.startswith(token, self.pos):
                kind = token
                self.pos += len(token)
                break
        if kind is None:
            self.fail("reaction arrow")
        above = self.try_rule(self.condition)
        below = self.try_rule(self.condition)
        conditions = None
        if above or below:
            conditions = ReactionConditions(above, below)
        return (kind, conditions)

    def condition(self):
        if not self.eat("["):
            self.fail("[")
        end = self.s.find("]", self.pos)
        if end == -1:
            self.fail("]")
        text = self.s[self.pos:end]
        self.pos = end + 1
        return text

    # -- molecules ----------------------------------------------------------

    def annotated_molecule(self):
        molecule = self.molecule()
        annotations = [self.molecule_annotation()]
        while True:
            save = self.pos
            ann = self.try_rule(self.molecule_annotation)
            if ann is None:
                self.pos = save
                break
            annotations.append(ann)
        for ann in annotations:
            apply_annotation(molecule, ann)
        return molecule

    def molecule_annotation(self):
        self.skip_ws()
        if self.s.startswith('@meta("', self.pos):
            self.pos += len('@meta("')
            key = self.until_quote()
            self.expect('","')
            value = self.until_quote()
            self.expect('")')
            return ("meta", key, value)
        if self.eat("@"):
            ann_type = self.word()
            if ann_type is None:
                self.fail("annotation type")
            self.expect('("')
            value = self.until_quote()
            self.expect('")')
            return ("ann", ann_type, value)
        self.fail("@annotation")

    def molecule(self):
        stereo = self.try_rule(self.stereo_prefix)
        coefficient = self.try_rule(self.coefficient)
        units = [self.unit_or_bond()]
        while True:
            unit = self.try_rule(self.unit_or_bond)
            if unit is None:
                break
            units.append(unit)
        return Molecule(nodes=units, coefficient=coefficient, stereo=stereo)

    def unit_or_bond(self):
        for rule in (self.unit, self.bond_token):
            save = self.pos
            try:
                return rule()
            except ParseError:
                self.pos = save
        self.fail("atom, group, or bond")

    def unit(self):
        for rule in (self.prefixed_atom, self.hydrogen_atom, self.group):
            save = self.pos
            try:
                return rule()
            except ParseError:
                self.pos = save
        return self.plain_atom()

    def stereo_prefix(self):
        if not self.eat("("):
            self.fail("(")
        letter = None
        for token in ("alpha", "beta", "R", "S", "E", "Z", "α", "β", "a", "b"):
            if self.s.startswith(token, self.pos):
                letter = token
                self.pos += len(token)
                break
        if letter is None:
            self.fail("stereo letter")
        self.expect(")")
        self.expect("-")
        return STEREO[letter]

    def coefficient(self):
        save = self.pos
        digits = self.digits()
        if digits is None:
            self.fail("digits")
        nxt = self.s[self.pos:self.pos + 2]
        if re.match(r"[A-Z][a-z]?", nxt) or (self.pos < len(self.s) and self.s[self.pos] in "([{"):
            return digits
        self.pos = save
        self.fail("coefficient lookahead")

    def prefixed_atom(self):
        lone_pairs = self.try_rule(self.lewis_prefix)
        isotope = self.try_rule(self.isotope_marker)
        if isotope is None:
            self.fail("isotope marker")
        element = self.element_symbol()
        subscript = self.try_rule(self.subscript_marker)
        superscript = self.try_rule(self.superscript_marker)
        radical_electrons = self.try_rule(self.lewis_radicals)
        ring_closures = self.try_rule(self.ring_closures)
        annotations = self.atom_annotations()
        return build_atom(element, isotope, subscript, superscript,
                          lone_pairs, radical_electrons, ring_closures, annotations)

    def isotope_marker(self):
        save = self.pos
        if self.pos < len(self.s) and self.s[self.pos] in "^_":
            marker = self.s[self.pos]
            self.pos += 1
            digits = self.digits()
            if digits is None:
                self.pos = save
                self.fail("isotope digits")
            return marker + digits
        self.fail("isotope marker")

    def hydrogen_atom(self):
        lone_pairs = self.try_rule(self.lewis_prefix)
        if not self.eat("H"):
            self.fail("H")
        if self.pos < len(self.s) and self.s[self.pos].islower():
            self.fail("two-letter element")
        subscript = self.try_rule(self.h_subscript)
        superscript = self.try_rule(self.superscript_marker)
        radical_electrons = self.try_rule(self.lewis_radicals)
        annotations = self.atom_annotations()
        return build_atom("H", None, subscript, superscript,
                          lone_pairs, radical_electrons, None, annotations)

    def h_subscript(self):
        if self.eat("_"):
            return "_" + self.subscript_value()
        return self.digits()

    def plain_atom(self):
        lone_pairs = self.try_rule(self.lewis_prefix)
        element = self.element_symbol()
        subscript = self.try_rule(self.subscript_marker)
        superscript = self.try_rule(self.superscript_marker)
        radical_electrons = self.try_rule(self.lewis_radicals)
        ring_closures = self.try_rule(self.ring_closures)
        annotations = self.atom_annotations()
        return build_atom(element, None, subscript, superscript,
                          lone_pairs, radical_electrons, ring_closures, annotations)

    def bond_token(self):
        for token, kind in BOND_KINDS:
            if self.s.startswith(token, self.pos):
                self.pos += len(token)
                return Bond(kind)
        self.fail("bond")

    def group(self):
        if self.pos >= len(self.s) or self.s[self.pos] not in "([{":
            self.fail("opening bracket")
        open_char = self.s[self.pos]
        self.pos += 1
        nodes = [self.group_node()]
        while True:
            node = self.try_rule(self.group_node)
            if node is None:
                break
            nodes.append(node)
        if self.pos >= len(self.s) or self.s[self.pos] not in ")]}":
            self.fail("closing bracket")
        self.pos += 1
        multiplicity = self.try_rule(self.multiplicity)
        bracket = {"(": "paren", "[": "square", "{": "brace"}[open_char]
        return Group(nodes=nodes, multiplicity=multiplicity, bracket=bracket)

    def group_node(self):
        for rule in (self.reaction, self.electron_config, self.molecule,
                     self.embedded_math, self.group_text_run):
            result = self.try_rule(rule)
            if result is not None:
                return result
        self.fail("group node")

    def multiplicity(self):
        if not self.eat("_"):
            self.fail("_")
        return self.digits()

    # -- atoms -----------------------------------------------------------

    def atom_annotations(self):
        out = {}
        for rule, key in ((self.coordinate_annotation, "coord"),
                          (self.parity_annotation, "parity"),
                          (self.multiplicity_annotation, "spin"),
                          (self.title_annotation, "title"),
                          (self.fractional_annotation, "fract")):
            result = self.try_rule(rule)
            if result is not None:
                out[key] = result
        return out

    def coordinate_annotation(self):
        if not self.eat("@("):
            self.fail("@(")
        x = self.float_number()
        self.expect(",")
        y = self.float_number()
        z = None
        save = self.pos
        if self.eat(","):
            try:
                z = self.float_number()
            except ParseError:
                self.pos = save
        self.expect(")")
        return (x, y, z)

    def parity_annotation(self):
        if not self.eat("@"):
            self.fail("@")
        if self.pos < len(self.s) and self.s[self.pos] in "RS":
            parity = self.s[self.pos]
            self.pos += 1
            return parity
        self.fail("R/S")

    def multiplicity_annotation(self):
        if not self.eat("@m("):
            self.fail("@m(")
        digits = self.digits()
        if digits is None:
            self.fail("digits")
        self.expect(")")
        return digits

    def title_annotation(self):
        if not self.eat('@t("'):
            self.fail('@t("')
        title = self.until_quote()
        self.expect('")')
        return title

    def fractional_annotation(self):
        if not self.eat("@f("):
            self.fail("@f(")
        x = self.float_number()
        self.expect(",")
        y = self.float_number()
        self.expect(",")
        z = self.float_number()
        self.expect(")")
        return (x, y, z)

    def float_number(self):
        m = re.match(r"-?\d+(\.\d*)?", self.s[self.pos:])
        if not m:
            self.fail("number")
        self.pos += m.end()
        return m.group()

    def ring_closures(self):
        m = re.match(r"\d+", self.s[self.pos:])
        if not m:
            self.fail("ring digits")
        self.pos += m.end()
        return m.group()

    def subscript_marker(self):
        if not self.eat("_"):
            self.fail("_")
        return "_" + self.subscript_value()

    def subscript_value(self):
        if self.eat("{"):
            end = self.s.find("}", self.pos)
            if end == -1:
                self.fail("}")
            value = "{" + self.s[self.pos:end] + "}"
            self.pos = end + 1
            return value
        digits = self.digits()
        if digits is None:
            self.fail("subscript digits")
        return digits

    def superscript_marker(self):
        if not self.eat("^"):
            self.fail("^")
        return "^" + self.superscript_value()

    def superscript_value(self):
        save = self.pos
        m = re.match(r"\([IVXLCDM]+\)", self.s[self.pos:])
        if m:
            self.pos += m.end()
            return m.group()
        self.pos = save
        m = re.match(r"\d+[+-]|[+-]\d*|\d+", self.s[self.pos:])
        if m:
            self.pos += m.end()
            return m.group()
        self.pos = save
        if self.eat("{"):
            end = self.s.find("}", self.pos)
            if end == -1:
                self.fail("}")
            value = "{" + self.s[self.pos:end] + "}"
            self.pos = end + 1
            return value
        m = re.match(r"[0-9a-zA-Z]+", self.s[self.pos:])
        if m:
            self.pos += m.end()
            return m.group()
        self.fail("superscript")

    def lewis_prefix(self):
        m = re.match(r":+", self.s[self.pos:])
        if not m:
            self.fail(":")
        self.pos += m.end()
        return m.group()

    def lewis_radicals(self):
        m = re.match(r"\.+", self.s[self.pos:])
        if not m:
            self.fail(".")
        self.pos += m.end()
        return m.group()

    def element_symbol(self):
        m = re.match(r"[A-Z][a-z]?", self.s[self.pos:])
        if not m:
            self.fail("element symbol")
        self.pos += m.end()
        return m.group()

    # -- beyond formulas ---------------------------------------------------

    def crystal(self):
        self.expect("crystal")
        name = self.try_rule(lambda: self.bracketed("[", "]"))
        params = self.try_rule(lambda: self.bracketed("(", ")"))
        body = self.try_rule(lambda: self.bracketed("{", "}"))
        return build_crystal(name, params, body)

    def bracketed(self, open_char, close_char):
        if not self.eat(open_char):
            self.fail(open_char)
        end = self.s.find(close_char, self.pos)
        if end == -1:
            self.fail(close_char)
        content = self.s[self.pos:end]
        self.pos = end + 1
        return content

    def spectrum(self):
        self.expect("spectrum")
        technique = self.try_rule(lambda: self.bracketed("[", "]"))
        params = self.try_rule(lambda: self.bracketed("(", ")"))
        body = self.try_rule(lambda: self.bracketed("{", "}"))
        return build_spectrum(technique, params, body)

    def calculation(self):
        self.expect("calc")
        params = self.try_rule(lambda: self.bracketed("(", ")"))
        body = self.try_rule(lambda: self.bracketed("{", "}"))
        return build_calculation(params, body)

    def zmatrix(self):
        self.expect("zmatrix")
        body = self.try_rule(lambda: self.bracketed("{", "}"))
        rows = []
        if body:
            for line in body.split("\n"):
                tokens = line.split()
                if not tokens:
                    continue
                rows.append(ZRow(*tokens, atom=tokens[0]) if False else ZRow(
                    atom=tokens[0],
                    ref1=tokens[1] if len(tokens) > 1 else None,
                    distance=tokens[2] if len(tokens) > 2 else None,
                    ref2=tokens[3] if len(tokens) > 3 else None,
                    angle=tokens[4] if len(tokens) > 4 else None,
                    ref3=tokens[5] if len(tokens) > 5 else None,
                    dihedral=tokens[6] if len(tokens) > 6 else None))
        return ZMatrix(rows)

    def mechanism(self):
        self.expect("mechanism")
        body = self.try_rule(lambda: self.bracketed("{", "}"))
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
                    steps.append(MechanismStep(key.strip(), value.strip()))
        return Mechanism(steps, spectators)

    def electron_config(self):
        pairs = []
        for _ in range(2):
            orbital = self.orbital()
            self.expect("^")
            occupancy = self.digits()
            if occupancy is None:
                self.fail("occupancy digits")
            self.skip_ws()
            pairs.append((orbital, occupancy))
        while True:
            save = self.pos
            try:
                orbital = self.orbital()
                self.expect("^")
                occupancy = self.digits()
                if occupancy is None:
                    raise ParseError("occupancy")
                self.skip_ws()
                pairs.append((orbital, occupancy))
            except ParseError:
                self.pos = save
                break
        return ElectronConfiguration(pairs)

    def orbital(self):
        m = re.match(r"\d+[spdfgh]", self.s[self.pos:])
        if not m:
            self.fail("orbital")
        self.pos += m.end()
        return m.group()

    def embedded_math(self):
        if not self.eat("`"):
            self.fail("`")
        end = self.s.find("`", self.pos)
        if end == -1:
            self.fail("closing `")
        source = self.s[self.pos:end]
        self.pos = end + 1
        return EmbeddedMath(source)

    def text_run(self):
        return Text(self.quoted())

    def group_text_run(self):
        return Text(self.quoted())

    def quoted(self):
        if not self.eat('"'):
            self.fail('"')
        end = self.s.find('"', self.pos)
        if end == -1:
            self.fail('closing "')
        content = self.s[self.pos:end]
        self.pos = end + 1
        return content

    # Stops AT the closing quote (the caller consumes it together
    # with the paren, mirroring the reference grammar).
    def until_quote(self):
        end = self.s.find('"', self.pos)
        if end == -1:
            self.fail('closing "')
        content = self.s[self.pos:end]
        self.pos = end
        return content

    def word(self):
        m = re.match(r"[a-z]+", self.s[self.pos:])
        if not m:
            self.fail("word")
        self.pos += m.end()
        return m.group()

    # -- primitives --------------------------------------------------------

    def digits(self):
        m = re.match(r"\d+", self.s[self.pos:])
        if not m:
            return None
        self.pos += m.end()
        return m.group()

    def skip_ws(self):
        while self.pos < len(self.s) and self.s[self.pos] in WS:
            self.pos += 1

    def eat(self, token):
        if self.s.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def expect(self, token):
        if not self.eat(token):
            self.fail(token)

    def fail(self, what):
        raise ParseError(
            f"Parse error at char {self.pos + 1}: expected {what}\n"
            f"  {self.s[self.pos:self.pos + 40]}\n"
            f"  {'^':>{min(self.pos + 3, 43)}}")


ARROW_TOKENS = {"<=>": "equilibrium", "<->": "resonance", "->": "forward", "<-": "reverse"}


# -- AtomBuilder (ported verbatim) ----------------------------------------

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


def build_atom(element, isotope, subscript, superscript, lone_pairs,
               radical_electrons, ring_closures, annotations):
    sup = strip_marker(superscript)
    charge = detect_charge(sup)
    oxidation = detect_oxidation(sup)
    coord = annotations.get("coord")
    fract = annotations.get("fract")
    return Atom(
        element=element,
        isotope=strip_marker(isotope),
        subscript=strip_marker(subscript, "_"),
        superscript=None if (charge or oxidation) else sup,
        charge=charge,
        oxidation_state=oxidation,
        lone_pairs=len(lone_pairs) if lone_pairs else None,
        radical_electrons=len(radical_electrons) if radical_electrons else None,
        ring_closures=ring_closures or None,
        x2=float(coord[0]) if coord else None,
        y2=float(coord[1]) if coord else None,
        z2=float(coord[2]) if coord else None,
        atom_parity=annotations.get("parity"),
        spin_multiplicity=annotations.get("spin"),
        atom_title=annotations.get("title"),
        x_fract=float(fract[0]) if fract else None,
        y_fract=float(fract[1]) if fract else None,
        z_fract=float(fract[2]) if fract else None,
    )


def apply_annotation(molecule, ann):
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


def build_crystal(name, params_raw, body):
    params = parse_params(params_raw)
    atoms = []
    if body:
        inner = Parser(body).parse()
        for node in inner.nodes:
            if isinstance(node, Molecule):
                atoms.extend(n for n in node.nodes if isinstance(n, Atom))
    def num(key):
        return float(params[key]) if key in params else None
    return Crystal(name=name or None, a=num("a"), b=num("b"), c=num("c"),
                   alpha=num("alpha"), beta=num("beta"), gamma=num("gamma"),
                   spacegroup=params.get("sg"), atoms=atoms)


def build_spectrum(technique, params_raw, body):
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


def build_calculation(params_raw, body):
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


def parse_text(source: str) -> Formula:
    return Parser(source).parse()
