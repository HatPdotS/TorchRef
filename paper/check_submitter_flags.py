#!/usr/bin/env python3
"""Assert every benchmark submitter still emits flags its runner accepts.

The submitters paste command-line flags into sbatch script bodies **as text**. Nothing
connects that text to the parser on the other end, so a renamed or removed flag goes stale
silently until a few thousand SLURM jobs each die at argparse. That is not hypothetical: the
x-ray target rework removed ``--sigma-m-scale`` and five live submitters kept passing it, so
the whole benchmark harness could not submit a single job. The same hole swallows the subtler
defect -- a flag whose *value* left ``choices`` (``--xray-mode gaussian`` after ``gaussian``
was retired as an alias).

This closes it, and **changes nothing under** ``torchref/``: the guard reaches into
``refine.py`` rather than asking ``refine.py`` to expose anything for it. A check that
requires editing the thing it checks is a worse check.

Design
------
Driven by the explicit :data:`SUBMITTERS` table rather than by grepping for anything that
looks like a command, because the submitters do not all target the same parser:
``run_warm.py`` has its own, and ``log_shapes.py`` is a ``runpy`` shim that forwards to
``refine.py`` after eating two flags of its own. A table makes that visible; a regex would
quietly validate one script against the wrong parser.

Two levels of checking, by what the runner can hand over:

* **full** -- the real :class:`argparse.ArgumentParser` is captured out of ``refine.main()``
  (see :func:`refine_parser`), so flag names *and* literal values are validated, ``choices``
  and ``type`` included.
* **names** -- the runner builds its parser inline and is not worth executing, so the option
  strings are recovered by walking ``add_argument`` calls in the AST and only flag *names*
  are checked. Weaker, and labelled as such in the output.

Flag **values** are only ever checked where the submitter writes a literal
(``--xray-mode ml``). A value that comes from a placeholder (``{xray_mode}``, ``"$XMODE"``) is
replaced with one the parser accepts, so a placeholder can never be the reason this fails.
That split is deliberate: the campaign picks placeholder values at submit time, but a
*literal* that left ``choices`` is a bug in the file.

The epilog examples in ``refine.py`` are checked the same way -- a documented command the
parser rejects is invisible until someone types it.

Usage
-----
    ./.dev/bin/python paper/check_submitter_flags.py           # exit 1 on any failure
    ./.dev/bin/python paper/check_submitter_flags.py -v        # show every command checked
"""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # paper/
REPO = HERE.parent

sys.path.insert(0, str(REPO))

#: One row per submitter: the token naming the runner inside the sbatch body, and which parser
#: validates the flags. ``runner`` is a regex, and it is what makes detection unambiguous --
#: these files also name their runner in docstrings and in keyword arguments, and only the
#: sbatch body is a command line.
#:
#: ``parser``:
#:   ``("refine", None)``        -> the captured torchref.cli.refine parser  [full check]
#:   ``("ast", "<rel path>")``   -> option strings from that file's add_argument calls
#: ``extra`` lists flags the runner consumes itself before delegating.
SUBMITTERS = [
    ("paper/figure2_alphafold_start/run_af_pipeline.py", r"\{REFINE_SCRIPT\}", ("refine", None), ()),
    ("paper/figure2_alphafold_start/analysis/submit_local_arm.py", r"\{P\.REFINE_SCRIPT\}", ("refine", None), ()),
    ("paper/figure2_alphafold_start/analysis/submit_weight_grid.py", r"\{P\.REFINE_SCRIPT\}", ("refine", None), ()),
    ("paper/figure2_alphafold_start/analysis/submit_seeded_benchmark.py", r"\{P\.REFINE_SCRIPT\}", ("refine", None), ()),
    ("paper/extended_figures/exF4/submit_singlecore.py", r"\{P\.REFINE_SCRIPT\}", ("refine", None), ()),
    ("sigma_a_rework/submit_arms.py", r"\{refine\}", ("refine", None), ()),
    ("sigma_a_rework/submit_arm_grid.py", r"\{refine\}", ("refine", None), ()),
    ("sigma_a_rework/submit_weight_screen.py", r"\{refine\}", ("refine", None), ()),
    # runpy shim over refine.py; it strips its own two flags and delegates the rest.
    ("sigma_a_rework/estimator_lab/shapes_array.sh", r"log_shapes\.py", ("refine", None),
     ("--shape-log", "--shape-code")),
    # Own parser, in the same directory as its submitter.
    ("paper/seeded_warm_corefine/submit_warm.py", r"\{RUNNER\}",
     ("ast", "paper/seeded_warm_corefine/run_warm.py"), ()),
]

#: A command line **begins** with the interpreter. Anchoring there is what separates the sbatch
#: body from prose that merely names the runner (``PYTHON, REFINE_SCRIPT, CCP4_SETUP,
#: _sbatch)``) or passes it as a keyword (``python=P.PYTHON, refine=P.REFINE_SCRIPT``) -- both
#: of which mention an interpreter *somewhere* on the line, so "appears before the runner" is
#: not enough.
RE_INTERPRETER = re.compile(
    r"""^(?:\w+\s*=\s*\(?\s*)?       # `body = (` -- a body built by concatenation
         (?:f?["']\s*)?              # the opening f-string quote
         (?:\{[^{}]*PYTHON[^{}]*\}   # {PYTHON} / {P.PYTHON}
           |\{python\}               # lowercase template field
           |"?\$\{?PY\w*\}?"?        # "$PY" / ${PYTHON}
           |\S*/python[\d.]*         # an absolute interpreter path
           |python[\d.]*)\s""",
    re.VERBOSE)

#: A line of a Python body assembled by *concatenating* f-strings, e.g.
#: ``body = (f"{P.PYTHON} -u {P.REFINE_SCRIPT} \\\\\\n"`` or its last part
#: ``f"    -n 10 --mode separate")``. Keyed on the **opening** quote, not on a trailing ``\\n``
#: escape: the final part of a concatenation has no trailing newline, and missing it leaves a
#: stray ``f`` and ``)`` in the argv. A plain shell line never opens with a quote, which is
#: what makes this a safe discriminator -- stripping quote tails unconditionally would eat the
#: closing quote of ``--weights '{wjson}'``.
RE_FSTRING_PART = re.compile(r"""^\s*(?:\w+\s*=\s*\(?\s*)?f["']""")

#: A placeholder whose value is only known at submit time: an f-string field (``{mode}``), a
#: shell variable (``$MODE`` / ``"$MODE"`` / ``${MODE}``), or a leftover ``%s``.
RE_PLACEHOLDER = re.compile(r"\{[^{}]*\}|\$\{?\w+\}?|%[sdf]")

SENTINEL = "\x00"


# ── locating the invocation ───────────────────────────────────────────────────

def is_invocation(line: str, runner: re.Pattern) -> bool:
    """True if `line` *is* a shell invocation of `runner`, not a mention of it."""
    return bool(runner.search(line)) and bool(RE_INTERPRETER.match(line.lstrip()))


def _command_text(line: str):
    """One source line reduced to its shell-command text, plus whether it continues.

    Two spellings, both in use here, and they must be told apart. A plain shell line ends with
    one backslash and is otherwise verbatim. A concatenated-f-string part opens with a quote
    and carries a quote/paren/comma tail plus an escaped newline that have to come off -- but
    stripping those unconditionally would eat the closing quote of ``--weights '{wjson}'`` on
    a plain line and break tokenizing.
    """
    s = line.rstrip("\n").rstrip()
    if RE_FSTRING_PART.match(s):
        s = RE_FSTRING_PART.sub("", s)                     # the opening quote
        s = re.sub(r"""["'),]+$""", "", s).rstrip()        # the closing quote / paren / comma
        if s.endswith("\\n"):
            s = s[:-2].rstrip()                            # the escaped newline
    cont = s.endswith("\\")
    return s.rstrip("\\").rstrip(), cont


def continued_block(lines, start):
    """`start` plus every line the previous one continues onto."""
    out, i = [lines[start]], start
    while i < len(lines) and _command_text(lines[i])[1]:
        i += 1
        if i < len(lines):
            out.append(lines[i])
    return out


def to_argv(block, runner: re.Pattern):
    """Flatten an invocation block into ``(argv, literal_mask)``, blanking placeholders.

    ``literal_mask[i]`` is True when ``argv[i]`` came from the file verbatim; a placeholder
    contributes ``None``. The caller needs that to know which values are worth validating.
    """
    text = " ".join(_command_text(ln)[0] for ln in block)
    # Everything up to and including the runner token is the interpreter + script path.
    text = text[runner.search(text).end():]
    # A placeholder is replaced **in place**, keeping the surrounding word intact:
    # `${CODE}_af.pdb` is one path, not a sentinel plus a stray `_af.pdb` positional, and
    # `'{wjson}'` keeps its balanced quotes so shlex does not choke on a field we are not
    # validating anyway.
    marked = RE_PLACEHOLDER.sub(f"PH{SENTINEL}", text)
    try:
        words = shlex.split(marked)
    except ValueError as exc:
        raise ValueError(f"cannot tokenize: {exc}") from exc
    argv, literal = [], []
    for w in words:
        argv.append(None if SENTINEL in w else w)
        literal.append(SENTINEL not in w)
    return argv, literal


# ── the two parser providers ──────────────────────────────────────────────────

class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def refine_parser():
    """The real ``refine.py`` parser, captured without modifying ``refine.py``.

    ``main()`` builds its parser inline and there is no factory to import, so rather than
    asking the library to export one for this script's benefit, patch ``parse_args`` to hand
    the parser back and abort. ``main()`` does nothing but ``add_argument`` calls before that
    point, so nothing runs and no refinement starts.

    The alternative -- re-declaring the flag surface here -- would be a second copy of exactly
    the thing whose drift this script exists to detect.
    """
    import torchref.cli.refine as refine

    def grab(self, *a, **kw):
        raise _Captured(self)

    orig = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        refine.main()
    except _Captured as c:
        return c.parser
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError(
        "refine.main() returned without calling parse_args -- the capture hook is stale; "
        "check whether refine.py now builds its parser somewhere else.")


def flags_from_ast(rel):
    """Option strings from a script that builds its parser inline.

    Returns a flag -> expects_a_value mapping, recovered from literal ``add_argument`` string
    arguments. Names only -- no types, no choices.
    """
    tree = ast.parse((REPO / rel).read_text(), filename=rel)
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        opts = [a.value for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
                and a.value.startswith("-")]
        kw = {k.arg: k.value for k in node.keywords}
        action = kw.get("action")
        store_flag = (isinstance(action, ast.Constant)
                      and action.value in ("store_true", "store_false", "count"))
        for o in opts:
            out[o] = not store_flag
    return out


# ── checking ──────────────────────────────────────────────────────────────────

def _actions(parser):
    """flag -> (action, expects_a_value) for a real parser."""
    out = {}
    for a in parser._actions:
        needs = a.nargs != 0 and not isinstance(
            a, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                argparse._HelpAction, argparse._CountAction))
        for opt in a.option_strings:
            out[opt] = (a, needs)
    return out


def _stand_in(action):
    """A value this action will certainly accept, for use where the file has a placeholder."""
    if action is None:
        return "PLACEHOLDER"
    if action.choices:
        return str(list(action.choices)[0])
    if action.type is int:
        return "1"
    if action.type is float:
        return "1.0"
    return "PLACEHOLDER"


def check_argv(argv, literal, where, problems, *, parser=None, flags=None,
               extra=(), verbose=False):
    """Validate one argv. With `parser`, values are checked too; with `flags`, names only."""
    known = _actions(parser) if parser is not None else {
        f: (None, needs) for f, needs in flags.items()}
    for e in extra:
        known.setdefault(e, (None, True))

    resolved, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok is None:                       # placeholder in flag position: nothing to check
            i += 1
            continue
        if tok.startswith("-") and len(tok) > 1 and not tok[1:2].isdigit():
            flag = tok.split("=", 1)[0]
            if flag not in known:
                problems.append(f"{where}: unknown flag {flag!r}")
                # Swallow its value too, if it looks like one: a retired flag's argument
                # would otherwise reappear as a bare positional and produce a second,
                # confusing complaint about the same one defect.
                i += 1
                if i < len(argv) and (argv[i] is None or not argv[i].startswith("-")):
                    i += 1
                continue
            action, needs = known[flag]
            keep = flag not in extra          # runner-private flags never reach the parser
            if keep:
                resolved.append(tok)
            if needs and "=" not in tok:
                nxt_literal = i + 1 < len(argv) and literal[i + 1]
                if keep:
                    resolved.append(argv[i + 1] if nxt_literal else _stand_in(action))
                i += 2
                continue
            i += 1
            continue
        resolved.append(tok)
        i += 1

    if parser is None:                        # names-only: nothing left to validate
        if verbose:
            print(f"    [names] {' '.join(resolved)}")
        return

    # Required arguments come from placeholders in every submitter; supply them so the only
    # thing this can fail on is the flag surface itself.
    for a in parser._actions:
        if a.required and a.option_strings and a.option_strings[0] not in resolved:
            resolved += [a.option_strings[0], _stand_in(a)]

    if verbose:
        print(f"    {' '.join(resolved)}")
    try:
        parser.parse_args(resolved)
    except SystemExit:
        problems.append(f"{where}: argparse rejected: {' '.join(resolved)}")


def epilog_commands(parser):
    for raw in (parser.epilog or "").splitlines():
        line = raw.strip()
        if line.startswith("torchref.refine"):
            yield line


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print every resolved command line that gets parsed.")
    args = ap.parse_args()

    problems, n_checked = [], 0
    rp = refine_parser()

    print("── refine.py epilog examples ──")
    n_epilog = 0
    for cmd in epilog_commands(rp):
        argv = shlex.split(cmd)[1:]                       # drop the program name
        check_argv(argv, [True] * len(argv), f"refine.py epilog: {cmd}", problems,
                   parser=rp, verbose=args.verbose)
        n_checked += 1
        n_epilog += 1
        if args.verbose:
            print(f"  {cmd}")
    if not n_epilog:
        problems.append("refine.py: no epilog examples found (did the epilog move?)")
    print(f"  {n_epilog} example(s)")

    print("── submitters ──")
    for rel, runner, (kind, target), extra in SUBMITTERS:
        path = REPO / rel
        if not path.exists():
            problems.append(f"{rel}: MISSING (listed in SUBMITTERS but not on disk)")
            continue
        parser = rp if kind == "refine" else None
        flags = None if parser is not None else flags_from_ast(target)
        runner_re = re.compile(runner)
        lines = path.read_text().splitlines(keepends=True)
        found = 0
        for i, line in enumerate(lines):
            if not is_invocation(line.rstrip("\n"), runner_re):
                continue
            found += 1
            where = f"{rel}:{i + 1}"
            if args.verbose:
                print(f"  {where}")
            try:
                argv, literal = to_argv(continued_block(lines, i), runner_re)
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
                continue
            check_argv(argv, literal, where, problems, parser=parser, flags=flags,
                       extra=extra, verbose=args.verbose)
            n_checked += 1
        if not found:
            # Silence here would read as "clean" when it means "found nothing to check" --
            # the way a grep-for-survivors check goes vacuous.
            problems.append(
                f"{rel}: no {runner!r} invocation found (renamed runner, or moved?)")
        else:
            level = "full" if parser is not None else "names-only"
            print(f"  {rel}: {found} invocation(s) [{level}]")

    print()
    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  * {p}")
        return 1
    print(f"OK — {n_checked} command line(s) accepted by their runner's parser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
