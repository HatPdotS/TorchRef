"""Phenix-style atom selections evaluated over per-atom identity arrays.

The grammar is the contract. Terms: ``chain <id>``, ``resseq <num>``,
``resseq <start>:<end>`` (inclusive), ``resname <name>``, ``name <atom>``,
``element <elem>``, ``altloc <id>`` and ``all``. They combine with ``not``, ``and`` and
``or`` -- binding in that order, tightest first -- and ``(...)`` groups, e.g.
``"chain A and (name CA or name CB)"``. ``resname``, ``name`` and ``element`` match
case-insensitively; ``chain`` and ``altloc`` do not.

A selection is evaluated against a mapping of per-atom arrays rather than an atom table,
so the same parser serves :meth:`torchref.topology.Topology.select` and anything else
that can produce the columns.
"""

import re
from typing import List, Mapping

import numpy as np
import torch

#: Columns a selection may read, each an array of shape ``(N,)``.
SELECTION_COLUMNS = ("chain", "resseq", "resname", "name", "element", "altloc")

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def select_atoms(columns: Mapping[str, np.ndarray], selection: str) -> torch.Tensor:
    """Evaluate a Phenix-style selection.

    Parameters
    ----------
    columns : mapping of str to numpy.ndarray
        Per-atom arrays keyed by :data:`SELECTION_COLUMNS`, all of shape ``(N,)``;
        ``resseq`` is integer, the rest strings.
    selection : str
        Selection string; see the module docstring for the grammar.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(N,)``, on the CPU.

    Raises
    ------
    ValueError
        On an empty selection, an unknown keyword, a term without a value, unbalanced
        parentheses or a trailing token.
    """
    tokens = _TOKEN.findall(selection)
    if not tokens:
        raise ValueError("Selection string cannot be empty")
    parser = _Parser(tokens, columns, selection)
    mask = parser.expression()
    if parser.pos != len(tokens):
        raise ValueError(
            f"Invalid selection syntax: unexpected {tokens[parser.pos]!r} in "
            f"{selection!r}"
        )
    return torch.as_tensor(mask, dtype=torch.bool)


class _Parser:
    """Recursive descent: ``or`` over ``and`` over ``not`` over terms and groups."""

    def __init__(self, tokens: List[str], columns: Mapping[str, np.ndarray], text: str):
        self.tokens = tokens
        self.columns = columns
        self.text = text
        self.pos = 0
        self.n = len(columns["name"])

    def _peek(self) -> str:
        return self.tokens[self.pos].lower() if self.pos < len(self.tokens) else ""

    def _take(self) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError(f"Invalid selection syntax: {self.text!r} ends early")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expression(self) -> np.ndarray:
        mask = self._conjunction()
        while self._peek() == "or":
            self._take()
            mask = mask | self._conjunction()
        return mask

    def _conjunction(self) -> np.ndarray:
        mask = self._unary()
        while self._peek() == "and":
            self._take()
            mask = mask & self._unary()
        return mask

    def _unary(self) -> np.ndarray:
        token = self._peek()
        if token == "not":
            self._take()
            return ~self._unary()
        if token == "(":
            self._take()
            mask = self.expression()
            if self._take() != ")":
                raise ValueError(f"Unbalanced parentheses in {self.text!r}")
            return mask
        if token == ")":
            raise ValueError(f"Unbalanced parentheses in {self.text!r}")
        return self._term()

    def _term(self) -> np.ndarray:
        keyword = self._take().lower()
        if keyword == "all":
            return np.ones(self.n, dtype=bool)
        if keyword in ("and", "or"):
            raise ValueError(f"Invalid selection syntax: {self.text!r}")
        if self._peek() in ("", "and", "or", ")", "("):
            raise ValueError(f"Invalid selection syntax: '{keyword}' has no value")
        value = self._take()
        cols = self.columns
        if keyword == "chain":
            return cols["chain"] == value
        if keyword == "resseq":
            resseq = np.asarray(cols["resseq"])
            if ":" in value:
                start, end = (int(v) for v in value.split(":"))
                return (resseq >= start) & (resseq <= end)
            return resseq == int(value)
        if keyword in ("resname", "name"):
            return _upper(cols[keyword]) == value.upper()
        if keyword == "element":
            return np.char.capitalize(np.char.strip(cols["element"].astype(str))) == (
                value.capitalize()
            )
        if keyword == "altloc":
            return cols["altloc"] == value
        raise ValueError(f"Unknown selection keyword: '{keyword}'")


def _upper(values: np.ndarray) -> np.ndarray:
    return np.char.upper(np.char.strip(np.asarray(values).astype(str)))


__all__ = ["select_atoms", "SELECTION_COLUMNS"]
