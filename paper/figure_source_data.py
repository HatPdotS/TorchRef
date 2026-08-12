"""Shared source-data writer for the paper's plot scripts.

Every ``plot_*.py`` calls :func:`dump` from inside the panel that computed the values, so the
CSV is written from the *same* arrays the panel draws. That is the point: a separate script
that re-derived the panel values would be a second implementation to keep in step, and would
silently disagree the moment a panel's filtering changed.

CSVs land in ``paper/source_data/`` and are named ``<figure>_<panel>_<what>.csv``.

Typical use, at the end of a panel function::

    from figure_source_data import dump
    dump("figure2_panelA_rfactors",
         [{"engine": e, "code": c, "r_work": w, "r_free": f} for ...])
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["dump", "outdir", "rounded"]

_WRITTEN: list[str] = []


def outdir() -> Path:
    """``paper/source_data/``, located by walking up from this file."""
    d = Path(__file__).resolve().parent / "source_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rounded(x, nd: int = 6):
    """Round floats for output; pass None/NaN through as an empty cell.

    Bools, ints and integral numpy scalars are preserved as integers rather than coerced to
    float -- otherwise counts and indices come out as ``723.0`` / ``0.0`` and a boolean flag
    reads as ``0.0``, which is both ugly and ambiguous in a published data file.

    Strings are written verbatim, never parsed as numbers. Several PDB codes are valid
    float literals -- ``float("3E98")`` is 3e+98 -- so coercing them silently renamed
    structure 3E98 to ``3e+98`` in the published source data. A string in a cell is a
    label, not a measurement.
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return x
    if fx != fx:          # NaN
        return None
    # Deliberately NOT collapsing integral floats to int: a measured RMSZ of exactly 1.0 is
    # a float measurement, and writing it as "1" would misrepresent the column's type.
    return round(fx, nd)


def dump(name: str, rows, cols: list[str] | None = None, nd: int = 6) -> Path | None:
    """Write ``rows`` (list of dicts) to ``paper/source_data/<name>.csv``.

    ``cols`` defaults to the first row's keys, so column order follows insertion order.
    Float values are passed through :func:`rounded`. An empty ``rows`` writes nothing and
    warns -- a 0-row source-data file looks like a successful export of an empty panel.
    """
    rows = list(rows)
    if not rows:
        print(f"  source-data: {name} has NO rows — not written")
        return None
    cols = cols or list(rows[0].keys())
    path = outdir() / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: rounded(r.get(k), nd) for k in cols})
    _WRITTEN.append(name)
    print(f"  source-data: {path.name}  ({len(rows)} rows)")
    return path
