"""One result schema, one writer, one aggregator input format.

Each old experiment invented its own CSV columns, so each needed its own
bespoke aggregator. Rows written through :class:`ResultWriter` all carry the
same core fields plus experiment-specific extras, so a single aggregator works
across experiments and a stale result is identifiable from the row itself.
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

#: Fields every row carries. `orbit_side` / `orbit_frame` are here because a
#: truth rank cannot be interpreted without knowing which convention produced
#: it, and `torchref_version` / `git_sha` because results outlive the checkout.
CORE_FIELDS = (
    "pdb",
    "seed",
    "trial",
    "spacegroup",
    "n_ops",
    "truth_rank",
    "truth_angle_deg",
    "orbit_side",
    "orbit_frame",
    "lmax_cap",
    "d_min",
    "d_max",
    "device",
    "torchref_version",
    "git_sha",
)


def _git_sha(default: str = "unknown") -> str:
    """Short SHA of the checkout this is running from, or ``default``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or default
    except Exception:
        return default


def provenance() -> Dict[str, str]:
    """Version and checkout identity for a result row.

    Returns
    -------
    dict
        ``{'torchref_version': ..., 'git_sha': ...}``.
    """
    try:
        import torchref

        version = getattr(torchref, "__version__", "unknown")
    except Exception:
        version = "unknown"
    return {"torchref_version": version, "git_sha": _git_sha()}


def append_row(csv_path: str | os.PathLike, row: Mapping[str, Any]) -> None:
    """Append one row, writing the header only when the file is new.

    The header is taken from the file once it exists, not from each row. Taking
    it per row silently loses data: a later row carrying a column the first row
    lacked gets written wider than the header, and ``DictReader`` then drops the
    surplus values into ``restkey``. That is how a sweep's anisotropy columns
    went missing while every other column still read back correctly. A row with
    an unknown column now raises; a row missing a known one gets a blank.

    Parameters
    ----------
    csv_path : path-like
        Destination CSV. Parent directories are created.
    row : mapping
        Column name -> value.

    Raises
    ------
    ValueError
        If ``row`` carries a column the file's header does not have.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: Optional[list] = None
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), None)
    if header:
        unknown = [k for k in row if k not in header]
        if unknown:
            raise ValueError(
                f"{path.name}: row has columns absent from the header: "
                f"{unknown}. Emit a stable set of columns for every row, using "
                f"blanks where a value does not apply."
            )
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header or list(row.keys()),
                                restval="")
        if fh.tell() == 0:
            writer.writeheader()
        writer.writerow(dict(row))


class ResultWriter:
    """Writes rows sharing a fixed core schema plus per-experiment extras.

    Parameters
    ----------
    csv_path : path-like
        Destination CSV.
    experiment : str
        Experiment tag, recorded in every row.
    extra_fields : iterable of str, optional
        Experiment-specific column names, appended after the core fields.

    Notes
    -----
    Column order is fixed at construction, so every row in a file has the same
    header even if a caller omits a value (missing entries are written empty).
    """

    def __init__(
        self,
        csv_path: str | os.PathLike,
        experiment: str,
        extra_fields: Optional[Iterable[str]] = None,
    ):
        self.path = Path(csv_path)
        self.experiment = experiment
        self.extra_fields = tuple(extra_fields or ())
        self.fieldnames = ("experiment",) + CORE_FIELDS + self.extra_fields
        self._provenance = provenance()

    def write(self, **values: Any) -> None:
        """Write one row; unknown keys raise rather than being dropped silently.

        Raises
        ------
        KeyError
            If a value is passed whose column was not declared.
        """
        unknown = set(values) - set(self.fieldnames)
        if unknown:
            raise KeyError(
                f"{self.experiment}: undeclared column(s) {sorted(unknown)}; "
                f"add them to extra_fields so every row keeps the same header"
            )
        row = {k: "" for k in self.fieldnames}
        row["experiment"] = self.experiment
        row.update(self._provenance)
        row.update(values)
        append_row(self.path, row)
