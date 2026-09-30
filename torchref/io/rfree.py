"""
Uniform R-free flag assignment across several structure-factor files.

Time-resolved experiments refine many datasets of one crystal form (dark,
light, time points) and difference-refine them against each other. Every file
must then share one free set, otherwise reflections that are free in one
dataset are work reflections in another and cross-dataset R-free is biased.

Flags are assigned per unique reciprocal-ASU index (Friedel mates merged), so
symmetry equivalents, Friedel mates and F(+)/F(-) rows always share a flag.
Output follows the CCP4 ``FreeR_flag`` convention: integers ``0..N-1`` with
``0`` = free, so a different test set ``k`` can still be selected later.

All functions here operate on :class:`reciprocalspaceship.DataSet` objects so
that every original column of the input files is preserved on output.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
import reciprocalspaceship as rs

from torchref.io.mtz import MTZReader

FREE_COLUMN = "FreeR_flag"

# Existing flag columns that are replaced on output.
FLAG_COLUMN_NAMES = tuple(dict.fromkeys([*MTZReader.RFREE_FLAG_NAMES, FREE_COLUMN]))

_KEY_OFFSET = 1 << 10  # |h|, |k|, |l| < 1024


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_sf_file(path: str, cif_block: Optional[str] = None) -> rs.DataSet:
    """Read an MTZ or SF-mmCIF file into a DataSet, keeping every column.

    Parameters
    ----------
    path : str
        ``.mtz`` or ``.cif`` / ``.mmcif`` file.
    cif_block : str, optional
        Data block name for multi-block CIF files. Defaults to the first
        block that carries reflections.

    Returns
    -------
    rs.DataSet
        Reflections indexed by H, K, L with cell and space group attached.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".mtz":
        return rs.read_mtz(str(path))
    if suffix in (".cif", ".mmcif", ".ent"):
        blocks = gemmi.as_refln_blocks(gemmi.cif.read(str(path)))
        if cif_block is not None:
            blocks = [b for b in blocks if b.block.name == cif_block]
        if not blocks:
            raise ValueError(f"No reflection block found in {path}")
        mtz = gemmi.CifToMtz().convert_block_to_mtz(blocks[0])
        return rs.io.from_gemmi(mtz)
    raise ValueError(f"Unsupported structure-factor format: {path}")


def write_sf_file(ds: rs.DataSet, path: str) -> None:
    """Write a DataSet as MTZ or SF-mmCIF depending on the extension.

    CIF output uses gemmi's MTZ-to-mmCIF conversion, with ``FreeR_flag == 0``
    written as ``_refln.status 'f'`` and negative (excluded) flags as ``'x'``.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".mtz":
        ds.write_mtz(str(path))
    elif suffix in (".cif", ".mmcif"):
        converter = gemmi.MtzToCif()
        converter.free_flag_value = 0
        text = converter.write_cif_to_string(ds.to_gemmi())
        if FREE_COLUMN in ds.columns:
            text = _mark_excluded(text, ds)
        Path(path).write_text(text)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def _mark_excluded(text: str, ds: rs.DataSet) -> str:
    """Set ``_refln.status`` to ``x`` for rows whose ``FreeR_flag`` is negative.

    gemmi writes every non-free flag as ``o``, which would turn excluded
    reflections back into work reflections on the next read.
    """
    flags = ds[FREE_COLUMN].to_numpy(dtype=float)
    excluded = set(map(tuple, ds.get_hkls()[np.nan_to_num(flags, nan=0) < 0].tolist()))
    if not excluded:
        return text
    doc = gemmi.cif.read_string(text)
    for block in doc:
        table = block.find("_refln.", ["index_h", "index_k", "index_l", "status"])
        for row in table:
            if (int(row[0]), int(row[1]), int(row[2])) in excluded:
                row[3] = "x"
    return doc.as_string()


# ---------------------------------------------------------------------------
# Existing flags
# ---------------------------------------------------------------------------


def flag_column(ds: rs.DataSet, column: Optional[str] = None) -> Optional[str]:
    """Name of the R-free column in ``ds`` (``column`` if given), or None."""
    if column is not None:
        return column if column in ds.columns else None
    return next((c for c in FLAG_COLUMN_NAMES if c in ds.columns), None)


def excluded_rows(ds: rs.DataSet, column: Optional[str] = None) -> np.ndarray:
    """Rows marked excluded (negative or missing flag, CIF ``x``), shape (N,).

    All-false when ``ds`` has no R-free column. Independent of whether the
    remaining rows form a valid partition, so a column that excludes every
    row still excludes every row.
    """
    column = flag_column(ds, column)
    if column is None:
        return np.zeros(len(ds), dtype=bool)
    values = ds[column].to_numpy(dtype=float)
    return ~np.isfinite(values) | (np.nan_to_num(values, nan=-1) < 0)


def read_free_set(ds: rs.DataSet, column: Optional[str] = None) -> dict:
    """Interpret an existing R-free column row by row.

    Negative or missing values (MTZ ``-1``, CIF ``x``) are *excluded*. Among the
    remaining rows, a column with more than two values is CCP4 ``0..K``
    (0 = free); a binary column takes its majority value as work, which covers
    both CCP4 ``0 = free`` and Phenix ``1 = free``.

    Returns
    -------
    dict
        ``column``, ``convention``, ``raw`` (int, -1 where excluded),
        ``free`` and ``excluded`` (bool per row).
    """
    column = flag_column(ds, column)
    if column is None:
        raise ValueError("no recognised R-free column")
    values = ds[column].to_numpy(dtype=float)
    excluded = excluded_rows(ds, column)
    raw = np.where(excluded, -1, np.nan_to_num(values, nan=-1)).astype(np.int64)
    uvals, counts = np.unique(raw[~excluded], return_counts=True)
    if len(uvals) == 0:
        raise ValueError(f"R-free column {column!r} has no valid values")
    if len(uvals) > 2:
        convention = "ccp4"
        free = raw == 0
    else:
        work_value = uvals[np.argmax(counts)]
        convention = f"binary (work={int(work_value)})"
        free = ~excluded & (raw != work_value)
    return {
        "column": column,
        "convention": convention,
        "raw": raw,
        "free": free,
        "excluded": excluded,
    }


def _group_free(keys: np.ndarray, free: np.ndarray):
    """Reduce rows to unique ASU keys; returns (keys, free, conflicting).

    ``conflicting`` marks keys whose rows (symmetry equivalents, Friedel mates
    or duplicates) are not all free or all work.
    """
    ukeys, inverse = np.unique(keys, return_inverse=True)
    n_free = np.bincount(inverse, weights=free, minlength=len(ukeys))
    n_rows = np.bincount(inverse, minlength=len(ukeys))
    return ukeys, n_free > 0, (n_free > 0) & (n_free < n_rows)


def compare_free_sets(
    datasets: Dict[str, rs.DataSet], column: Optional[str] = None
) -> dict:
    """Report whether the existing free sets of several datasets agree.

    Returns
    -------
    dict
        ``files``: name to ``column``/``convention``/``n``/``n_free``/
        ``n_excluded``/``n_conflicting``/``dmin`` (or ``column: None`` without
        a usable free set, with ``problem`` saying why). ``n_conflicting``
        counts unique reflections whose equivalents within that file disagree.
        ``pairs``: ``(a, b)`` to ``(n_common, n_disagree)`` over unique
        reflections that are present, not excluded and not conflicting in
        both. ``consistent`` requires every file to have a free set with free
        reflections, no internal conflicts and no pairwise disagreement.
    """
    files, tables = {}, {}
    for name, ds in datasets.items():
        dmin = float(ds.compute_dHKL()["dHKL"].min())
        try:
            fs = read_free_set(ds, column)
        except ValueError as exc:
            files[name] = {
                "column": None,
                "n": len(ds),
                "dmin": dmin,
                "problem": str(exc),
            }
            continue
        keep = ~fs["excluded"]
        ukeys, ufree, conflict = _group_free(
            hkl_keys(asu_hkl(ds))[keep], fs["free"][keep]
        )
        tables[name] = (ukeys[~conflict], ufree[~conflict])
        files[name] = {
            "column": fs["column"],
            "convention": fs["convention"],
            "n": len(ds),
            "n_free": int(fs["free"].sum()),
            "n_excluded": int(fs["excluded"].sum()),
            "n_conflicting": int(conflict.sum()),
            "dmin": dmin,
        }
    pairs = {}
    names = list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ka, fa = tables[a]
            kb, fb = tables[b]
            common, ia, ib = np.intersect1d(
                ka, kb, assume_unique=True, return_indices=True
            )
            pairs[(a, b)] = (len(common), int((fa[ia] != fb[ib]).sum()))
    consistent = (
        len(tables) == len(datasets)
        and all(
            files[n]["n_free"] > 0 and files[n]["n_conflicting"] == 0 for n in tables
        )
        and all(d == 0 for _, d in pairs.values())
    )
    return {"files": files, "pairs": pairs, "consistent": consistent}


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def check_compatible(
    datasets: Dict[str, rs.DataSet],
    length_tol: float = 0.01,
    angle_tol: float = 0.5,
) -> List[str]:
    """Check that all datasets share a space group and (nearly) a cell.

    Parameters
    ----------
    datasets : dict
        Name to DataSet.
    length_tol : float
        Allowed relative deviation of a, b, c from the first dataset.
    angle_tol : float
        Allowed absolute deviation of alpha, beta, gamma in degrees.

    Returns
    -------
    list of str
        Human-readable problems; empty when all datasets are compatible.
    """
    names = list(datasets)
    ref_name = names[0]
    ref = datasets[ref_name]
    ref_cell = np.array(ref.cell.parameters)
    problems = []
    for name in names[1:]:
        ds = datasets[name]
        if ds.spacegroup.xhm() != ref.spacegroup.xhm():
            problems.append(
                f"{name}: space group {ds.spacegroup.xhm()!r} != "
                f"{ref.spacegroup.xhm()!r} ({ref_name})"
            )
        cell = np.array(ds.cell.parameters)
        rel = np.abs(cell[:3] - ref_cell[:3]) / ref_cell[:3]
        dang = np.abs(cell[3:] - ref_cell[3:])
        if (rel > length_tol).any() or (dang > angle_tol).any():
            problems.append(
                f"{name}: cell {tuple(np.round(cell, 3))} differs from "
                f"{tuple(np.round(ref_cell, 3))} ({ref_name})"
            )
    return problems


# ---------------------------------------------------------------------------
# Miller-index keys
# ---------------------------------------------------------------------------


def asu_hkl(ds: rs.DataSet) -> np.ndarray:
    """Friedel-merged reciprocal-ASU indices for every row, shape (N, 3)."""
    hkl = ds.get_hkls()
    return rs.utils.hkl_to_asu(hkl, ds.spacegroup)[0].astype(np.int64)


def hkl_keys(hkl: np.ndarray) -> np.ndarray:
    """Encode integer Miller indices (N, 3) as unique int64 scalars."""
    h = hkl.astype(np.int64) + _KEY_OFFSET
    span = 2 * _KEY_OFFSET
    return (h[:, 0] * span + h[:, 1]) * span + h[:, 2]


def _unkey(keys: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hkl_keys`."""
    span = 2 * _KEY_OFFSET
    return (
        np.stack([keys // span**2, keys // span % span, keys % span], 1) - _KEY_OFFSET
    )


def _lookup(table_keys: np.ndarray, table_values: np.ndarray, keys: np.ndarray):
    """Look up ``keys`` in sorted ``table_keys``; returns (values, found)."""
    pos = np.searchsorted(table_keys, keys)
    pos = np.clip(pos, 0, len(table_keys) - 1)
    found = table_keys[pos] == keys
    return np.where(found, table_values[pos], -1), found


# ---------------------------------------------------------------------------
# Flag generation
# ---------------------------------------------------------------------------


def resolution_bins(dstar2: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count resolution bin index (0..n_bins-1) for each 1/d^2 value."""
    n_bins = max(1, min(n_bins, len(dstar2)))
    order = np.argsort(dstar2, kind="stable")
    bins = np.empty(len(dstar2), dtype=np.int64)
    bins[order] = np.arange(len(dstar2)) * n_bins // max(len(dstar2), 1)
    return bins


def _hash_flags(hkl: np.ndarray, n_flags: int, seed: int) -> np.ndarray:
    """Deterministic pseudo-random flag from the Miller index alone."""
    k = hkl_keys(hkl).astype(np.uint64) ^ np.uint64(seed * 0x9E3779B97F4A7C15 % 2**64)
    k ^= k >> np.uint64(33)
    k *= np.uint64(0xFF51AFD7ED558CCD)
    k ^= k >> np.uint64(33)
    return (k % np.uint64(n_flags)).astype(np.int32)


def partition_seed(keys: np.ndarray, free: np.ndarray) -> int:
    """Seed derived from a free/work partition (SHA-256 of keys and free mask).

    Depends only on which unique reflections are free and which are work, not
    on file format, row order or flag convention, so every extension of the
    same deposited free set is identical.
    """
    order = np.argsort(keys)
    digest = hashlib.sha256(
        np.ascontiguousarray(keys[order], dtype="<i8").tobytes()
        + np.ascontiguousarray(free[order], dtype=np.uint8).tobytes()
    ).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def complete_flag_table(
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    dmin: float,
    n_flags: int,
    seed: int = 0,
    shell_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified CCP4 flags on the complete reciprocal ASU out to ``dmin``.

    The complete ASU is sorted by resolution (ties by index) and cut into
    consecutive shells of ``shell_size`` reflections. Each shell is shuffled
    with its own seed ``(seed, shell)`` and dealt flag values round-robin, so
    every value (in particular the free value 0) holds exactly
    ``1 / n_flags`` of every shell. The last shell is always completed with
    reflections beyond ``dmin``, so the flag of any reflection depends only on
    cell, space group, ``n_flags``, ``shell_size`` and ``seed``: a larger
    ``dmin`` (a later, better dataset) never changes existing flags.

    Returns
    -------
    keys : np.ndarray
        Sorted ASU keys (see :func:`hkl_keys`), shape (M,).
    flags : np.ndarray
        Flag per key, shape (M,).
    """
    shell_size = max(n_flags, shell_size - shell_size % n_flags)
    d = dmin
    while True:
        hkl = rs.utils.generate_reciprocal_asu(cell, spacegroup, d, anomalous=False)
        hkl = hkl.astype(np.int64)
        dstar2 = np.round(1.0 / rs.utils.compute_dHKL(hkl, cell) ** 2, 10)
        n_needed = int((dstar2 <= 1.0 / dmin**2).sum())
        n_total = -(-n_needed // shell_size) * shell_size
        if len(hkl) >= n_total:
            break
        d *= 0.95
    order = np.lexsort((hkl[:, 2], hkl[:, 1], hkl[:, 0], dstar2))[:n_total]
    hkl = hkl[order]
    flags = np.empty(len(hkl), dtype=np.int32)
    deal = np.arange(shell_size) % n_flags
    for shell in range(n_total // shell_size):
        rng = np.random.default_rng([seed, shell])
        flags[shell * shell_size + rng.permutation(shell_size)] = deal
    keys = hkl_keys(hkl)
    idx = np.argsort(keys)
    return keys[idx], flags[idx]


def reference_flags(
    ref: rs.DataSet,
    n_flags: int,
    column: Optional[str] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Extract a CCP4-style flag per unique ASU reflection from a reference.

    Conventions are read by :func:`read_free_set`; excluded reflections are
    skipped. Multi-valued columns (CCP4 ``0..K``) are kept as-is. For binary
    columns free becomes ``0`` and work reflections are dealt pseudo-random
    values ``1..n_flags-1``.

    Returns
    -------
    keys : np.ndarray
        Sorted unique ASU keys, shape (M,).
    flags : np.ndarray
        Flag per key, shape (M,).
    info : dict
        ``column``, ``convention``, ``n_inconsistent`` (unique reflections
        whose symmetry equivalents carried different flags), ``n_values``,
        ``free_fraction`` (per unique reflection) and ``dmin``.
    """
    fs = read_free_set(ref, column)
    column, convention = fs["column"], fs["convention"]
    keep = ~fs["excluded"]
    raw = fs["raw"][keep]
    free_rows = fs["free"][keep]
    ref_hkl = asu_hkl(ref)[keep]
    keys = hkl_keys(ref_hkl)
    ref_dmin = float(rs.utils.compute_dHKL(ref_hkl, ref.cell).min())

    if not free_rows.any():
        raise ValueError(
            f"reference column {column!r} ({convention}) marks no reflection as "
            "free; choose another --reference or use --fresh"
        )
    ukeys, inverse = np.unique(keys, return_inverse=True)
    # a unique reflection is inconsistent if its equivalents disagree
    lo = np.full(len(ukeys), np.iinfo(np.int64).max)
    hi = np.full(len(ukeys), np.iinfo(np.int64).min)
    np.minimum.at(lo, inverse, raw)
    np.maximum.at(hi, inverse, raw)
    n_inconsistent = int((lo != hi).sum())

    if convention == "ccp4":
        # conflicting equivalents: free (0) wins, else the smallest value
        uflags = lo
    else:
        free_any = np.zeros(len(ukeys), dtype=bool)
        np.logical_or.at(free_any, inverse, free_rows)
        work = 1 + _hash_flags(_unkey(ukeys), n_flags - 1, seed)
        uflags = np.where(free_any, 0, work)
    info = {
        "column": column,
        "convention": convention,
        "n_inconsistent": n_inconsistent,
        "n_values": int(uflags.max()) + 1,
        "free_fraction": float((uflags == 0).mean()),
        "dmin": ref_dmin,
    }
    return ukeys, uflags.astype(np.int32), info


def uniform_rfree(
    datasets: Dict[str, rs.DataSet],
    free_fraction: Optional[float] = None,
    shell_size: int = 1000,
    seed: Optional[int] = None,
    dmin: Optional[float] = None,
    reference: Optional[rs.DataSet] = None,
    reference_column: Optional[str] = None,
    max_free: Optional[int] = None,
    keep_excluded: bool = True,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Compute one shared CCP4 ``FreeR_flag`` column for several datasets.

    Flags come from :func:`complete_flag_table` on the reference's (else the
    first dataset's) cell,
    so they depend only on cell, space group and settings, not on which
    reflections happen to be measured. A dataset processed later with the
    same settings therefore gets identical flags for every reflection.

    Parameters
    ----------
    datasets : dict
        Name to DataSet (same cell / space group).
    free_fraction : float, optional
        Target free fraction; the number of flag values is ``round(1/f)``.
        Defaults to the reference's own fraction when inheriting, else 0.05.
    shell_size : int
        Reflections per stratification shell (see :func:`complete_flag_table`).
    seed : int, optional
        Random seed. By default ``0`` for a new set and, when extending a
        reference, :func:`partition_seed` of the reference's free set, so the
        flags assigned to reflections the reference lacks (e.g. its missing
        high-resolution shells) are a deterministic function of the reference.
    dmin : float, optional
        High-resolution limit of the flag table; the best resolution of the
        inputs is used if it is finer.
    reference : rs.DataSet, optional
        Dataset whose existing flags are inherited; reflections it lacks are
        newly assigned.
    reference_column : str, optional
        Flag column in ``reference`` (auto-detected by default).
    max_free : int, optional
        Cap on the number of free reflections in the complete set to ``dmin``
        (Phenix-style); lowers the fraction for large datasets. The cap depends
        on ``dmin``, so reuse the reported fraction to reproduce a flag set.
    keep_excluded : bool
        Rows a dataset itself marks as excluded (negative flag, CIF ``x``)
        stay ``-1`` in that dataset's output.

    Returns
    -------
    flags : dict
        Name to per-row ``FreeR_flag`` array aligned with each DataSet.
    info : dict
        Summary: ``n_flags``, ``free_fraction``, ``fraction_source``, ``seed``,
        ``seed_source``, ``dmin``, ``n_gaps_in_reference``,
        ``n_unique``, ``n_off_asu``, ``n_inherited``, ``n_generated``,
        ``n_generated_beyond_reference``, ``n_excluded`` (per file) and the
        reference ``info``.
    """
    if free_fraction is not None and not 0 < free_fraction < 1:
        raise ValueError("free_fraction must be between 0 and 1")

    # the flag table lives on the reference's cell when there is one, so the
    # result does not depend on input order
    first = reference if reference is not None else next(iter(datasets.values()))
    cell, sg = first.cell, first.spacegroup
    row_hkl = {name: asu_hkl(ds) for name, ds in datasets.items()}
    observed = np.unique(np.concatenate(list(row_hkl.values())), axis=0)
    data_dmin = float(rs.utils.compute_dHKL(observed, cell).min())
    dmin = min(dmin, data_dmin) if dmin is not None else data_dmin

    rkeys = rflags = rinfo = None
    if reference is not None:
        # n_flags and seed only affect the pseudo-random work values here
        rkeys, rflags, rinfo = reference_flags(
            reference, 20, column=reference_column, seed=0
        )
    if seed is not None:
        seed_source = "user"
    elif rinfo is not None:
        seed, seed_source = partition_seed(rkeys, rflags == 0), "reference free set"
    else:
        seed, seed_source = 0, "default"
    if free_fraction is not None:
        n_flags, source = max(2, int(round(1.0 / free_fraction))), "user"
    elif rinfo is not None and rinfo["convention"] == "ccp4":
        n_flags, source = max(2, rinfo["n_values"]), "reference"
    elif rinfo is not None:
        n_flags, source = max(2, int(round(1.0 / rinfo["free_fraction"]))), "reference"
    else:
        n_flags, source = 20, "default"
    if max_free is not None:
        n_complete = len(
            rs.utils.generate_reciprocal_asu(cell, sg, dmin, anomalous=False)
        )
        if n_complete / n_flags > max_free:
            n_flags, source = (
                int(np.ceil(n_complete / max_free)),
                f"max_free={max_free}",
            )
    if rinfo is not None and rinfo["convention"] != "ccp4":
        rkeys, rflags, rinfo = reference_flags(
            reference, n_flags, column=reference_column, seed=seed
        )
    ukeys, flags = complete_flag_table(cell, sg, dmin, n_flags, seed, shell_size)

    # observed indices outside the complete set (e.g. systematic absences)
    okeys = hkl_keys(observed)
    _, found = _lookup(ukeys, flags, okeys)
    if not found.all():
        ukeys = np.concatenate([ukeys, okeys[~found]])
        flags = np.concatenate([flags, _hash_flags(observed[~found], n_flags, seed)])
        idx = np.argsort(ukeys)
        ukeys, flags = ukeys[idx], flags[idx]
    info = {
        "n_flags": n_flags,
        "free_fraction": 1.0 / n_flags,
        "fraction_source": source,
        "seed": seed,
        "seed_source": seed_source,
        "dmin": dmin,
        "n_unique": len(ukeys),
        "n_off_asu": int((~found).sum()),
    }

    if rinfo is not None:
        inherited, found = _lookup(rkeys, rflags, ukeys)
        flags = np.where(found, inherited, flags).astype(np.int32)
        # restrict counts to reflections some input actually has
        seen = np.isin(ukeys, okeys)
        d = rs.utils.compute_dHKL(_unkey(ukeys[seen & ~found]), cell)
        n_beyond = int((d < rinfo["dmin"] - 1e-6).sum())
        info.update(
            reference=rinfo,
            n_inherited=int((found & seen).sum()),
            n_generated=int((~found & seen).sum()),
            n_generated_beyond_reference=n_beyond,
            n_gaps_in_reference=int((~found & seen).sum()) - n_beyond,
        )
    else:
        info.update(
            n_inherited=0,
            n_generated=len(okeys),
            n_generated_beyond_reference=0,
            n_gaps_in_reference=0,
        )

    per_file, n_excluded = {}, {}
    for name, hkl in row_hkl.items():
        values, found = _lookup(ukeys, flags, hkl_keys(hkl))
        assert found.all(), "every observed reflection is in the universe"
        values = values.astype(np.int32)
        if keep_excluded:
            excluded = excluded_rows(datasets[name])
        else:
            excluded = np.zeros(len(values), dtype=bool)
        values[excluded] = -1
        n_excluded[name] = int(excluded.sum())
        per_file[name] = values
    info["n_excluded"] = n_excluded
    return per_file, info


def apply_flags(
    ds: rs.DataSet, flags: np.ndarray, keep_old: bool = False
) -> rs.DataSet:
    """Return a copy of ``ds`` whose only R-free column is ``FreeR_flag``.

    Existing flag columns are dropped, or renamed ``<name>_orig`` with
    ``keep_old``. Row order and all other columns are unchanged.
    """
    out = ds.copy()
    for col in [c for c in out.columns if c in FLAG_COLUMN_NAMES]:
        if keep_old:
            out = out.rename(columns={col: f"{col}_orig"})
        else:
            out = out.drop(columns=col)
    out[FREE_COLUMN] = rs.DataSeries(flags, index=out.index, dtype="I")
    return out


# ---------------------------------------------------------------------------
# Scale application
# ---------------------------------------------------------------------------

_AMPLITUDE_TYPES = {"F", "G", "D", "L"}  # F, F(+/-), anomalous diff, sigma F(+/-)
_INTENSITY_TYPES = {"J", "K", "M"}  # I, I(+/-), sigma I(+/-)


_CALC_PREFIXES = (
    "FC",
    "FCALC",
    "FMODEL",
    "F-MODEL",
    "FCAL",
    "FWT",
    "DELFWT",
    "2FOFC",
    "FOFC",
)


def _is_calculated(name: str) -> bool:
    """Amplitude column names that hold model or map values, not observations."""
    n = name.upper()
    return n.startswith(_CALC_PREFIXES) or n.endswith("WT")


def scale_columns(ds: rs.DataSet, factor: np.ndarray) -> Tuple[rs.DataSet, List[str]]:
    """Multiply amplitude columns by ``factor`` and intensity columns by its square.

    Generic standard deviations (MTZ type ``Q``) follow the preceding column
    or their ``SIG<name>`` partner. Map coefficients and calculated amplitudes
    (an amplitude directly followed by a phase column, e.g. ``FWT``/``PHWT``,
    or a ``FC``/``FCALC``/``FMODEL``-style name), phases, weights, flags and
    other columns are untouched.

    Returns
    -------
    rs.DataSet
        Scaled copy.
    list of str
        Names of the columns that were scaled.
    """
    out = ds.copy()
    columns = list(out.columns)
    types = {c: out[c].dtype.mtztype for c in columns}
    scaled = []
    prev = None
    for i, col in enumerate(columns):
        t = types[col]
        power = None
        nxt = columns[i + 1] if i + 1 < len(columns) else None
        if t == "F" and (types.get(nxt) == "P" or _is_calculated(col)):
            pass  # map coefficient or model amplitude
        elif t in _AMPLITUDE_TYPES:
            power = 1
        elif t in _INTENSITY_TYPES:
            power = 2
        elif t == "Q":
            partner = col[3:] if col.upper().startswith("SIG") else None
            ref_type = types.get(partner) or (types.get(prev) if prev else None)
            if ref_type in _AMPLITUDE_TYPES:
                power = 1
            elif ref_type in _INTENSITY_TYPES:
                power = 2
        if power is not None:
            dtype = out[col].dtype
            out[col] = rs.DataSeries(
                out[col].to_numpy(dtype=float) * factor**power,
                index=out.index,
                dtype=dtype,
            )
            scaled.append(col)
        prev = col
    return out, scaled
