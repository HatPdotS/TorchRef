#!/usr/bin/env python3 -u

"""Give several structure-factor files one shared R-free set.

Intended as a pre-processing step for time-resolved (TR-SFX) data: every
dataset of one crystal form (dark, light, time points) gets the same CCP4
``FreeR_flag`` column before refinement / difference refinement, so no
reflection is free in one dataset and work in another. Optionally, the
datasets are also put on a common scale with the joint dataset scaler
(:meth:`torchref.io.datasets.collection.DatasetCollection.scale`).

If any input already carries an R-free column, its free set is inherited and
extended, at its own fraction, to the reflections it lacks (``--reference
auto``, the default); otherwise a new set is generated. ``--free-fraction`` and
``--max-free`` size a new set, so they need ``--fresh`` when an input has
flags. ``--check`` only reports whether the inputs' existing free sets agree.

Usage::

    torchref.uniform-rfree dark.mtz light_*.mtz --check
    torchref.uniform-rfree dark.mtz light_*.mtz -o flagged/
    torchref.uniform-rfree dark.mtz light.cif --reference deposited-sf.cif \\
        --format mtz cif --scale --scale-reference dark -o flagged/
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from torchref.cli._common import (
    add_device_arg,
    add_outdir_arg,
    add_verbose_arg,
    configure_unbuffered_output,
)

configure_unbuffered_output()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="torchref.uniform-rfree",
        description=(
            "Assign one uniform R-free set (CCP4 FreeR_flag, 0 = free) to any "
            "number of MTZ / SF-mmCIF files sharing a cell and space group, "
            "optionally scaling them together."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Do the existing free sets agree? (writes nothing; exit code 2 if not)
  torchref.uniform-rfree dark.mtz light_*.mtz --check

  # Shared free set: inherited from the first input that has one, else new
  torchref.uniform-rfree dark.mtz light_*.mtz -o flagged/

  # Inherit the free set of a deposited dark structure, extend to new reflections
  torchref.uniform-rfree dark.mtz light.mtz --reference 1abc-sf.cif -o flagged/

  # Ignore all existing flags; new 5% set capped at 2000 free reflections
  torchref.uniform-rfree dark.mtz light_*.mtz --fresh --max-free 2000 -o flagged/

  # Write MTZ and mmCIF, and scale all light datasets onto dark
  torchref.uniform-rfree dark.mtz light_*.mtz --format mtz cif \\
      --scale --scale-reference dark -o flagged/
""",
    )

    inp = parser.add_argument_group("Input")
    inp.add_argument("files", nargs="+", help="Structure-factor files (.mtz or .cif)")
    inp.add_argument(
        "--cif-block", default=None, help="Data block to read from CIF inputs"
    )
    inp.add_argument(
        "--length-tol",
        type=float,
        default=0.01,
        help="Allowed relative cell-length deviation (default: 0.01)",
    )
    inp.add_argument(
        "--angle-tol",
        type=float,
        default=0.5,
        help="Allowed cell-angle deviation in degrees (default: 0.5)",
    )
    inp.add_argument(
        "--force",
        action="store_true",
        help="Proceed despite cell / space-group mismatches",
    )

    flg = parser.add_argument_group("Flags")
    flg.add_argument(
        "--check",
        action="store_true",
        help=(
            "Only report the inputs' existing free sets and whether they agree; "
            "write nothing (exit code 0 if consistent, 2 if not)"
        ),
    )
    flg.add_argument(
        "--free-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of a new free set; FreeR_flag takes round(1/f) values "
            "(default: 0.05). An inherited set keeps its own fraction, so this "
            "needs --fresh when an input has flags"
        ),
    )
    flg.add_argument(
        "--max-free",
        type=int,
        default=None,
        help=(
            "Cap a new free set at this many reflections of the complete set to "
            "the best resolution (e.g. 2000, as in Phenix); lowers the fraction. "
            "Needs --fresh when an input has flags"
        ),
    )
    flg.add_argument(
        "--shell-size",
        type=int,
        default=1000,
        help=(
            "Reflections per resolution shell; each shell holds exactly the "
            "free fraction (default: 1000)"
        ),
    )
    flg.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed (default: 0 for a new set; when extending a reference, a "
            "hash of its free set, so every extension of it is identical)"
        ),
    )
    flg.add_argument(
        "--dmin",
        type=float,
        default=None,
        help=(
            "Extend the flag table to at least this resolution (default: best "
            "resolution of the inputs). Flags never depend on this value; it "
            "only matters for what gets reported."
        ),
    )
    ref = flg.add_mutually_exclusive_group()
    ref.add_argument(
        "--reference",
        default="auto",
        help=(
            "File whose existing R-free flags are inherited and extended (may be "
            "one of the inputs). 'auto' (default): the first input with an R-free "
            "column; a new set is generated if none has one"
        ),
    )
    ref.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing flags and generate a new free set",
    )
    flg.add_argument(
        "--reference-column",
        default=None,
        help=(
            "R-free column in --reference, or in the inputs with --reference auto "
            "and --check (default: auto-detect)"
        ),
    )

    scl = parser.add_argument_group("Scaling")
    scl.add_argument(
        "--scale",
        action="store_true",
        help=(
            "Jointly scale the datasets (overall + anisotropic) with the "
            "dataset scaler; work reflections only"
        ),
    )
    scl.add_argument(
        "--scale-reference",
        default=None,
        help=(
            "Input (file name or stem) kept unscaled; others are put on its scale. "
            "Default: the consensus scale of all inputs"
        ),
    )
    scl.add_argument("--scale-nsteps", type=int, default=10, help=argparse.SUPPRESS)
    scl.add_argument("--scale-max-iter", type=int, default=100, help=argparse.SUPPRESS)
    add_device_arg(scl)

    out = parser.add_argument_group("Output")
    add_outdir_arg(
        out, required=False, help="Output directory (required unless --check)"
    )
    out.add_argument(
        "--format",
        nargs="+",
        choices=["mtz", "cif"],
        default=["mtz"],
        help="Output format(s) (default: mtz)",
    )
    out.add_argument(
        "--suffix",
        default="_rfree",
        help="Suffix appended to each output file stem (default: _rfree)",
    )
    out.add_argument(
        "--keep-old-flags",
        action="store_true",
        help="Keep existing flag columns renamed to <name>_orig instead of dropping them",
    )
    add_verbose_arg(out)
    return parser.parse_args(argv)


def _unique_names(paths):
    """Short, unique dataset names from file stems."""
    names = {}
    for p in paths:
        stem = Path(p).stem
        name, i = stem, 1
        while name in names:
            i += 1
            name = f"{stem}_{i}"
        names[name] = p
    return names


def _resolve_input(key, names):
    """Match a user-given file name / stem against the dataset names."""
    for name, path in names.items():
        if key in (name, path, Path(path).name, str(Path(path).resolve())):
            return name
    return None


MIN_FREE_WARN = 500


def _existing_report(report):
    """Print the existing free set of every input and pairwise agreement."""
    print("\nExisting R-free flags:")
    for name, f in report["files"].items():
        if f["column"] is None:
            problem = f.get("problem", "")
            why = "" if "no recognised" in problem else f" ({problem})"
            print(f"  {name:<24s} none{why}  dmin {f['dmin']:5.2f} A")
            continue
        print(
            f"  {name:<24s} {f['column']:<12s} {f['convention']:<18s} "
            f"dmin {f['dmin']:5.2f} A  free {f['n_free']:>7d} "
            f"({100 * f['n_free'] / max(f['n'], 1):5.2f} %)  excluded {f['n_excluded']}"
        )
        if f["n_free"] == 0:
            print("    Warning: no free reflections")
        if f["n_conflicting"]:
            print(
                f"    Warning: {f['n_conflicting']} reflections have symmetry "
                "equivalents with different flags in this file"
            )
    for (a, b), (n_common, n_bad) in report["pairs"].items():
        status = "agree" if n_bad == 0 else f"DISAGREE on {n_bad}"
        print(f"  {a} vs {b}: {n_common} common reflections, {status}")
    if report["consistent"]:
        print("  -> free sets are consistent")
    else:
        files = report["files"]
        missing = [n for n, f in files.items() if f["column"] is None]
        bad = [
            n
            for n, f in files.items()
            if f["column"] is not None and (f["n_free"] == 0 or f["n_conflicting"])
        ]
        if missing:
            why = f"no flags in {', '.join(missing)}"
        elif bad:
            why = f"invalid free set in {', '.join(bad)}"
        else:
            why = "flags disagree"
        print(f"  -> free sets are NOT consistent ({why})")


def _new_report(datasets, flags, info, ref_label, args):
    """Print the assigned free set: summary, warnings, one line per file."""
    pct = f"{100 * info['free_fraction']:.2f} %"
    print(
        f"\nFreeR_flag: {info['n_flags']} values, {pct} free "
        f"({info['fraction_source']}), to {info['dmin']:.2f} A"
    )
    if info["fraction_source"].startswith("max_free"):
        print(f"  reproduce with --free-fraction {info['free_fraction']:.6g}")
    if args.verbose > 1:
        print(f"  seed {info['seed']} ({info['seed_source']})")
    if "reference" in info:
        r = info["reference"]
        print(
            f"  inherited from {ref_label} ({r['column']}, {r['convention']}, "
            f"{r['dmin']:.2f} A): {info['n_inherited']} kept, {info['n_generated']} new"
        )
        n_beyond, n_gaps = (
            info["n_generated_beyond_reference"],
            info["n_gaps_in_reference"],
        )
        if n_beyond:
            print(
                f"  Warning: reference ends at {r['dmin']:.2f} A; {n_beyond} reflections "
                f"to {info['dmin']:.2f} A newly flagged ({pct} free, "
                f"seed from {info['seed_source']})"
            )
        if n_gaps:
            frac = n_gaps / max(n_gaps + info["n_inherited"], 1)
            print(
                ("  Warning: " if frac > 0.05 else "  ")
                + f"{n_gaps} reflections ({100 * frac:.1f} %) missing within the "
                "reference's range, newly flagged"
            )
        if r["n_inconsistent"]:
            print(
                f"  Warning: {r['n_inconsistent']} reference reflections have "
                "conflicting equivalents (free wins)"
            )
    if info["n_off_asu"]:
        print(
            f"  Warning: {info['n_off_asu']} reflections outside the complete ASU "
            "(systematic absences?), flagged by hash"
        )
    for name, ds in datasets.items():
        _flag_report(name, flags[name], ds, info, args.verbose)
    if len(datasets) > 1 and args.verbose > 1:
        common = set.intersection(
            *(set(_rfree_keys(ds).tolist()) for ds in datasets.values())
        )
        print(f"  {len(common)} unique reflections common to all inputs")


def _rfree_keys(ds):
    """Unique-ASU keys of every row of ``ds``."""
    from torchref.io import rfree

    return rfree.hkl_keys(rfree.asu_hkl(ds))


def _flag_report(name, flags, ds, info, verbose):
    from torchref.io import rfree

    n = len(flags)
    n_free = int((flags == 0).sum())
    n_excl = info["n_excluded"].get(name, 0)
    dmin = ds.compute_dHKL()["dHKL"].min()
    print(
        f"  {name:<24s} {n:>9d} rows  dmin {dmin:5.2f} A  free {n_free:>7d} "
        f"({100 * n_free / max(n - n_excl, 1):5.2f} %)"
        + (f"  excluded {n_excl} (kept as -1)" if n_excl else "")
    )
    if n_free < MIN_FREE_WARN:
        print(f"    Warning: only {n_free} free reflections; R-free will be noisy")
    if verbose > 1:
        dstar2 = 1.0 / ds.compute_dHKL()["dHKL"].to_numpy() ** 2
        bins = rfree.resolution_bins(dstar2, 10)
        for b in range(bins.max() + 1):
            sel = bins == b
            d_lo = 1 / np.sqrt(dstar2[sel].min())
            d_hi = 1 / np.sqrt(dstar2[sel].max())
            frac = (flags[sel] == 0).sum() / max((flags[sel] >= 0).sum(), 1)
            print(f"      {d_lo:6.2f} - {d_hi:5.2f} A   free {100 * frac:5.2f} %")


def _scale_datasets(flagged, args, device):
    """Jointly scale flagged datasets; returns name -> per-row amplitude factor."""
    import torch

    from torchref.cli._common import load_reflection_data
    from torchref.config import get_int_dtype
    from torchref.io import rfree
    from torchref.io.datasets.collection import DatasetCollection

    collection = DatasetCollection(verbose=args.verbose, device=device)
    with tempfile.TemporaryDirectory() as tmp:
        for name, ds in flagged.items():
            path = Path(tmp) / f"{name}.mtz"
            rfree.write_sf_file(ds, str(path))
            data = load_reflection_data(str(path), device=device, verbose=0)
            collection.add_dataset(name, data)
    collection.scale(nsteps=args.scale_nsteps, max_iter=args.scale_max_iter)

    scaler = collection.scaler

    def row_factors(key, ds, sg):
        # the scaler's anisotropy lives in the canonical-ASU setting
        hkl = torch.as_tensor(ds.get_hkls(), dtype=get_int_dtype())
        canonical, _, _, order = sg.canonicalize_hkl(hkl)
        with torch.no_grad():
            f_sorted = scaler(key, canonical).cpu().numpy()
        f = np.empty(len(hkl))
        f[order.cpu().numpy()] = f_sorted
        return f

    factors = {}
    for name, ds in flagged.items():
        sg = collection[name].spacegroup
        f = row_factors(name, ds, sg)
        if args.scale_reference is not None:
            f = f / row_factors(args.scale_reference, ds, sg)
        factors[name] = f
    return factors, collection.scaling_metrics


def main(argv=None):
    """Entry point for ``torchref.uniform-rfree``; returns the process exit code."""
    args = _parse_args(argv)
    from torchref.io import rfree

    names = _unique_names(args.files)
    for name, path in names.items():
        if not Path(path).is_file():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1

    # ---- read -------------------------------------------------------------
    datasets = {}
    for name, path in names.items():
        try:
            datasets[name] = rfree.read_sf_file(path, cif_block=args.cif_block)
        except Exception as exc:  # noqa: BLE001 - report and exit cleanly
            print(f"Error: cannot read {path}: {exc}", file=sys.stderr)
            return 1
        if args.verbose:
            ds = datasets[name]
            dmin = ds.compute_dHKL()["dHKL"].min()
            print(
                f"Read {path}: {len(ds)} rows, {ds.spacegroup.xhm()}, "
                f"cell {tuple(round(x, 3) for x in ds.cell.parameters)}, dmin {dmin:.2f} A"
            )

    problems = rfree.check_compatible(datasets, args.length_tol, args.angle_tol)
    if problems:
        for p in problems:
            print(("Warning: " if args.force else "Error: ") + p, file=sys.stderr)
        if not args.force:
            print("Use --force to proceed anyway.", file=sys.stderr)
            return 1

    # ---- existing flags -------------------------------------------------
    # With --reference auto the reference is one of the inputs, so the named
    # column is the one to look for in them; an explicit reference file's column
    # says nothing about the inputs' own flags.
    input_column = args.reference_column if args.reference == "auto" else None
    report = rfree.compare_free_sets(datasets, input_column)
    if args.check or args.verbose > 1 or (args.verbose and not report["consistent"]):
        _existing_report(report)
    elif args.verbose:
        print(f"\nExisting free sets are consistent across all {len(datasets)} inputs.")
    if args.check:
        return 0 if report["consistent"] else 2
    if args.outdir is None:
        print("Error: -o/--outdir is required (or use --check)", file=sys.stderr)
        return 1

    reference, ref_label = None, None
    with_flags = [n for n, f in report["files"].items() if f["column"] is not None]
    usable = [
        n
        for n in with_flags
        if report["files"][n]["n_free"] > 0 and report["files"][n]["n_conflicting"] == 0
    ]
    if args.fresh:
        if with_flags and args.verbose:
            print(
                f"--fresh: replacing the free set(s) of {', '.join(with_flags)}; "
                "R-free of models refined against them becomes biased."
            )
    elif args.reference == "auto":
        for name in [n for n in with_flags if n not in usable]:
            print(
                f"Warning: not inheriting from {name!r}: its free set is empty or "
                "has conflicting symmetry equivalents",
                file=sys.stderr,
            )
        if usable:
            ref_label = usable[0]
            reference = datasets[ref_label]
            if not report["consistent"] and len(with_flags) > 1:
                print(
                    f"Warning: existing free sets disagree; inheriting from {ref_label!r} "
                    "(the first input with flags). Pass --reference to choose another.",
                    file=sys.stderr,
                )
        elif args.verbose:
            print("No input carries a usable free set; generating a new free set.")
    else:
        ref_label = _resolve_input(args.reference, names)
        try:
            reference = (
                datasets[ref_label]
                if ref_label is not None
                else rfree.read_sf_file(args.reference, cif_block=args.cif_block)
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Error: cannot read reference {args.reference}: {exc}", file=sys.stderr
            )
            return 1
        ref_label = ref_label or args.reference
        if rfree.flag_column(reference, args.reference_column) is None:
            print(
                f"Error: reference {args.reference} has no R-free column",
                file=sys.stderr,
            )
            return 1
        problems = rfree.check_compatible(
            {"inputs": next(iter(datasets.values())), "reference": reference},
            args.length_tol,
            args.angle_tol,
        )
        if problems and not args.force:
            for p in problems:
                print("Error: " + p, file=sys.stderr)
            return 1

    sizing = [
        option
        for option, value in [
            ("--free-fraction", args.free_fraction),
            ("--max-free", args.max_free),
        ]
        if value is not None
    ]
    if reference is not None and sizing:
        options = " and ".join(sizing)
        print(
            f"Error: cannot combine {options} with the free set inherited from "
            f"{ref_label!r}, which is extended at its own fraction. Drop {options} "
            "to extend it, or add --fresh to generate a new set instead.",
            file=sys.stderr,
        )
        return 1

    # ---- flags ------------------------------------------------------------
    try:
        flags, info = rfree.uniform_rfree(
            datasets,
            free_fraction=args.free_fraction,
            shell_size=args.shell_size,
            seed=args.seed,
            dmin=args.dmin,
            reference=reference,
            reference_column=args.reference_column,
            max_free=args.max_free,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        _new_report(datasets, flags, info, ref_label, args)

    flagged = {
        name: rfree.apply_flags(ds, flags[name], keep_old=args.keep_old_flags)
        for name, ds in datasets.items()
    }

    # ---- optional scaling -------------------------------------------------
    if args.scale:
        if len(flagged) < 2:
            print("Error: --scale needs at least two inputs", file=sys.stderr)
            return 1
        if args.scale_reference is not None:
            key = _resolve_input(args.scale_reference, names)
            if key is None:
                print(
                    f"Error: --scale-reference {args.scale_reference!r} is not an input",
                    file=sys.stderr,
                )
                return 1
            args.scale_reference = key
        from torchref.config import normalize_device

        device = normalize_device(args.device)
        if args.verbose:
            print(f"\nScaling {len(flagged)} datasets jointly on {device} ...")
        try:
            factors, metrics = _scale_datasets(flagged, args, device)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: scaling failed: {exc}", file=sys.stderr)
            return 1
        for name in flagged:
            flagged[name], cols = rfree.scale_columns(flagged[name], factors[name])
            if args.verbose:
                f = factors[name]
                print(
                    f"  {name:<24s} factor median {np.median(f):.4f} "
                    f"[{f.min():.4f}, {f.max():.4f}]  columns: {', '.join(cols) or '-'}"
                )
        if args.verbose > 1 and metrics:
            print(f"  metrics: {metrics}")

    # ---- write ------------------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, ds in flagged.items():
        for fmt in args.format:
            path = outdir / f"{name}{args.suffix}.{fmt}"
            try:
                rfree.write_sf_file(ds, str(path))
            except Exception as exc:  # noqa: BLE001
                print(f"Error: cannot write {path}: {exc}", file=sys.stderr)
                return 1
            if args.verbose:
                print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
