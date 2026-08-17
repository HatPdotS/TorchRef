#!/usr/bin/env python3
"""Submit the exF4 single-core runtime benchmark: 3 programs x 1 core x n=715 structs.

This reproduces Figure 2c (the wall-clock box plot, plot_figure_af.py:150-193) but with
every program pinned to a SINGLE CPU core, over the full conserved AlphaFold-start set
(``codes_conserved.txt``, n=715). The main Figure-2 benchmark ran all three programs at
4 cores; this re-levels them onto a clean per-core single-threaded comparison.

For every (program, code) cell, write+sbatch a self-contained job that refines the
Phaser-placed AlphaFold model for 10 macro cycles with the program pinned to 1 thread,
and records the wall-clock runtime to ``timing.txt`` via a uniform shell timer
(``date +%s.%N`` around the program call) so all three engines are measured identically.
Per-cell output lands in ``runs/{program}/{code}/n1/``.

Reuses figure2_alphafold_start paths/helpers via ``run_af_pipeline`` (PLACED, _mtz,
PYTHON, REFINE_SCRIPT, CCP4_SETUP, _sbatch). Submits ALL cells fresh (no resume-skip):
an existing ``timing.txt`` is overwritten so the whole set is timed under identical
conditions. Memory is 8G to match the main Figure-2 benchmark allocation.

Usage
-----
    ./.dev/bin/python paper/extended_figures/exF4/submit_singlecore.py --dry-run --limit 1
    ./.dev/bin/python paper/extended_figures/exF4/submit_singlecore.py --programs torchref
    ./.dev/bin/python paper/extended_figures/exF4/submit_singlecore.py            # all 2145 cells
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AF = HERE.parent.parent / "figure2_alphafold_start"
sys.path.insert(0, str(AF))
import run_af_pipeline as P  # noqa: E402

RUNS = HERE / "runs"
CODES_TXT = HERE / "codes_conserved.txt"
PROGRAMS = ["torchref", "phenix", "refmac"]
NCORES = 1                  # single core: the whole point of this figure
LABEL = "n1"
MEM = "8G"                  # match the main Figure-2 benchmark allocation

# Per-program SLURM partition + walltime. Generous so a 1-core run never times out
# (the timing IS the measurement). The unbounded n=715 set may contain structures larger
# than the exF4 size cap; aggregate_singlecore.py reports any rc!=0 / not-run cell so a
# straggler can be resubmitted to `week` if a giant structure overruns.
PARTITION = {"torchref": "day", "phenix": "day", "refmac": "hour"}
WALLTIME = {"torchref": "24:00:00", "phenix": "24:00:00", "refmac": "01:00:00"}

# One CPU model for all three programs, because this figure compares wall-clock ACROSS
# programs. EPYC 9335 is 2.2-2.7x faster than EPYC 7453 on identical code, and the
# partitions above are not the same node pool by default -- refmac on `hour` would draw a
# different hardware mix than torchref on `day`, which is a systematic per-engine bias, not
# noise that n=715 averages away. The same 29 epyc9335 nodes belong to hour/day/week, so
# constraining does not change which partition each program uses. Single-core jobs cannot be
# --exclusive at this scale, so co-tenancy noise remains -- but it now falls on all three
# engines from the same pool. Verify before changing:
#   sinfo -h -p day -o "%n %f" | grep -c <model>
CPU_MODEL = "cpu_epyc9335"


def cell_dir(program, code):
    return RUNS / program / code / LABEL


def _header(program, code, outdir):
    return f"""#!/bin/bash
#SBATCH --job-name=exF4sc_{program}_{code}
#SBATCH --partition={PARTITION[program]}
#SBATCH --constraint={CPU_MODEL}
#SBATCH --cpus-per-task={NCORES}
#SBATCH --time={WALLTIME[program]}
#SBATCH --mem={MEM}
#SBATCH --output={outdir / 'out.log'}
#SBATCH --error={outdir / 'out.log'}
"""


# Uniform timer: wraps the program call (a plain command, NOT a pipe, so $? is the
# program's own exit code) and writes "wall_s <float>\nrc <int>" to timing.txt. No
# `set -e`: sourcing the phenix / ccp4 env can return nonzero internally, and the rc field
# lets aggregation drop failed runs rather than record a bogus time.
def _timed(body):
    return (f"T0=$(date +%s.%N)\n{body}\nRC=$?\nT1=$(date +%s.%N)\n"
            f'awk "BEGIN{{printf \\"wall_s %.3f\\nrc %d\\n\\", $T1-$T0, $RC}}"'
            f' > "$OUTDIR/timing.txt"\n')


def script_torchref(code, pdb, mtz, outdir):
    body = (f"{P.PYTHON} -u {P.REFINE_SCRIPT} \\\n"
            f"    -m {pdb} -sf {mtz} -o $OUTDIR \\\n"
            f"    -n 10 --mode separate --xray-mode ml --device cpu")
    return _header("torchref", code, outdir) + f"""
OUTDIR={outdir}
export TORCHREF_NUM_THREADS={NCORES}
{_timed(body)}"""


def script_phenix(code, pdb, mtz, outdir):
    # AF models are protein-only -> no elbow/ligand restraints needed. Plain refine in a
    # private TMPDIR; phenix stdout flows to the SLURM out.log (no tee, so the timed $? is
    # phenix's). CRYST1 "None" is sed-fixed (mirrors phenix_refine.sh) so phenix never
    # chokes on the Z field.
    #
    # IMPORTANT: flags MUST mirror the main Fig-2 benchmark (phenix_refine.sh, phenix_norb
    # arm) so this is a like-for-like re-timing at 1 core. In particular the production run
    # DISABLES automatic target-weight optimization (optimize_xyz/adp_weight=false) and
    # pins a fixed no-rigid-body strategy; phenix DEFAULTS instead run an expensive
    # per-macrocycle weight grid-search (~2.4x slower) that has nothing to do with cores.
    body = (f'phenix.refine --overwrite --quiet \\\n'
            f'    "$WORK/input.pdb" {mtz} \\\n'
            f"    output.prefix=ref \\\n"
            f"    refinement.main.number_of_macro_cycles=10 \\\n"
            f"    refinement.main.nproc={NCORES} \\\n"
            f"    refinement.refine.strategy=individual_sites+individual_adp+occupancies \\\n"
            f"    refinement.main.simulated_annealing=false \\\n"
            f"    refinement.target_weights.optimize_xyz_weight=false \\\n"
            f"    refinement.target_weights.optimize_adp_weight=false \\\n"
            f"    refinement.main.bulk_solvent_and_scale=true \\\n"
            f"    refinement.main.ordered_solvent=false \\\n"
            f"    refinement.ordered_solvent.mode=every_macro_cycle \\\n"
            f"    refinement.pdb_interpretation.ramachandran_plot_restraints.enabled=false \\\n"
            f"    write_def_file=false write_eff_file=false write_geo_file=false")
    return _header("phenix", code, outdir) + f"""
OUTDIR={outdir}
source {P.PHENIX_ENV}
WORK=$(mktemp -d /tmp/exF4sc_phenix_{code}_XXXX)
cd "$WORK"
if grep -q "^CRYST1.*None" {pdb}; then
    sed 's/\\(CRYST1.*P [0-9]* *[0-9]* *[0-9]* *\\)None/\\1   12/' {pdb} > "$WORK/input.pdb"
else
    cp {pdb} "$WORK/input.pdb"
fi
{_timed(body)}cd / && rm -rf "$WORK"
"""


def script_refmac(code, pdb, mtz, outdir):
    body = ("refmac5 HKLIN input.mtz HKLOUT output.mtz "
            "XYZIN input.pdb XYZOUT output.pdb > $OUTDIR/refmac.log 2>&1 << EOF\n"
            "NCYCLES 10\nMAKE HYDR NO\nEND\nEOF")
    return _header("refmac", code, outdir) + f"""
OUTDIR={outdir}
source {P.CCP4_SETUP}
export OMP_NUM_THREADS={NCORES}
WORK=$(mktemp -d /tmp/exF4sc_refmac_{code}_XXXX)
export CCP4_SCR=$WORK
cd "$WORK"
cp {pdb} input.pdb
cp {mtz} input.mtz
{_timed(body)}cd / && rm -rf "$WORK"
"""


BUILDERS = {"torchref": script_torchref, "phenix": script_phenix,
            "refmac": script_refmac}


def load_codes():
    return [c.strip() for c in CODES_TXT.read_text().split() if c.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--programs", nargs="+", default=PROGRAMS, choices=PROGRAMS)
    ap.add_argument("--codes", nargs="+", default=None,
                    help="Override codes_conserved.txt with an explicit list.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N structures (smoke test).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    codes = args.codes if args.codes else load_codes()
    if args.limit:
        codes = codes[:args.limit]

    n_cells = len(args.programs) * len(codes)
    print(f"programs={args.programs}  cores={NCORES}  mem={MEM}  structures={len(codes)}")
    print(f"cells = {n_cells}  (all submitted fresh, existing timings overwritten)\n")

    submitted = missing = 0
    first = True
    # STRUCTURE-major, not program-major. This benchmark compares wall-clock ACROSS
    # programs, and a full sweep spans hours during which cluster load changes. Submitting
    # all of one program before the next times each engine in its own load window, so
    # engine identity ends up confounded with time -- measured 2026-08-04: on a
    # program-major sweep two programs whose code had not changed moved by -43% and +31%
    # against the previous sweep, with per-structure IQRs spanning ~2x. Emitting a
    # structure's three engines adjacently puts them under near-identical conditions, which
    # is what makes the per-structure three-way comparison meaningful.
    for code in codes:
        for program in args.programs:
            pdb, mtz = P.PLACED / f"{code}_af.pdb", P._mtz(code)
            if not pdb.exists() or not mtz.exists():
                missing += 1
                continue
            outdir = cell_dir(program, code)
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
            script = BUILDERS[program](code, pdb, mtz, outdir)
            if first and args.dry_run:
                print(f"── example sbatch ({program}, {code}, {LABEL}) ──")
                print(script)
                print("──────────────────────────────────────────────")
                first = False
            tmp = RUNS / program / "tmp_scripts"
            if P._sbatch(script, tmp / f"{code}_{LABEL}.sh", args.dry_run) or args.dry_run:
                submitted += 1

    print(f"\nsubmitted={submitted}, missing(no pdb/mtz)={missing}")


if __name__ == "__main__":
    main()
