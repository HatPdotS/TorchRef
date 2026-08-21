# alignment_lab

Harness for the FRF rotation-function work: shared primitives, one file per
experiment, one aggregator.

```
lab/          shared library — import from here, do not re-derive
diagnostics/  one experiment per file, each with --out-csv
analysis/     aggregator + SLURM array template
tests/        self-tests for the primitives
runs/         CSVs and Phaser working dirs   (gitignored)
slurm/        scheduler logs                 (gitignored)
```

## Why a library

The scripts this replaces carried ~37 copies of the rotation generator, ~30 of
the benchmark list, ~28 of the CSV writer and ~20 of the rank-of-truth
computation — and several disagreed with each other. Two of those divergences
changed results silently:

- **Two rotation generators.** One omitted the `sign(diag(R))` QR correction, so
  the same seed produced a *different* rotation. Results from the two families
  were never comparable. `lab.truth.random_rotation` is the corrected form and
  `tests/test_lab.py` pins it against the other variant.
- **Four orbit conventions.** Rank-of-truth was computed with the symmetry
  operators applied on either side, in either the fractional or Cartesian frame.
  The choice changes the rank, so `orbit_rank` takes it as an explicit argument
  and every result row records it.

## Running

```bash
PY=.dev/bin/python                      # or another worktree's interpreter
PYTHONPATH=. $PY alignment_lab/diagnostics/ghost_origin.py --pdb 3K7M --trial 0
PYTHONPATH=. $PY -m pytest alignment_lab/tests -q

sbatch --array=0-29 --partition=hour --time=00:55:00 --cpus-per-task=4 \
       --mem=32G alignment_lab/analysis/array_template.sh ghost_origin
PYTHONPATH=. $PY alignment_lab/analysis/aggregate.py \
       'alignment_lab/runs/ghost_origin_*/*.csv' --compare obs_mode
```

## Reading a result

Seed-to-seed truth-rank spread at `lmax_cap=64` is **±4–6** (1AK5 has been seen
at 9, 11 and 17 for one configuration). Below ~10 trials nothing is
interpretable; three findings that looked strong at n≤7 evaporated at full n.
`aggregate.py` therefore reports paired per-trial differences with the
per-trial values visible, never a bare median, and prints whatever it dropped.

## Traps worth knowing

- `model.spacegroup = SpaceGroup("P 1")` is a **silent no-op** — `SpaceGroup` is
  an `nn.Module`, so `nn.Module.__setattr__` intercepts the assignment and the
  property setter never runs. Assign the **name string**. `ghost_origin.py`'s P1
  arm depends on this.
- `Model.rotate` / `.translate` mutate in place and return `self`. Copy first if
  you still need the original — `rotated_case` does.
- Phaser **exits 0 on fatal input errors**; an empty peak list is the real
  signal. Its keyword file needs absolute paths.
- `PEAKS ROT SELECT ALL` returns ~80–92k densely spaced samples (median nearest
  neighbour under 1°), so "the closest sample is within a degree" means nothing
  by itself.

## Known gap

`bench_stages.py` currently attributes time only to `phaser_rotation_search`
(~81–85%) and `dense_calc_via_box` (~14–17%). The inner Bessel/Wigner/peak
stages register **0 calls** — the separated engine does not route through those
module-level symbols, so wrapping them there intercepts nothing. They are
printed with their zero counts rather than omitted, because an absent row reads
as a free stage. Getting the inner breakdown needs different instrumentation
points.
