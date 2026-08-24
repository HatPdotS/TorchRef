#!/bin/bash
# Stage D gate: the obs chain now runs on the unique set and only the geometry
# unrolls, with one shared shell assignment. This CHANGES numbers, so quantify
# how much and confirm truth is still where the pipeline can reach it. Also
# measure what it bought, paired and interleaved on one node.
#SBATCH --job-name=staged2
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
NEW=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
OLD=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/_stagea_baseline
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
OUT=$NEW/alignment_lab/slurm
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname)  OLD=$(cd $OLD && git rev-parse --short HEAD)  NEW=working tree"

cd "$NEW"; export PYTHONPATH="$NEW"
export TORCHREF_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=$OUT/staged_tests_$SLURM_JOB_ID.log
"$PY" -m pytest tests/unit/alignment tests/unit/frf_separate tests/unit/model \
      tests/unit/test_imports_smoke.py -q > "$LOG" 2>&1
rc=$?
echo "=== TESTS rc=$rc ==="
tail -14 "$LOG"

echo "=== how far did the peak lists move (single-threaded) ==="
for pdb in 3K7M 1DAW; do
  for cap in 64 100; do
    for tree in OLD NEW; do
      eval "root=\$$tree"
      cd "$root"
      PYTHONPATH="$root" "$PY" -u "$NEW/alignment_lab/analysis/frf_fingerprint.py" \
        --pdb "$pdb" --lmax-cap "$cap" > "$OUT/d_${tree}_${pdb}_${cap}.txt" 2>/dev/null \
        || echo "RUN FAILED $tree $pdb $cap"
    done
    "$PY" - "$OUT/d_OLD_${pdb}_${cap}.txt" "$OUT/d_NEW_${pdb}_${cap}.txt" "$pdb" "$cap" <<'PYEOF'
import sys
rp, np_, pdb, cap = sys.argv[1:5]
def load(p):
    return [tuple(map(float, l.split()[2:7])) for l in open(p) if l.startswith("FP ")]
a, b = load(rp), load(np_)
if not a or not b:
    print(f"  {pdb} cap{cap}: EMPTY {len(a)} vs {len(b)}"); sys.exit()
n = min(len(a), len(b))
slot = sum(1 for i in range(n) if a[i][:3] == b[i][:3])
# Is the same peak SET recovered, regardless of order?
sa, sb = {r[:3] for r in a}, {r[:3] for r in b}
rel = sorted(abs(b[i][3]-a[i][3])/max(abs(a[i][3]),1e-30) for i in range(n))
print(f"  {pdb} cap{cap}: {slot}/{n} slots identical | set overlap "
      f"{len(sa & sb)}/{len(sa)} | |dscore|/score p50={rel[n//2]:.2e} "
      f"p99={rel[int(0.99*n)]:.2e}")
print(f"      top-1 old {tuple(round(v,6) for v in a[0][:3])} z={a[0][4]:.4f}"
      f"  new {tuple(round(v,6) for v in b[0][:3])} z={b[0][4]:.4f}")
PYEOF
  done
done

echo "=== paired timing, 4 threads, 3 rounds ==="
export TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for round in 1 2 3; do
  for pdb in 3K7M 1DAW; do
    for tree in OLD NEW; do
      eval "root=\$$tree"; cd "$root"
      line=$(PYTHONPATH="$root" "$PY" -u -m alignment_lab.diagnostics.frf_benchmark \
               --pdb "$pdb" --arms cap64 --trials 2 2>/dev/null | grep -E "^ *cap64" | tail -1)
      echo "round$round $pdb $tree  $line"
    done
  done
done
echo "staged_tests_rc=$rc"
