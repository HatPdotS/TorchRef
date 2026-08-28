#!/bin/bash
# Post-merge gate. Two questions, in order:
#   1. does the merged tree still work at all (imports, full unit suite);
#   2. did dev's changes move the FRF's answers? The fingerprints are compared
#      against 775576bc, our last pre-merge commit, so any difference is dev's
#      or the merge resolution's -- not ours.
#SBATCH --job-name=mergegate
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
NEW=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
OUT=$NEW/alignment_lab/slurm
cd "$NEW"; export PYTHONPATH="$NEW"
export TORCHREF_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname)"

echo "=== import smoke ==="
"$PY" -c "import torchref; import torchref.experimental.alignment as a; print('  torchref + alignment import OK')" 2>&1 | tail -5

LOG=$OUT/mergegate_tests_$SLURM_JOB_ID.log
"$PY" -m pytest tests/unit -q > "$LOG" 2>&1
rc=$?
echo "=== FULL UNIT SUITE rc=$rc ==="
tail -20 "$LOG"

echo "=== does a copy still have a usable, correct partition? ==="
"$PY" -m pytest tests/unit/model/test_copy.py -q 2>&1 | tail -6

echo "=== FRF fingerprints vs 775576bc (pre-merge) ==="
for pdb in 3K7M 1DAW; do
  "$PY" -u "$NEW/alignment_lab/analysis/frf_fingerprint.py" --pdb "$pdb" --lmax-cap 64 \
    > "$OUT/merge_NEW_${pdb}.txt" 2>/dev/null || echo "  RUN FAILED $pdb"
  ref=$OUT/cap_NEW_${pdb}.txt      # captured at 775576bc by capability_gate
  if [ -s "$ref" ] && [ -s "$OUT/merge_NEW_${pdb}.txt" ]; then
    if diff -q <(grep '^FP ' "$ref") <(grep '^FP ' "$OUT/merge_NEW_${pdb}.txt") >/dev/null; then
      echo "  IDENTICAL $pdb ($(grep -c '^FP ' "$OUT/merge_NEW_${pdb}.txt") peaks)"
    else
      nd=$(diff <(grep '^FP ' "$ref") <(grep '^FP ' "$OUT/merge_NEW_${pdb}.txt") | grep -c '^[<>]')
      echo "  DIFFERS $pdb ($nd lines)"
      diff <(grep '^FP ' "$ref") <(grep '^FP ' "$OUT/merge_NEW_${pdb}.txt") | head -4
    fi
  else
    echo "  no comparable reference for $pdb"
  fi
done
echo "mergegate_rc=$rc"
