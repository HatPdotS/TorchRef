#!/bin/bash
# Stage C1, third pass. Hypothesis: with the back-half accumulation restored to
# double, the whole of C1 is numerically neutral -- so the fingerprints should be
# bit-identical to d1244c45, and the 1e-4 score shift seen in pass 2 was entirely
# the narrowed accumulation.
#SBATCH --job-name=stagec3
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
NEW=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
OLD=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/_stagea_baseline
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
SCRIPT=$NEW/alignment_lab/analysis/frf_fingerprint.py
OUT=$NEW/alignment_lab/slurm
export TORCHREF_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"

cd "$NEW"; export PYTHONPATH="$NEW"
LOG=$OUT/stagec3_tests_$SLURM_JOB_ID.log
"$PY" -m pytest tests/unit/alignment tests/unit/frf_separate tests/unit/model \
      tests/unit/test_imports_smoke.py -q > "$LOG" 2>&1
rc=$?
echo "=== TESTS rc=$rc ==="
tail -12 "$LOG"

status=0
for pdb in 3K7M 1DAW; do
  for cap in 64 100; do
    cd "$NEW"
    PYTHONPATH="$NEW" "$PY" -u "$SCRIPT" --pdb "$pdb" --lmax-cap "$cap" \
      > "$OUT/c3_NEW_${pdb}_${cap}.txt" 2>/dev/null
    a=$OUT/c2_OLD_${pdb}_${cap}.txt          # baseline captured in pass 2
    b=$OUT/c3_NEW_${pdb}_${cap}.txt
    if diff -q <(grep '^FP ' "$a") <(grep '^FP ' "$b") >/dev/null; then
      echo "IDENTICAL  $pdb cap$cap  ($(grep -c '^FP ' "$b") peaks)"
    else
      nd=$(diff <(grep '^FP ' "$a") <(grep '^FP ' "$b") | grep -c '^[<>]')
      echo "DIFFERS    $pdb cap$cap  ($nd lines)"
      diff <(grep '^FP ' "$a") <(grep '^FP ' "$b") | head -4
      status=1
    fi
  done
done

echo "=== timing, 4 threads for comparability with the 1.08s note ==="
export TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -u -m alignment_lab.diagnostics.frf_benchmark --pdb 3K7M --arms cap64 --trials 2 2>&1 \
  | grep -vE "Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|warn|^ *$"
echo "stagec3_status=$status tests_rc=$rc"
