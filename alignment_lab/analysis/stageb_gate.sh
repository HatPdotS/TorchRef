#!/bin/bash
#SBATCH --job-name=stageb
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
NEW=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$NEW"
export PYTHONPATH="$NEW" TORCHREF_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"

LOG=alignment_lab/slurm/stageb_bessel_$SLURM_JOB_ID.log
"$PY" -m pytest tests/unit/frf_separate/test_bessel_rescale.py -q > "$LOG" 2>&1
rc_b=$?
echo "=== BESSEL TESTS rc=$rc_b ==="
tail -20 "$LOG"

# End-to-end: the rescaled ladder must leave the peak lists untouched. Compare
# against the fingerprints captured for the Stage A gate (same tree, pre-Stage-B).
OUT=$NEW/alignment_lab/slurm
status=0
for pdb in 3K7M 1DAW; do
  for cap in 64 100; do
    "$PY" -u "$NEW/alignment_lab/analysis/frf_fingerprint.py" --pdb "$pdb" \
      --lmax-cap "$cap" > "$OUT/fp_STAGEB_${pdb}_${cap}.txt" 2>"$OUT/fp_STAGEB_${pdb}_${cap}.err"
    rc=$?
    ref="$OUT/fp_NEW_${pdb}_${cap}.txt"
    new="$OUT/fp_STAGEB_${pdb}_${cap}.txt"
    if [ $rc -ne 0 ]; then
      echo "RUN FAILED $pdb cap$cap rc=$rc"; tail -5 "$OUT/fp_STAGEB_${pdb}_${cap}.err"; status=1; continue
    fi
    if [ ! -s "$ref" ]; then echo "NO STAGE-A REFERENCE for $pdb cap$cap"; status=1; continue; fi
    if diff -q <(grep -v '^#' "$ref") <(grep -v '^#' "$new") >/dev/null; then
      echo "IDENTICAL  $pdb cap$cap  ($(grep -vc '^#' "$new") peaks)"
    else
      nd=$(diff <(grep -v '^#' "$ref") <(grep -v '^#' "$new") | grep -c '^[<>]')
      echo "DIFFERS    $pdb cap$cap  ($nd lines)"
      diff <(grep -v '^#' "$ref") <(grep -v '^#' "$new") | head -6
      status=1
    fi
  done
done
echo "stageb_status=$status rc_bessel=$rc_b"
