#!/bin/bash
# Stage A gate: peak-list fingerprints must be bit-identical between the
# baseline worktree (d1244c45) and this one. SINGLE-THREADED -- at the default
# thread count the float32 SF reduction reorders ~12 of 500 peaks on 3K7M.
#SBATCH --job-name=stagea_fp
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

status=0
for pdb in 3K7M 1DAW; do
  for cap in 64 100; do
    for tree in OLD NEW; do
      eval "root=\$$tree"
      cd "$root"
      PYTHONPATH="$root" "$PY" -u "$SCRIPT" --pdb "$pdb" --lmax-cap "$cap" \
        > "$OUT/fp_${tree}_${pdb}_${cap}.txt" 2> "$OUT/fp_${tree}_${pdb}_${cap}.err"
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "RUN FAILED tree=$tree pdb=$pdb cap=$cap rc=$rc"
        tail -6 "$OUT/fp_${tree}_${pdb}_${cap}.err"
        status=1
      fi
    done
    a="$OUT/fp_OLD_${pdb}_${cap}.txt"; b="$OUT/fp_NEW_${pdb}_${cap}.txt"
    if [ -s "$a" ] && [ -s "$b" ]; then
      if diff -q <(grep -v '^#' "$a") <(grep -v '^#' "$b") >/dev/null; then
        echo "IDENTICAL  $pdb cap$cap  ($(grep -vc '^#' "$a") peaks)"
      else
        nd=$(diff <(grep -v '^#' "$a") <(grep -v '^#' "$b") | grep -c '^[<>]')
        echo "DIFFERS    $pdb cap$cap  ($nd differing lines)"
        diff <(grep -v '^#' "$a") <(grep -v '^#' "$b") | head -8
        status=1
      fi
    else
      echo "MISSING    $pdb cap$cap"; status=1
    fi
  done
done
echo "fingerprint_status=$status"
