#!/bin/bash
# Paired timing: baseline worktree vs this one, same node, same job, INTERLEAVED
# and structure-major. A cross-job comparison against a remembered number is not
# a measurement -- node and contention differ.
#SBATCH --job-name=stagec_time
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
export TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "OLD=$(cd $OLD && git rev-parse --short HEAD)  NEW=working tree"

for round in 1 2 3; do
  for pdb in 3K7M 1DAW; do
    for tree in OLD NEW; do
      eval "root=\$$tree"
      cd "$root"
      line=$(PYTHONPATH="$root" "$PY" -u -m alignment_lab.diagnostics.frf_benchmark \
               --pdb "$pdb" --arms cap64 --trials 2 2>/dev/null \
             | grep -E "^ *cap64" | tail -1)
      echo "round$round $pdb $tree  $line"
    done
  done
done
