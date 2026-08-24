#!/bin/bash
# Where the time goes NOW. cap64 only -- that is what ships, and medianing it
# with cap100 inverted the stage ranking once already.
#SBATCH --job-name=wherenow
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
for pdb in 3K7M 1AK5 1DAW; do
  echo "############ $pdb ############"
  "$PY" -u -m alignment_lab.diagnostics.frf_benchmark --pdb "$pdb" --arms cap64 --trials 3 2>&1 \
    | grep -vE "Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|warn|^ *$"
done
