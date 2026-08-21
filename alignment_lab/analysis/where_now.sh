#!/bin/bash
# Full stage breakdown of one rotation search, now that the SH expansion is no
# longer the dominant term.
#SBATCH --job-name=frf_where
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
for pdb in 3K7M 1DAW; do
  "$PY" -u -m alignment_lab.diagnostics.frf_benchmark \
      --pdb "$pdb" --arms cap100,cap64 --trials 2 2>&1 \
    | grep -vE "Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|warn|^ *$"
done
echo "exit_code=$?"
