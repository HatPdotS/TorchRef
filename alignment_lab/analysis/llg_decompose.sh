#!/bin/bash
#SBATCH --job-name=llgdec
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname)"
for trial in 0 1; do
  for flag in "" "--full-prep"; do
    "$PY" -u alignment_lab/analysis/llg_decompose.py --pdb 6G9X --trial $trial $flag 2>&1 \
      | grep -vE "UserWarning|FutureWarning|^ *from |^ *warnings\.|Loaded|LINK|Wilson outlier|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization"
    echo
  done
done
echo "rc=$?"
