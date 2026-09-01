#!/bin/bash
# Where does an alignment's wall clock actually go?
#
# Expectation from the component measurements: FRF 1-3 s, and the translation
# stage bounded by one structure-factor call per candidate at well under 100 ms,
# so 25 candidates should be a couple of seconds. That predicts ~5 s and the
# panel medians are 34 s by R and 75-93 s by likelihood. The per-stage timer has
# been in the pipeline the whole time; this reads it.
#SBATCH --job-name=stageprof
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=day
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for ARM in analytic_r llg; do
  for P in 1DAW 2DQ6; do
    echo "########## $P  rank_by=$ARM ##########"
    "$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb $P --trial 0 \
      --arms $ARM --verbose 2 2>/dev/null \
      | sed -n '/^stage /,/^TOTAL/p;/^ROW /p'
  done
done
echo DONE
