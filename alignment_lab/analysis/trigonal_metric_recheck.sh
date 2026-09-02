#!/bin/bash
# Were 2DQ6's "failures" symmetry mates the success metric could not recognise?
#
# residual_rotation_deg compared a Cartesian Kabsch rotation against the
# FRACTIONAL symmetry matrices. In P3(1)21 two of the six mates of a correct
# solution then read as 30.00 and 21.09 deg; 2DQ6's failing residuals were
# 28.5-31.0 and 19.5-21.8. This re-scores the same seeds with Cartesian mates
# and a translation check modulo allowed origin shifts. Controls: 3GR5 (P6(5)22,
# four of twelve mates affected), 6G9X t4 corr (a genuine 55.9 deg miss, must
# stay a miss) and 1DAW t0 (monoclinic, must be unchanged at 1.519).
#SBATCH --job-name=trig
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-3
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
run() { "$PY" -u alignment_lab/diagnostics/pose_recovery.py "$@" 2>/dev/null | grep -E '^ROW |SOLN|^===' ; }
case $SLURM_ARRAY_TASK_ID in
  0) for T in $(seq 0 9); do run --pdb 2DQ6 --trial $T --arms llg; done ;;
  1) for T in $(seq 0 9); do run --pdb 2DQ6 --trial $T --arms analytic_r; done ;;
  2) for T in $(seq 0 9); do run --pdb 2DQ6 --trial $T --arms corr; done ;;
  3) for T in 0 1 2; do run --pdb 3GR5 --trial $T --arms llg,analytic_r,corr; done
     run --pdb 6G9X --trial 4 --arms corr
     run --pdb 1DAW --trial 0 --arms llg
     run --pdb 2DQ6 --trial 3 --arms llg --verbose 2 ;;
esac
echo DONE
