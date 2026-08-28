#!/bin/bash
# Same script, same data, two trees: post-merge and the commit before it.
# Single-threaded because the FRF peak list only reproduces bit-for-bit at one
# thread -- ~5e-8 of score noise reorders peaks otherwise, and this gate is
# about bit-identity.
#SBATCH --job-name=mergeid
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-3
set -uo pipefail
POST=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PRE=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/premerge_check
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
# 1DAW is C2 (monoclinic) and 2DQ6 is P3121: on an orthogonal cell the
# fractionalisation matrix is diagonal, so a transposed edge-vector convention
# in the new voxel_size would not show. On these it would.
PDBS=(1DAW 2DQ6 3K7M 4BX9)
PDB=${PDBS[$SLURM_ARRAY_TASK_ID]}
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export TORCHREF_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for TREE in "$PRE" "$POST"; do
  TAG=$([ "$TREE" = "$PRE" ] && echo pre || echo post)
  cd "$TREE"
  PYTHONPATH="$TREE" "$PY" -u alignment_lab/analysis/merge_numeric_identity.py \
    --pdb "$PDB" --tag "$TAG" 2>/dev/null | grep '^OUT '
done
