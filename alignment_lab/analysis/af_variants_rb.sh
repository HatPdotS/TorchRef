#!/bin/bash
# What in our assembled file, other than the pose, costs R-free? Refine (rigid
# body first) the placement with Phaser's mean B shift, with chains split at
# breaks, and with both.
#SBATCH --job-name=afvar
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-14
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
DEV=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper
CODES=(1BIA 1A0F 1AK5 1DAW 1R1K); VARS=(bshift split bshift_split)
CODE=${CODES[$((SLURM_ARRAY_TASK_ID / 3))]}; VAR=${VARS[$((SLURM_ARRAY_TASK_ID % 3))]}
cd "$REPO"; export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
OUT=$REPO/alignment_lab/runs/af_mr/$CODE/refine_${VAR}_rb; mkdir -p $OUT
"$PY" -u torchref/cli/refine.py -m alignment_lab/runs/af_mr/$CODE/placed_${VAR}.pdb -sf $DEV/data/$CODE/$CODE.mtz -o $OUT -n 10 --mode separate --xray-mode ml --weights '{"adp": 0.02}' --with-rigid-body > $OUT/refine.log 2>&1
echo "VAR code=$CODE var=$VAR rc=$? $("$PY" -c "import json;d=json.load(open('$OUT/refinement_history.json'))['final_statistics'];print(f\"rwork={d['R_work']:.4f} rfree={d['R_free']:.4f}\")" 2>/dev/null)"
