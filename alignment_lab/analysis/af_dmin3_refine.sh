#!/bin/bash
# Refine the 15-3 A placements with the same recipe as the 15-4 A pilot, so the
# window's effect is read on R-free and not on the pose alone.
#SBATCH --job-name=afr3
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-23
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
DEV=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper
read CODE NCOMP NCOP <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$REPO/alignment_lab/analysis/af_pilot_worklist.txt")"
cd "$REPO"; export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PDB=$REPO/alignment_lab/runs/af_mr_dmin3.0/$CODE/torchref_placed.pdb
[ -f "$PDB" ] || { echo "R3 code=$CODE MISSING"; exit 0; }
OUT=$REPO/alignment_lab/runs/af_mr_dmin3.0/$CODE/refine_rb; mkdir -p $OUT
"$PY" -u torchref/cli/refine.py -m $PDB -sf $DEV/data/$CODE/$CODE.mtz -o $OUT -n 10 --mode separate --xray-mode ml --weights '{"adp": 0.02}' --with-rigid-body > $OUT/refine.log 2>&1
echo "R3 code=$CODE rc=$? $("$PY" -c "import json;d=json.load(open('$OUT/refinement_history.json'))['final_statistics'];print(f\"rwork={d['R_work']:.4f} rfree={d['R_free']:.4f}\")" 2>/dev/null)"
