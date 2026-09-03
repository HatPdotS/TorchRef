#!/bin/bash
# Does a converged rigid-body step close 1BIA's gap between our placement (B at
# the Wilson level) and Phaser's?
#SBATCH --job-name=rbiter
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-3
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
DEV=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper
ARMS=(ours phaser); ITERS=(200 1000)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % 2))]}; IT=${ITERS[$((SLURM_ARRAY_TASK_ID / 2))]}
CODE=1BIA
cd "$REPO"; export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
if [ $ARM = ours ]; then PDB=alignment_lab/runs/af_mr/$CODE/placed_bshift.pdb; else PDB=$DEV/figure2_alphafold_start/placed/${CODE}_af.pdb; fi
OUT=$REPO/alignment_lab/runs/af_mr/$CODE/refine_${ARM}_rb${IT}; mkdir -p $OUT
"$PY" -u torchref/cli/refine.py -m $PDB -sf $DEV/data/$CODE/$CODE.mtz -o $OUT -n 10 --mode separate --xray-mode ml --weights '{"adp": 0.02}' --with-rigid-body --rigid-body-iter $IT > $OUT/refine.log 2>&1
echo "RBI code=$CODE arm=$ARM iters=$IT rc=$? $("$PY" -c "import json;d=json.load(open('$OUT/refinement_history.json'))['final_statistics'];print(f\"rwork={d['R_work']:.4f} rfree={d['R_free']:.4f}\")" 2>/dev/null)"
grep "rigid-body d_min" $OUT/refine.log
