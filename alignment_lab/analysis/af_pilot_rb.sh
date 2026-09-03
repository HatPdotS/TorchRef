#!/bin/bash
# Rigid-body-first refinement of both placements from the pilot: does refining
# the placement as a rigid body first close the gap to Phaser's refined pose?
#SBATCH --job-name=afrb
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
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
OUT=$REPO/alignment_lab/runs/af_mr/$CODE
MTZ=$DEV/data/$CODE/$CODE.mtz
for ARM in ours phaser; do
  if [ $ARM = ours ]; then PDB=$OUT/torchref_placed.pdb; else PDB=$DEV/figure2_alphafold_start/placed/${CODE}_af.pdb; fi
  mkdir -p $OUT/refine_${ARM}_rb
  "$PY" -u torchref/cli/refine.py -m $PDB -sf $MTZ -o $OUT/refine_${ARM}_rb -n 10 --mode separate --xray-mode ml --weights '{"adp": 0.02}' --with-rigid-body > $OUT/refine_${ARM}_rb/refine.log 2>&1
  echo "RB code=$CODE arm=$ARM rc=$? $("$PY" -c "import json;d=json.load(open('$OUT/refine_${ARM}_rb/refinement_history.json'))['final_statistics'];print(f\"rwork={d['R_work']:.4f} rfree={d['R_free']:.4f}\")" 2>/dev/null)"
done
