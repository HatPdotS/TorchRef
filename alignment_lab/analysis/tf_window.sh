#!/bin/bash
# The two hexagonal misses: is it the rotation window or the translation window
# that needs the finer high-resolution limit?
#SBATCH --job-name=tfwin
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-3
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
D=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
ARMS=(
 "$D/data/1BYW/1BYW.mtz $D/figure2_alphafold_start/placed/1BYW_af.pdb --d-min 4.0 --tf-d-min 3.0 --tag 1BYW_rot4_tf3"
 "$D/data/1A4E/1A4E.mtz $D/figure2_alphafold_start/placed/1A4E_af.pdb --d-min 4.0 --tf-d-min 3.0 --tag 1A4E_rot4_tf3"
 "$D/data/1BYW/1BYW.mtz $D/figure2_alphafold_start/placed/1BYW_af.pdb --d-min 3.0 --tf-d-min 4.0 --tag 1BYW_rot3_tf4"
 "$D/data/1A4E/1A4E.mtz $D/figure2_alphafold_start/placed/1A4E_af.pdb --d-min 3.0 --tf-d-min 4.0 --tag 1A4E_rot3_tf4"
)
"$PY" -u alignment_lab/diagnostics/debug_place_paths.py ${ARMS[$SLURM_ARRAY_TASK_ID]} 2>&1 | grep -v "Warning\|warnings.warn\|ModelFT copied\|Parametrization built\|SfFFT:" | grep -A6 "^PLACE\|nearest-image\|Traceback"
