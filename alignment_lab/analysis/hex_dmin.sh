#!/bin/bash
# The two hexagonal AlphaFold placements that miss: does a higher-resolution
# window put the truth into the shortlist?
#SBATCH --job-name=hexdmin
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-5
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
D=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/alignment_lab" TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
ARMS=(
 "$D/data/1BYW/1BYW.mtz $D/figure2_alphafold_start/placed/1BYW_af.pdb --d-min 3.5 --tag 1BYW_phaserAF_dmin3.5"
 "$D/data/1BYW/1BYW.mtz $D/figure2_alphafold_start/placed/1BYW_af.pdb --d-min 3.0 --tag 1BYW_phaserAF_dmin3.0"
 "$D/data/1BYW/1BYW.mtz $D/figure2_alphafold_start/placed/1BYW_af.pdb --d-min 2.6 --tag 1BYW_phaserAF_dmin2.6"
 "$D/data/1A4E/1A4E.mtz $D/figure2_alphafold_start/placed/1A4E_af.pdb --d-min 3.0 --tag 1A4E_phaserAF_dmin3.0"
 "$D/data/1A4E/1A4E.mtz $D/figure2_alphafold_start/placed/1A4E_af.pdb --d-min 4.0 --tag 1A4E_phaserAF_dmin4.0"
 "$D/data/1BYW/1BYW.mtz $D/data/1BYW/1BYW.pdb --chain A --shift center --tag 1BYW_dep_centered_scan"
)
"$PY" -u alignment_lab/diagnostics/debug_place_paths.py ${ARMS[$SLURM_ARRAY_TASK_ID]} 2>&1 | grep -v "Warning\|warnings.warn\|ModelFT copied\|Parametrization built\|SfFFT:" | grep -A25 "^PLACE\|rmsd\|shift\|Traceback"
