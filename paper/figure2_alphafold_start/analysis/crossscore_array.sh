#!/bin/bash
# PHENIX-vs-REFMAC cross-scoring: score one final model with phenix.model_vs_data.
# One array task per worklist line (build_crossscore_worklist.py).
#
#   sbatch --array=1-N%60 crossscore_array.sh
#
#SBATCH --job-name=xscore
#SBATCH --partition=hour
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/runs/crossscore/slurm/%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/runs/crossscore/slurm/%A_%a.out

set +o nounset
# `module load phenix` errors on these nodes (modulefile uses an unsupported
# `module-url` command), so source the phenix env directly. Edit PHENIX_ENV to
# switch versions; this is the only place the version is pinned.
PHENIX_ENV=/opt/psi/MX/phenix/phenix-1.20-4459/phenix_env.sh
source "$PHENIX_ENV"

WORKLIST=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/runs/crossscore/worklist.txt

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")
if [ -z "$LINE" ]; then
    echo "no worklist line for task ${SLURM_ARRAY_TASK_ID}"; exit 0
fi
IFS=$'\t' read -r ENGINE CODE MODEL MTZ OUTLOG <<< "$LINE"
echo "[$(date)] scoring model_engine=$ENGINE code=$CODE"
echo "  model: $MODEL"
echo "  data : $MTZ"

SCRATCH=/tmp/mvd_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$SCRATCH" && cd "$SCRATCH" || exit 1
export CCP4_SCR="$SCRATCH"

phenix.model_vs_data "$MODEL" "$MTZ" > "$OUTLOG" 2>&1
RC=$?

cd / && rm -rf "$SCRATCH"
echo "[$(date)] done rc=$RC -> $OUTLOG"
exit $RC
