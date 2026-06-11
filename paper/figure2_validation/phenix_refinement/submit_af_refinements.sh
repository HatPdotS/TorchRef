#!/bin/bash
# Submit PHENIX refinement for the AlphaFold-start arm: one job per Phaser-placed
# model (figure2_alphafold_start/placed/{code}_af.pdb), passing START_MODEL=af so
# phenix_refine.sh reads the placed model and writes to phenix_refinements_af/.
# Non-interactive (no confirm prompt), so it can be driven programmatically.
#
# Usage: ./submit_af_refinements.sh
set -euo pipefail

PAPER_DIR="/das/work/p17/p17490/Peter/Library/work_trees_torchref/review/paper"
PLACED_DIR="${PAPER_DIR}/figure2_alphafold_start/placed"
REFINE_SCRIPT="${PAPER_DIR}/figure2_validation/phenix_refinement/phenix_refine.sh"

submitted=0
for pdb in "${PLACED_DIR}"/*_af.pdb; do
    [ -e "$pdb" ] || { echo "no placed models in ${PLACED_DIR}"; exit 1; }
    code=$(basename "$pdb" _af.pdb)
    JOB=$(sbatch --parsable --job-name="phenix_af_${code}" "${REFINE_SCRIPT}" "${code}" af)
    echo "SUBMITTED ${code}: job ${JOB}"
    submitted=$((submitted + 1))
    sleep 0.1
done

echo
echo "Submitted ${submitted} PHENIX af-arm jobs."
echo "Outputs:  ${PAPER_DIR}/phenix_refinements_af/{code}/"
echo "Track:    squeue -u \$USER --name=phenix_af_*"
