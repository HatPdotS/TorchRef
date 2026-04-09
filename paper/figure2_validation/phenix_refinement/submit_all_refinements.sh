#!/bin/bash

################################################################################
# Batch submission script for PHENIX refinement with Ramachandran restraints
################################################################################
# Usage: ./submit_all_refinements.sh [optional: list_of_pdb_ids.txt]
#
# If no argument provided, will submit jobs for all structures in ../data/
# If a file is provided, will submit jobs only for PDB IDs listed in that file
################################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PAPER_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
DATA_DIR="$( cd "${PAPER_DIR}/../data" && pwd )"
REFINE_SCRIPT="${SCRIPT_DIR}/phenix_refine.sh"

# Check if refinement script exists
if [ ! -f "${REFINE_SCRIPT}" ]; then
    echo "ERROR: Refinement script not found: ${REFINE_SCRIPT}"
    exit 1
fi

# Get list of PDB IDs
if [ $# -eq 1 ] && [ -f "$1" ]; then
    echo "Reading PDB IDs from: $1"
    PDB_LIST=$(cat "$1")
else
    echo "Scanning data directory: ${DATA_DIR}"
    PDB_LIST=$(ls -1 "${DATA_DIR}" | grep -E '^[A-Z0-9]{4}$')
fi

# Count total
TOTAL=$(echo "${PDB_LIST}" | wc -w)
echo "Found ${TOTAL} structures to refine"
echo ""

# Confirm submission
read -p "Submit ${TOTAL} PHENIX refinement (with Ramachandran) jobs? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Submit jobs
SUBMITTED=0
SKIPPED=0

for PDB_ID in ${PDB_LIST}; do
    SOURCE_DIR="${DATA_DIR}/${PDB_ID}"
    PDB_FILE="${SOURCE_DIR}/${PDB_ID}_shaken.pdb"
    DATA_FILE="${SOURCE_DIR}/${PDB_ID}.mtz"

    # Check if required files exist
    if [ ! -f "${PDB_FILE}" ] || [ ! -f "${DATA_FILE}" ]; then
        echo "SKIP ${PDB_ID}: Missing _shaken.pdb or .mtz file"
        ((SKIPPED++))
        continue
    fi

    # Submit job
    JOB_ID=$(sbatch --job-name="phenix_rama_${PDB_ID}" "${REFINE_SCRIPT}" "${PDB_ID}" 2>&1 | grep -oP '\d+')

    if [ $? -eq 0 ]; then
        echo "SUBMITTED ${PDB_ID}: Job ID ${JOB_ID}"
        ((SUBMITTED++))
    else
        echo "FAILED ${PDB_ID}: Could not submit job"
        ((SKIPPED++))
    fi

    # Small delay to avoid overwhelming the scheduler
    sleep 0.1
done

echo ""
echo "================================================================================"
echo "Submission complete"
echo "================================================================================"
echo "Submitted: ${SUBMITTED}"
echo "Skipped:   ${SKIPPED}"
echo "Total:     ${TOTAL}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check status with: sacct -u \$USER"
