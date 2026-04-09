#!/bin/bash
#SBATCH --job-name=phenix_rama
#SBATCH --output=phenix_refine_%A_%a.out
#SBATCH --error=phenix_refine_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=day

################################################################################
# PHENIX Refinement Script with Ramachandran Restraints
################################################################################
# Usage: sbatch phenix_refine.sh <PDB_ID>
# Example: sbatch phenix_refine.sh 1A4E
#
# Same as phenix_refinement but with Ramachandran plot restraints ENABLED.
################################################################################

# Check if PDB ID was provided
if [ $# -eq 0 ]; then
    echo "ERROR: No PDB ID provided"
    echo "Usage: sbatch phenix_refine.sh <PDB_ID>"
    echo "Example: sbatch phenix_refine.sh 1A4E"
    exit 1
fi

PDB_ID=$1

# Define paths — all relative to this script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PAPER_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
DATA_DIR="$( cd "${PAPER_DIR}/../data" && pwd )"

# Input files
PDB_FILE="${DATA_DIR}/${PDB_ID}/${PDB_ID}_shaken.pdb"
DATA_FILE="${DATA_DIR}/${PDB_ID}/${PDB_ID}.mtz"
SOURCE_RESTRAINTS_DIR="${DATA_DIR}/restraints"

# Output under paper/phenix_refinements/ (symlink)
OUTPUT_DIR="$( cd "${PAPER_DIR}/../phenix_refinements" && pwd )/${PDB_ID}"
REFINE_DIR="${OUTPUT_DIR}"
RESTRAINTS_DIR="${OUTPUT_DIR}/restraints"
OUTPUT_PREFIX="${OUTPUT_DIR}/${PDB_ID}_refined"

# Log files
LOG_FILE="${OUTPUT_DIR}/phenix_refine.log"
RESTRAINT_LOG="${OUTPUT_DIR}/restraint_generation.log"

################################################################################
# Pre-flight checks
################################################################################

echo "================================================================================"
echo "PHENIX Refinement (with Ramachandran restraints) for ${PDB_ID}"
echo "================================================================================"
echo "Start time: $(date)"
echo "Source directory: ${SOURCE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check if source directory exists
if [ ! -d "${SOURCE_DIR}" ]; then
    echo "ERROR: Source directory does not exist: ${SOURCE_DIR}"
    exit 1
fi

# Check if PDB file exists
if [ ! -f "${PDB_FILE}" ]; then
    echo "ERROR: PDB file not found: ${PDB_FILE}"
    exit 1
fi

# Check if data file exists
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Structure factor file not found: ${DATA_FILE}"
    exit 1
fi

# Create output and restraints directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESTRAINTS_DIR}"

echo "Input files:"
echo "  Model: ${PDB_FILE}"
echo "  Data:  ${DATA_FILE}"

################################################################################
# Load PHENIX module
################################################################################

echo ""
echo "Loading PHENIX module..."
module load phenix/phenix-1.20-4459

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load PHENIX module"
    exit 1
fi

echo "PHENIX version:"
phenix.version

################################################################################
# Fix PDB CRYST1 record if needed (replace "None" Z value with proper value)
################################################################################

echo ""
echo "================================================================================"
echo "Checking and fixing PDB CRYST1 record..."
echo "================================================================================"

# Check if CRYST1 has "None" as Z value and fix it
FIXED_PDB="${OUTPUT_DIR}/${PDB_ID}_input.pdb"

if grep -q "^CRYST1.*None" "${PDB_FILE}"; then
    echo "Found 'None' in CRYST1 record, fixing..."
    # Replace "None" with "   12" (12 is common for P 61 2 2, formatted as 5 chars)
    # The MTZ file has the correct symmetry which PHENIX will use
    sed 's/\(CRYST1.*P [0-9]* *[0-9]* *[0-9]* *\)None/\1   12/' "${PDB_FILE}" > "${FIXED_PDB}"
    PDB_FILE="${FIXED_PDB}"
    echo "CRYST1 record fixed"
else
    echo "CRYST1 record is OK"
fi

################################################################################
# Extract ligands from PDB and generate restraints using phenix.elbow
################################################################################

echo ""
echo "================================================================================"
echo "Extracting ligands and generating restraints..."
echo "================================================================================"

# Extract HETNAM records to identify ligands
LIGANDS=$(grep "^HETNAM" "${PDB_FILE}" | awk '{print $2}' | sort -u)

# Filter out common solvents and ions
COMMON_SOLVENTS="HOH WAT NA CL K CA MG ZN FE SO4 PO4 GOL EDO"

# Build restraints argument
RESTRAINTS_ARG=""
LIGAND_COUNT=0

for ligand in ${LIGANDS}; do
    # Skip common solvents and ions
    if echo "${COMMON_SOLVENTS}" | grep -wq "${ligand}"; then
        echo "  Skipping ${ligand} (common solvent/ion)"
        continue
    fi

    # Check if restraint already exists in source directory
    SOURCE_RESTRAINT="${SOURCE_RESTRAINTS_DIR}/${ligand}.cif"
    TARGET_RESTRAINT="${RESTRAINTS_DIR}/${ligand}.cif"

    if [ -f "${SOURCE_RESTRAINT}" ]; then
        echo "  Copying existing restraint for ${ligand}..."
        cp "${SOURCE_RESTRAINT}" "${TARGET_RESTRAINT}"
        RESTRAINTS_ARG="${RESTRAINTS_ARG} ${TARGET_RESTRAINT}"
        ((LIGAND_COUNT++))
    else
        echo "  Generating restraint for ${ligand} using phenix.elbow..."

        # Generate restraint using phenix.elbow
        phenix.elbow \
            --residue="${ligand}" \
            --output-dir="${RESTRAINTS_DIR}" \
            --do-all \
            --opt \
            --final-geometry-pkl \
            --write-cif \
            --quiet >> "${RESTRAINT_LOG}" 2>&1

        if [ $? -eq 0 ] && [ -f "${TARGET_RESTRAINT}" ]; then
            echo "    SUCCESS: ${ligand}.cif generated"
            RESTRAINTS_ARG="${RESTRAINTS_ARG} ${TARGET_RESTRAINT}"
            ((LIGAND_COUNT++))
        else
            echo "    WARNING: Failed to generate restraint for ${ligand}"
            echo "    Check ${RESTRAINT_LOG} for details"
        fi
    fi
done

echo ""
echo "Total ligand restraints prepared: ${LIGAND_COUNT}"

if [ ${LIGAND_COUNT} -eq 0 ]; then
    echo "No ligands found or all were filtered as solvents/ions"
fi

################################################################################
# Run PHENIX refinement - Full refinement with Ramachandran restraints
################################################################################

echo ""
echo "================================================================================"
echo "Running phenix.refine (10 cycles, nproc=4, Ramachandran restraints ON)..."
echo "================================================================================"
echo "Using PDB: ${PDB_FILE}"
echo "Using MTZ with R-free flags: ${DATA_FILE}"
echo ""
echo "Refinement strategy:"
echo "  - 10 macro cycles"
echo "  - 4 CPU cores (nproc=4)"
echo "  - Coordinates (xyz) refinement"
echo "  - Individual B-factors (ADP)"
echo "  - Occupancy refinement"
echo "  - Real-space refinement"
echo "  - Ordered solvent update"
echo "  - Ramachandran plot restraints: ENABLED"
echo ""

cd "${OUTPUT_DIR}"

# Run phenix.refine with Ramachandran restraints enabled
phenix.refine \
    "${PDB_FILE}" \
    "${DATA_FILE}" \
    ${RESTRAINTS_ARG} \
    --overwrite \
    output.prefix="${PDB_ID}_refined" \
    refinement.main.number_of_macro_cycles=10 \
    refinement.main.nproc=4 \
    refinement.refine.strategy=individual_sites+individual_adp+occupancies \
    refinement.main.simulated_annealing=false \
    refinement.target_weights.optimize_xyz_weight=false \
    refinement.target_weights.optimize_adp_weight=false \
    refinement.main.bulk_solvent_and_scale=true \
    refinement.main.ordered_solvent=false \
    refinement.ordered_solvent.mode=every_macro_cycle \
    refinement.pdb_interpretation.ramachandran_plot_restraints.enabled=true \
    write_def_file=false \
    write_eff_file=false \
    write_geo_file=false \
    --quiet 2>&1 | tee "${LOG_FILE}"

PHENIX_EXIT_CODE=$?

################################################################################
# Check results
################################################################################

echo ""
echo "================================================================================"
echo "Refinement completed"
echo "================================================================================"

if [ ${PHENIX_EXIT_CODE} -eq 0 ]; then
    echo "Status: SUCCESS"

    # List output files
    echo ""
    echo "Output files in ${OUTPUT_DIR}:"
    ls -lh "${OUTPUT_DIR}/" | grep "${PDB_ID}_refined"

    # Extract and display final R-factors
    echo ""
    echo "Final R-factors:"
    grep -A 2 "Final R-work" "${OUTPUT_DIR}/${PDB_ID}_refined_001.log" 2>/dev/null || \
    grep "r_work\|r_free" "${LOG_FILE}" | tail -5

    # Extract runtime
    echo ""
    grep "wall clock time" "${OUTPUT_DIR}/${PDB_ID}_refined_001.log" 2>/dev/null || echo "Runtime: See log for details"
else
    echo "Status: FAILED"
    echo "Exit code: ${PHENIX_EXIT_CODE}"
    echo ""
    echo "Check logs for errors:"
    echo "  ${LOG_FILE}"
    echo "  ${OUTPUT_DIR}/${PDB_ID}_refined_001.log"
fi

echo ""
echo "End time: $(date)"
echo "================================================================================"
