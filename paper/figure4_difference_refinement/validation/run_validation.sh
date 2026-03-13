#!/bin/bash
# Regenerate validation results and CCP4 maps for Figure 4.
#
# Run from this directory:
#   cd paper/figure4_difference_refinement/validation
#   bash run_validation.sh

module load anaconda
conda activate /das/work/p17/p17490/CONDA/torchref

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
OUTPUT_DIR="${SCRIPT_DIR}/output"

BASEARGS="-dsf ${DATA_DIR}/dark-phenix.mtz -lsf ${DATA_DIR}/7YYZ-light.mtz -dm ${DATA_DIR}/8QL2.pdb --dmin 2.2 --plot --write-maps"
TORCH_ARGS="-lm ${DATA_DIR}/torchref_0p18.pdb --fraction 0.18"
EXTRAPOL_ARGS="-lm ${DATA_DIR}/7YYZ.pdb --fraction 0.22"

IBL="resname IBL"
FULL="chain A or chain B"

sbatch -o ${OUTPUT_DIR}/torchref_IBL.log -c 16 torchref.validate-ded ${BASEARGS} ${TORCH_ARGS} --selection "${IBL}" -o ${OUTPUT_DIR}/torchref_IBL
sbatch -o ${OUTPUT_DIR}/torchref_FULL.log -c 16 torchref.validate-ded ${BASEARGS} ${TORCH_ARGS} --selection "${FULL}" -o ${OUTPUT_DIR}/torchref_FULL
sbatch -o ${OUTPUT_DIR}/extrapol_IBL.log -c 16 torchref.validate-ded ${BASEARGS} ${EXTRAPOL_ARGS} --selection "${IBL}" -o ${OUTPUT_DIR}/extrapol_IBL
sbatch -o ${OUTPUT_DIR}/extrapol_FULL.log -c 16 torchref.validate-ded ${BASEARGS} ${EXTRAPOL_ARGS} --selection "${FULL}" -o ${OUTPUT_DIR}/extrapol_FULL

wait
