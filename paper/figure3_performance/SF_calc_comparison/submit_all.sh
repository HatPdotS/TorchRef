#!/bin/bash
# Submit the CPU + GPU SF-comparison benchmarks, then a dependent plotting job.
# Both device jobs write into a shared timestamped results dir so the plotter
# can combine them into per-device figures.
set -euo pipefail

BENCH_DIR="/das/work/p17/p17490/Peter/Library/work_trees_torchref/comparison_SFcalc/paper/figure3_performance/SF_calc_comparison"
PYTHON="/das/work/p17/p17490/Peter/Library/work_trees_torchref/comparison_SFcalc/.dev/bin/python"
cd "${BENCH_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${BENCH_DIR}/results_${TS}"
mkdir -p "${OUTPUT_DIR}"
echo "Shared results dir: ${OUTPUT_DIR}"

CPU_JID=$(OUTPUT_DIR="${OUTPUT_DIR}" sbatch --parsable submit_cpu.sbatch)
echo "Submitted CPU job: ${CPU_JID}"

GPU_JID=$(OUTPUT_DIR="${OUTPUT_DIR}" sbatch --parsable submit_gpu.sbatch)
echo "Submitted GPU job: ${GPU_JID}"

# Plot once both device jobs have finished (afterany: plot whatever completed).
PLOT_JID=$(sbatch --parsable \
    --job-name=sf_cmp_plot \
    --partition=hour \
    --time=00:15:00 \
    --cpus-per-task=2 \
    --mem=8G \
    --output="${BENCH_DIR}/sf_cmp_plot_%j.out" \
    --dependency="afterany:${CPU_JID}:${GPU_JID}" \
    --wrap "${PYTHON} ${BENCH_DIR}/plot_results.py --results-dir ${OUTPUT_DIR}")
echo "Submitted plot job: ${PLOT_JID} (after ${CPU_JID}, ${GPU_JID})"

echo
echo "Watch with: squeue -u \$USER"
echo "Results will be in: ${OUTPUT_DIR}/{cpu,gpu}/summary.csv"
echo "Figures in:        ${OUTPUT_DIR}/figure3_sf_comparison_{cpu,gpu}.png"
