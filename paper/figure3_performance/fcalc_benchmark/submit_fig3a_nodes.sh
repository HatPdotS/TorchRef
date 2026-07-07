#!/bin/bash
# Fig3a (fcalc) benchmark — one EXCLUSIVE node per structure.
#
# Why: this is a PERFORMANCE benchmark, so co-locating CPU-heavy jobs on a node
# corrupts the timing (shared memory bandwidth / L3). Each structure's CPU
# thread-scaling therefore runs as its own job on its OWN exclusive node (clean
# timing + full parallelism). The single GPU point per structure is CUDA-event
# timed (contention-insensitive), so all structures' GPU points run in one GPU
# job. A dependent job aggregates every per-structure JSON into summary.csv and
# renders the figure.
set -euo pipefail

REPO="/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev"
BENCH="${REPO}/paper/figure3_performance/fcalc_benchmark"
PY="${REPO}/.dev/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${REPO}/paper/figure3_performance/data/fcalc/results_${TS}"
THREADS="1 2 4 8 16 32"
mkdir -p "${OUT}"
cd "${BENCH}"

"${PY}" -c "import benchmark_thread_scaling as b; print('\n'.join(b.discover_structures()))" \
    > "${OUT}/structures.txt"
N=$(wc -l < "${OUT}/structures.txt"); ARR="0-$((N - 1))"
echo "Structures (${N}): $(tr '\n' ' ' < "${OUT}/structures.txt")"
echo "Output: ${OUT}"

# --- CPU thread-scaling: one exclusive node per structure ---
CPU=$(sbatch --parsable --array="${ARR}" --job-name=fig3a_cpu \
    --partition=day --exclusive --time=04:00:00 \
    --output="${BENCH}/fig3a_cpu_%A_%a.out" \
    --wrap "set -e; STRUCT=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" ${OUT}/structures.txt); \
echo \"node \$(hostname)  structure \${STRUCT}\"; \
cd ${BENCH} && ${PY} benchmark_thread_scaling.py --structures \${STRUCT} --no_gpu \
--threads ${THREADS} --n_iterations 10 --n_warmup 3 --no_summary \
--output_dir ${OUT} --timeout 1800")
echo "CPU array: ${CPU} [${ARR}] (exclusive nodes)"

# --- GPU point: all structures, one A100 job ---
GPU=$(sbatch --parsable --job-name=fig3a_gpu \
    --partition=gpu --gres=gpu:nvidia_a100-pcie-40gb:1 --cpus-per-task=8 --mem=32G \
    --time=02:00:00 --output="${BENCH}/fig3a_gpu_%j.out" \
    --wrap "cd ${BENCH} && ${PY} benchmark_thread_scaling.py --gpu_only \
--n_iterations 10 --n_warmup 3 --no_summary --output_dir ${OUT} --timeout 1800")
echo "GPU job: ${GPU}"

# --- Aggregate once everything finishes (build summary.csv only) ---
# NOTE: the Figure 3a speed panel needs BOTH this fcalc summary AND the
# SFcalculator-GPU data from SF_calc_comparison, so the figure is NOT rendered
# here. After this run and the SF run both finish, render it with:
#   plot_figure3a.py --fcalc-dir ${OUT} --sf-dir <SF results dir>
PLOT=$(sbatch --parsable --job-name=fig3a_agg --partition=hour --time=00:20:00 \
    --mem=8G --dependency="afterany:${CPU}:${GPU}" \
    --output="${BENCH}/fig3a_agg_%j.out" \
    --wrap "cd ${BENCH} && ${PY} benchmark_thread_scaling.py --aggregate --output_dir ${OUT}")
echo "Aggregate job: ${PLOT} (after ${CPU}, ${GPU})"
echo
echo "Watch: squeue -u \$USER   |   summary: ${OUT}/summary.csv"
echo "Then render Fig3a: ${PY} plot_figure3a.py --fcalc-dir ${OUT} --sf-dir <SF results dir>"
