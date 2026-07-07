#!/bin/bash
# Fig3b (refinement-cycle) benchmark — one EXCLUSIVE node per structure.
#
# Same rationale as Fig3a: clean per-structure CPU timing requires a dedicated
# node (no co-tenant memory-bandwidth contention). The refinement-cycle worker
# (loss + gradient over all targets, with a per-target breakdown) is the heavy
# one, so the CPU jobs get more walltime. GPU points run in one A100 job; a
# dependent job aggregates + plots.
set -euo pipefail

REPO="/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev"
BENCH="${REPO}/paper/figure3_performance/refinement_cycle_benchmark"
PY="${REPO}/.dev/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${REPO}/paper/figure3_performance/data/refinement_cycle/results_${TS}"
THREADS="1 2 4 8 16 32"
mkdir -p "${OUT}"
cd "${BENCH}"

"${PY}" -c "import benchmark_thread_scaling as b; print('\n'.join(b.discover_structures()))" \
    > "${OUT}/structures.txt"
N=$(wc -l < "${OUT}/structures.txt"); ARR="0-$((N - 1))"
echo "Structures (${N}): $(tr '\n' ' ' < "${OUT}/structures.txt")"
echo "Output: ${OUT}"

# --- CPU thread-scaling: one exclusive node per structure (heavy: longer walltime) ---
CPU=$(sbatch --parsable --array="${ARR}" --job-name=fig3b_cpu \
    --partition=day --exclusive --time=12:00:00 \
    --output="${BENCH}/fig3b_cpu_%A_%a.out" \
    --wrap "set -e; STRUCT=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" ${OUT}/structures.txt); \
echo \"node \$(hostname)  structure \${STRUCT}\"; \
cd ${BENCH} && ${PY} benchmark_thread_scaling.py --structures \${STRUCT} --no_gpu \
--threads ${THREADS} --n_iterations 10 --n_warmup 3 --no_summary \
--output_dir ${OUT} --timeout 3600")
echo "CPU array: ${CPU} [${ARR}] (exclusive nodes)"

# --- GPU point: all structures, one A100 job ---
GPU=$(sbatch --parsable --job-name=fig3b_gpu \
    --partition=gpu --gres=gpu:nvidia_a100-pcie-40gb:1 --cpus-per-task=8 --mem=32G \
    --time=03:00:00 --output="${BENCH}/fig3b_gpu_%j.out" \
    --wrap "cd ${BENCH} && ${PY} benchmark_thread_scaling.py --gpu_only \
--n_iterations 10 --n_warmup 3 --no_summary --output_dir ${OUT} --timeout 3600")
echo "GPU job: ${GPU}"

# --- Aggregate + plot once everything finishes ---
PLOT=$(sbatch --parsable --job-name=fig3b_plot --partition=hour --time=00:20:00 \
    --mem=8G --dependency="afterany:${CPU}:${GPU}" \
    --output="${BENCH}/fig3b_plot_%j.out" \
    --wrap "cd ${BENCH} && ${PY} benchmark_thread_scaling.py --aggregate --output_dir ${OUT} \
&& ${PY} ../plot_figure3b.py --results-dir ${OUT}")
echo "Plot job: ${PLOT} (after ${CPU}, ${GPU})"
echo
echo "Watch: squeue -u \$USER   |   summary: ${OUT}/summary.csv   |   figure: ${BENCH%/*}/output/"
