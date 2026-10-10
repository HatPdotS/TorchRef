#!/bin/bash
# GPU gate for the torch.library dispatch: per-call overhead plus a short 1DAW refinement.
#
#   sbatch -p <gpu-partition> --gres=gpu:1 tests/benchmarks/run_bench_op_overhead.sh
#
# Run it once on this branch and once on `main`, from the repo root of each checkout,
# with that checkout's torchref (and, on this branch, `pip install ./kernels`) installed.
# Gate: the refinement wall time on this branch is within 2% of main's.

#SBATCH -J bench-op-overhead
#SBATCH -c 8
#SBATCH -o bench_op_overhead.%j.txt
#SBATCH -e bench_op_overhead.%j.err

set -euo pipefail
export TORCHREF_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
git log --oneline -1
nvidia-smi --query-gpu=name --format=csv,noheader || true

if [ -f tests/benchmarks/bench_op_overhead.py ]; then
    python tests/benchmarks/bench_op_overhead.py --device cuda --iters 2000
fi

out="${TMPDIR:-/tmp}/bench_refine_${SLURM_JOB_ID:-local}"
# The first run warms caches (Triton autotune, CUDA context); time the next three.
torchref.refine -m tests/files/pdb/1DAW.pdb -sf tests/files/mtz/1DAW.mtz \
    -o "$out/warm" -n 2 --device cuda -v 0 > /dev/null
for i in 1 2 3; do
    /usr/bin/time -f "refine 1DAW 5 cycles: %e s wall" \
        torchref.refine -m tests/files/pdb/1DAW.pdb -sf tests/files/mtz/1DAW.mtz \
        -o "$out/run$i" -n 5 --device cuda -v 0 > /dev/null
done
