#!/bin/bash
# Stage A: the alignment + frf_separate suites, fast then --run-slow.
#SBATCH --job-name=stagea_tests
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname)"

LOG=alignment_lab/slurm/stagea_fast_$SLURM_JOB_ID.log
"$PY" -m pytest tests/unit/alignment tests/unit/frf_separate tests/unit/model \
      tests/unit/test_imports_smoke.py -q > "$LOG" 2>&1
rc_fast=$?
echo "=== FAST rc=$rc_fast ==="
tail -25 "$LOG"

LOG2=alignment_lab/slurm/stagea_slow_$SLURM_JOB_ID.log
"$PY" -m pytest --run-slow tests/unit/alignment tests/unit/frf_separate \
      tests/integration/alignment -q > "$LOG2" 2>&1
rc_slow=$?
echo "=== SLOW rc=$rc_slow ==="
tail -25 "$LOG2"
echo "rc_fast=$rc_fast rc_slow=$rc_slow"
