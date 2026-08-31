#!/bin/bash
#SBATCH --job-name=fullgate
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=day
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# `-c tests/pytest.ini` from the repo ROOT, with --run-slow. Both halves matter
# and they pull in opposite directions: a bare `pytest tests/unit` picks up
# pyproject.toml, where the slow marker silently skips the rotation-search and
# translation tests -- which is how seven of them stayed broken by a merge
# through a gate reporting everything green. But running from `tests/` to get
# the right config then breaks the io tests, which open `tests/files/...`
# relative to the root. Naming the config explicitly satisfies both.
"$PY" -m pytest -c tests/pytest.ini tests/unit --run-slow -q 2>&1 | tail -16
echo "PYTEST_RC=${PIPESTATUS[0]}"
echo "== end-to-end placement, one cell =="
"$PY" -u alignment_lab/diagnostics/pose_recovery.py --pdb 1DAW --trial 0 \
  --arms analytic_r 2>/dev/null | grep '^ROW '
