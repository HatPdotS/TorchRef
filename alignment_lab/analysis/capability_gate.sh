#!/bin/bash
# On a device WITH float64 the capability helpers must resolve to exactly what
# was hardcoded before, so this has to be bit-identical. That is the whole gate:
# the MPS branch cannot be exercised here and is not claimed to work.
#SBATCH --job-name=capgate
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
NEW=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
OUT=$NEW/alignment_lab/slurm
cd "$NEW"; export PYTHONPATH="$NEW"
export TORCHREF_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname)"
LOG=$OUT/capgate_tests_$SLURM_JOB_ID.log
"$PY" -m pytest tests/unit/alignment tests/unit/frf_separate tests/unit/model \
      tests/unit/test_imports_smoke.py -q > "$LOG" 2>&1
rc=$?; echo "=== TESTS rc=$rc ==="; tail -12 "$LOG"
echo "=== capability helpers resolve as expected on this host ==="
"$PY" - <<'PYEOF'
import torch
from torchref.config import (supports_double, widest_complex_dtype,
                             widest_float_dtype)
print(f"  cpu: supports_double={supports_double('cpu')} "
      f"float={widest_float_dtype('cpu')} complex={widest_complex_dtype('cpu')}")
assert widest_float_dtype("cpu") is torch.float64
assert widest_complex_dtype("cpu") is torch.complex128
print("  mps branch (not exercisable here, resolution only):")
from torchref.config import _NO_DOUBLE_DEVICE_TYPES
print(f"    device types without float64: {_NO_DOUBLE_DEVICE_TYPES}")
PYEOF
echo "=== fingerprint vs the committed state (baseline worktree at fe53d373) ==="
OLD=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/_stagea_baseline
for pdb in 3K7M 1DAW; do
  for tree in OLD NEW; do
    eval "root=\$$tree"; cd "$root"
    PYTHONPATH="$root" "$PY" -u "$NEW/alignment_lab/analysis/frf_fingerprint.py" \
      --pdb "$pdb" --lmax-cap 64 > "$OUT/cap_${tree}_${pdb}.txt" 2>/dev/null
  done
  diff -q <(grep '^FP ' "$OUT/cap_OLD_${pdb}.txt") <(grep '^FP ' "$OUT/cap_NEW_${pdb}.txt") >/dev/null \
    && echo "IDENTICAL $pdb ($(grep -c '^FP ' "$OUT/cap_NEW_${pdb}.txt") peaks)" \
    || { echo "DIFFERS $pdb"; diff <(grep '^FP ' "$OUT/cap_OLD_${pdb}.txt") <(grep '^FP ' "$OUT/cap_NEW_${pdb}.txt") | head -4; }
done
echo "capgate_rc=$rc"
