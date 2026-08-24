#!/bin/bash
# What the two knockouts actually SAVE. The panel run could not answer this: it
# ran production first in every trial, so production alone paid the process-level
# warm-up -- the fused C++ kernel build, the SO(3) sample list, the Wigner block
# memo -- and the later arms inherited all three warm. That is why it reported a
# ~10x "speedup" from deleting a 20-bin regression, which is not credible.
#
# Here: one throwaway run to warm every memo, then rounds with the arm order
# ROTATED so no arm is systematically first.
#SBATCH --job-name=arms_time
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
"$PY" -u - <<'PYEOF'
import statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path("alignment_lab").resolve()))
import torch
torch.set_grad_enabled(False)
sys.path.insert(0, str(Path("alignment_lab/analysis").resolve()))
from lab import FRFConfig, rotated_case, run_frf, seed_for
from panel_arms import knocked_out

ARMS = ["production", "no_brel", "no_friedel"]
cfg = FRFConfig(n_peaks=500, lmax_cap=64)

for pdb in ("3K7M", "1DAW"):
    model, data, _ = rotated_case(pdb, seed_for(pdb, 0))
    run_frf(model, data, cfg, capture_arf=False, verbose=0)   # warm every memo
    t = {a: [] for a in ARMS}
    for r in range(4):
        order = ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]    # rotate
        for arm in order:
            model, data, _ = rotated_case(pdb, seed_for(pdb, r))
            t0 = time.time()
            with knocked_out(arm):
                run_frf(model, data, cfg, capture_arf=False, verbose=0)
            t[arm].append(time.time() - t0)
    base = statistics.median(t["production"])
    print(f"--- {pdb} (warm, arm order rotated, 4 rounds) ---")
    for arm in ARMS:
        med = statistics.median(t[arm])
        pd = [t[arm][i] - t["production"][i] for i in range(len(t[arm]))]
        print(f"  {arm:12s} median {med:.3f}s  paired d vs production "
              f"median {statistics.median(pd):+.3f}s "
              f"({100*statistics.median(pd)/base:+.1f}%)  raw {[round(v,3) for v in t[arm]]}")
PYEOF
echo "rc=$?"
