#!/bin/bash
# Is reading the Wigner d-table off GPFS actually cheaper than recomputing it?
# Only a COLD read answers that, so the writer and the reader must be different
# nodes -- a same-node re-read is served from the page cache and is meaningless.
#SBATCH --job-name=frf_cold
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
MODE="$1"
DIR=alignment_lab/runs/gpfs_cold
mkdir -p "$DIR" alignment_lab/slurm
echo "mode=$MODE host=$(hostname)"
"$PY" -u -c "
import os, time, torch
mode, d = '$MODE', '$DIR'
for L, mb in ((65, 88), (101, 330)):
    p = os.path.join(d, f'dtable_L{L}.pt')
    if mode == 'write':
        n = int(mb * 1e6 / 4)
        torch.save(torch.zeros(n, dtype=torch.float32), p)
        print(f'  wrote L={L}  {os.path.getsize(p)/1e6:.0f} MB')
    else:
        t0 = time.perf_counter(); torch.load(p, map_location='cpu')
        cold = time.perf_counter() - t0
        t0 = time.perf_counter(); torch.load(p, map_location='cpu')
        warm = time.perf_counter() - t0
        print(f'  L={L}  {os.path.getsize(p)/1e6:5.0f} MB   cold {cold*1e3:7.0f} ms   '
              f'warm {warm*1e3:6.0f} ms')
"
echo "exit_code=$?"
