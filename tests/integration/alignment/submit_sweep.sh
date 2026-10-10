#!/bin/bash
# Submit one SLURM job per (PDB × trial) — 11 PDBs × 3 trials = 33 jobs.
#
# Usage: ./submit_sweep.sh [N_TRIALS_PER_PDB]   (default 3)

set -euo pipefail
N=${1:-3}

PDBS=(1AK5 1DAW 2DQ6 3A5V 3E98 3GR5 3K7M 3VRJ 4BX9 5BOV 6G9X)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="$SCRIPT_DIR"
LOG_DIR="$(pwd)/sweep_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
cd "$LOG_DIR"

# Pick a base seed (fixed for reproducibility across the sweep; per-trial
# seed is base + trial_idx so each (pdb, trial) gets a distinct seed).
BASE_SEED=${BASE_SEED:-42}

echo "Submitting $((${#PDBS[@]} * N)) jobs into $LOG_DIR" >&2
JOBIDS=()
for pdb in "${PDBS[@]}"; do
  for trial in $(seq 0 $((N - 1))); do
    seed=$((BASE_SEED + trial * 1000003))
    jobid=$(sbatch --parsable --job-name="fit_${pdb}_t${trial}" \
                   "$SUBMIT_DIR/sweep_slurm.sh" "$pdb" "$seed")
    echo "  $pdb trial $trial → seed=$seed jobid=$jobid"
    JOBIDS+=("$jobid")
  done
done

echo
echo "Submitted ${#JOBIDS[@]} jobs. Track with:"
echo "  squeue --user \$USER --jobs=$(IFS=,; echo "${JOBIDS[*]}")"
echo "Logs in: $LOG_DIR"
