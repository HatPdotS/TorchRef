#!/bin/bash
# TorchRef 0-cycle scoring: one model per array task. A fresh process per model
# is memory-safe (large structures, e.g. 1B37's 8.3M A^3 cell, need ~16 G and the
# CPU allocator does not return that grid between models, so long-lived batching
# OOMs). On compute nodes a task is only ~20-40 s (the 1m45s seen on the shared
# head node is contention, not the real cost).
#
#   N=$(wc -l < worklist.txt); sbatch --array=1-$N%96 torchref_score_array.sh
#
# `day` + 2 h, not `hour` + 20 min: the median task really is ~20-40 s, but the
# largest ~60 of 3024 models run far longer and the 20-minute cap silently timed
# every one of them out (2026-08-11). A timed-out task leaves no
# torchref_validate.json, which drops that model's row from the TorchRef-scorer
# column of ExtFig 3 without any error -- the figure just renders short. The
# walltime is sized for the tail, not the median; `hour` cannot express it (1 h cap).
#SBATCH --job-name=trscore
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/runs/crossscore/slurm_tr/%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/runs/crossscore/slurm_tr/%A_%a.out

set +o nounset
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
SCORE=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/analysis/torchref_score.py
WORKLIST=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/runs/crossscore/torchref_worklist.txt
export TORCHREF_NUM_THREADS=2

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")
if [ -z "$LINE" ]; then echo "no line ${SLURM_ARRAY_TASK_ID}"; exit 0; fi
IFS=$'\t' read -r ENGINE CODE MODEL MTZ OUTJSON <<< "$LINE"
echo "[$(date)] torchref-scoring $ENGINE/$CODE"
"$PY" "$SCORE" -m "$MODEL" -sf "$MTZ" -o "$OUTJSON" --device cpu --xray-mode ml
RC=$?
# Capture $? into RC on its OWN line, before anything else runs. `rc=$?` inside a
# string containing $(date) reports the *subshell's* status, not python's -- it read
# 0 for all 5308 tasks of a sweep in which every single one crashed, and because the
# script then exited on a successful echo, SLURM recorded COMPLETED throughout. The
# missing JSONs only surfaced as a nan in one cell of aggregate_crossscore.
echo "[$(date)] done rc=$RC -> $OUTJSON"
exit $RC
