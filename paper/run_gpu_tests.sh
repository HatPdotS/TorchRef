#!/bin/bash
# Run the CUDA-marked test suite on a real GPU node.
#
# This node (ra-l-005) has a Quadro P4000 = sm_61, which this PyTorch build does not support,
# so every CUDA-marked test here fails with `torch.AcceleratorError` and the whole GPU surface
# of the library is effectively untested locally. The x-ray target rework moved
# `estimate_beta` -- the exact code that memory `gpu_scale_refinement_blowup` records as
# non-deterministic ACROSS GPU processes -- so `test_gpu_matches_cpu` is the one correctness
# check with no local substitute.
#
# Runs the full `cuda` + `gpu` marked set rather than that single test: none of it has ever
# executed on this hardware, so "the one test I was worried about" is a poor sample.
#
#   sbatch paper/run_gpu_tests.sh
#
#SBATCH --job-name=trgputest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/gpu_tests_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/gpu_tests_%j.out

set -o pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev
PY=$REPO/.dev/bin/python
cd "$REPO" || exit 1

echo "== node: $(hostname)  job: $SLURM_JOB_ID =="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0),
          "capability", torch.cuda.get_device_capability(0))
    print("build supports:", torch.cuda.get_arch_list())
EOF

echo
echo "== 1. the determinism gate, verbosely (the reason this job exists) =="
"$PY" -m pytest tests/unit/refinement/test_estimate_beta_determinism.py -v -rA --no-header

echo
echo "== 2. every cuda- or gpu-marked test in the tree =="
"$PY" -m pytest tests/ -m "cuda or gpu" -q -rf --no-header --durations=15

echo
echo "== 3. x-ray target parity + gradient correctness, unfiltered, on GPU =="
"$PY" -m pytest tests/unit/refinement/test_xray_target_parity.py \
               tests/unit/test_gradient_correctness.py -q -rf --no-header

echo "== done rc=$? =="
