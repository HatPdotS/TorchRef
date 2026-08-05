"""
Integration tests for the unified DeviceMixin.

The primary acceptance test exercises a CPU -> GPU -> CPU round-trip on a
``ModelFT`` instance: structure factors are recomputed at each leg and
checked for the expected device placement and numerical agreement. This
covers the ``__dict__`` walk (Cell, SfFFT, anomalous cache), the always-
invalidate cache policy, and the post-move ``_rebuild_sf_indices`` hook.
"""

from __future__ import annotations

import pytest
import torch

def _load_model_ft(pdb_file, mtz_file):
    """Helper: load a ModelFT and matching reflection data on CPU.

    The round-trip tests in this file specifically check CPU placement
    before moving to GPU, so the model and data are constructed on CPU
    regardless of the configured default device.
    """
    from torchref.io import ReflectionData
    from torchref.model.model_ft import ModelFT

    cpu = torch.device("cpu")
    model = ModelFT(device=cpu)
    model.load_pdb(str(pdb_file))

    data = ReflectionData(device=cpu)
    data.load_mtz(str(mtz_file))
    return model, data


@pytest.mark.integration
@pytest.mark.cuda
def test_modelft_cpu_gpu_cpu_sf_round_trip(sample_pdb_file, sample_mtz_file):
    """CPU -> GPU -> CPU structure-factor round-trip via the unified mixin.

    Sequence:
    1. Load ModelFT on CPU and compute Fcalc; verify CPU placement.
    2. Move to CUDA, recompute Fcalc; verify CUDA placement and values
       match the CPU result within fp tolerance.
    3. Move back to CPU, recompute Fcalc; verify CPU placement and values
       match the original CPU result.
    """
    model, data = _load_model_ft(sample_pdb_file, sample_mtz_file)
    hkl, *_ = data()

    # ---- CPU leg ---------------------------------------------------------
    assert hkl.device.type == "cpu", "test setup: hkl should start on CPU"
    fcalc_cpu_initial = model(hkl).detach().clone()
    assert fcalc_cpu_initial.device.type == "cpu"
    assert model.device.type == "cpu"
    assert model.cell.device.type == "cpu"

    # ---- GPU leg ---------------------------------------------------------
    model.to("cuda")
    hkl_cuda = hkl.to("cuda")
    fcalc_cuda = model(hkl_cuda).detach()
    assert fcalc_cuda.device.type == "cuda"
    assert model.device.type == "cuda"
    assert model.cell.device.type == "cuda", "Cell did not migrate to GPU"
    for name, buf in model.named_buffers():
        assert buf.device.type == "cuda", f"buffer {name} left behind on CPU"

    # Values must agree within numerical tolerance after the round-trip.
    #
    # This tolerance is loose for a reason that no longer applies. It was set when the CPU
    # path was a box-separable splat and the GPU path a sphere-truncated work-queue kernel,
    # so the two genuinely differed in truncation shape (~0.3% RMS, ~0.12% R-factor,
    # concentrated on the strongest low-resolution reflections). Every production kernel now
    # applies the identical spherical cutoff -- see ``electron_density/main.py`` -- so the
    # residual should be float32 kernel arithmetic only, which is far smaller.
    #
    # It is deliberately NOT tightened here: this test is CUDA-only and has not been run
    # since the contract was unified, so any number picked now would be a guess. Re-measure
    # on a GPU host and tighten then. The combined relative-OR-absolute form stays either
    # way -- max-abs-vs-mean trips on a single very strong reflection (|F| >> mean |F|) for
    # a tiny relative error.
    fcalc_cuda_cpu = fcalc_cuda.cpu()
    magnitude = fcalc_cpu_initial.abs().mean().item()
    max_abs_diff = (fcalc_cpu_initial - fcalc_cuda_cpu).abs().max().item()
    assert torch.allclose(
        fcalc_cpu_initial, fcalc_cuda_cpu, rtol=5e-3, atol=1e-1 * magnitude
    ), (
        f"Fcalc differs between CPU and GPU computations: "
        f"max abs diff {max_abs_diff:.4g}, mean |Fcalc| {magnitude:.4g}"
    )

    # ---- Back to CPU -----------------------------------------------------
    model.to("cpu")
    fcalc_cpu_final = model(hkl).detach()
    assert fcalc_cpu_final.device.type == "cpu"
    assert model.device.type == "cpu"
    assert model.cell.device.type == "cpu", "Cell did not migrate back to CPU"

    # Round-trip should give bitwise-equivalent (or very close) results.
    assert torch.allclose(
        fcalc_cpu_initial,
        fcalc_cpu_final,
        atol=1e-5,
        rtol=1e-5,
    ), "Fcalc on CPU after round-trip differs from initial CPU result"


@pytest.mark.integration
def test_modelft_cpu_only_recompute_after_to(sample_pdb_file, sample_mtz_file):
    """Even a no-op CPU->CPU ``.to()`` should leave Fcalc usable.

    This guards against the cache-invalidation path destroying state. The
    second call must return a finite tensor of the same shape; numerically
    it should match the first call.
    """
    model, data = _load_model_ft(sample_pdb_file, sample_mtz_file)
    hkl, *_ = data()
    fcalc_before = model(hkl).detach().clone()

    model.to("cpu")  # idempotent move

    fcalc_after = model(hkl)
    assert fcalc_after.device.type == "cpu"
    assert fcalc_after.shape == fcalc_before.shape
    assert torch.allclose(fcalc_before, fcalc_after.detach(), atol=1e-6)
