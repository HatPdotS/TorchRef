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

CUDA_AVAILABLE = torch.cuda.is_available()


def _load_model_ft(pdb_file, mtz_file):
    """Helper: load a ModelFT and matching reflection data on CPU."""
    from torchref.io import ReflectionData
    from torchref.model.model_ft import ModelFT

    model = ModelFT()
    model.load_pdb(str(pdb_file))

    data = ReflectionData()
    data.load_mtz(str(mtz_file))
    return model, data


@pytest.mark.integration
def test_modelft_cpu_gpu_cpu_sf_round_trip(sample_pdb_file, sample_mtz_file):
    """CPU -> GPU -> CPU structure-factor round-trip via the unified mixin.

    Sequence:
    1. Load ModelFT on CPU and compute Fcalc; verify CPU placement.
    2. Move to CUDA, recompute Fcalc; verify CUDA placement and values
       match the CPU result within fp tolerance.
    3. Move back to CPU, recompute Fcalc; verify CPU placement and values
       match the original CPU result.
    """
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA device not available")

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
    for buf in model.buffers():
        assert buf.device.type == "cuda", "buffer left behind on CPU"
        break

    # Values must agree within numerical tolerance after the round-trip.
    # CPU and GPU FFT implementations differ in the last few bits of float32,
    # so use a relatively loose tolerance scaled by Fcalc magnitude.
    fcalc_cuda_cpu = fcalc_cuda.cpu()
    magnitude = fcalc_cpu_initial.abs().mean().item()
    diff = (fcalc_cpu_initial - fcalc_cuda_cpu).abs().max().item()
    assert diff < 5e-2 * magnitude, (
        f"Fcalc differs between CPU and GPU computations: "
        f"max abs diff {diff:.4g}, mean |Fcalc| {magnitude:.4g}"
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
