"""GPU refinement must not collapse the per-bin scale (regression).

Guards the ``estimate_beta`` non-determinism bug: on CUDA, a non-stable
``argsort`` + atomic ``scatter_add`` + the cancellation ``wi = A*B - C**2``
produced a different (often floored) beta per process, which froze a bad beta
into ``refine_lbfgs`` and collapsed a per-bin ``log_scale`` to ~-99 → R-free
blow-up. A short GPU refinement should now (a) stay finite / not collapse the
scale and (b) track the CPU refinement, which the deterministic + float32-stable
``estimate_beta`` restores.
"""

import pytest
import torch


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_5cycle_refinement_does_not_collapse_and_matches_cpu(pdb_dir, mtz_dir):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not pdb.exists() or not mtz.exists():
        pytest.skip("1DAW fixture not present")

    from torchref import LBFGSRefinement

    def run(device):
        ref = LBFGSRefinement(
            data_file=str(mtz),
            pdb=str(pdb),
            target_mode="ml",
            verbose=0,
            device=torch.device(device),
        )
        ref.refine(macro_cycles=5)
        rwork, rfree = ref.get_rfactor()
        log_scale_min = float(ref.scaler.log_scale.detach().min())
        return float(rwork), float(rfree), log_scale_min

    rw_gpu, rf_gpu, ls_gpu = run("cuda")
    rw_cpu, rf_cpu, ls_cpu = run("cpu")

    # (a) GPU refinement stays sane: R-free finite and not blown up, and no
    #     per-bin log_scale collapsed (the bug drove one to ~-99; healthy ~-3).
    assert rf_gpu == rf_gpu, "GPU R-free is NaN"
    assert rf_gpu < 0.6, f"GPU R-free blew up ({rf_gpu:.3f}) — scale likely collapsed"
    assert ls_gpu > -15.0, f"GPU per-bin log_scale collapsed (min={ls_gpu:.2f})"

    # (b) GPU tracks CPU — the device-parity the estimate_beta fix restores.
    assert abs(rf_gpu - rf_cpu) < 0.03, (
        f"GPU R-free {rf_gpu:.4f} diverged from CPU {rf_cpu:.4f}"
    )
