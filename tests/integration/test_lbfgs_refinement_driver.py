"""LBFGSRefinement driver behaviour on a deposited structure (1DAW)."""

import pytest
import torch

from torchref import LBFGSRefinement


@pytest.fixture(scope="module")
def build(mtz_dir, pdb_dir):
    def _build():
        return LBFGSRefinement(
            data_file=str(mtz_dir / "1DAW.mtz"),
            pdb=str(pdb_dir / "1DAW.pdb"),
            device=torch.device("cpu"),
            verbose=0,
        )

    return _build


@pytest.mark.integration
def test_load_state_restores_a_save_state_checkpoint(build, tmp_path):
    """A checkpoint loads onto a refinement built from the same files, metadata and
    all, and reproduces its coordinates, scale and R-factors exactly."""
    saved = build()
    with torch.no_grad():
        saved.model.xyz.refinable_params.add_(0.03)
        saved.scaler.c_iso.add_(0.01)
    path = tmp_path / "refinement.pt"
    saved.save_state(str(path))

    restored = build()
    restored.load_state(str(path))

    assert torch.equal(restored.model.xyz(), saved.model.xyz())
    assert torch.equal(restored.scaler.c_iso, saved.scaler.c_iso)
    assert restored.get_rfactor() == saved.get_rfactor()


@pytest.mark.integration
def test_load_state_refuses_a_checkpoint_missing_a_parameter(build, tmp_path):
    """A parameter of the refinement that the checkpoint lacks is an error."""
    ref = build()
    state = ref.state_dict()
    del state["scaler.c_iso"]
    path = tmp_path / "partial.pt"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match="scaler.c_iso"):
        ref.load_state(str(path))
