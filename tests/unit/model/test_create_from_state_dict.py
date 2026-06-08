"""
Round-trip tests for ModelFT.create_from_state_dict.

Guards the restore path that previously used stale ``b``/``b_mask`` keys
instead of ``adp``/``adp_mask``.
"""
import pytest
import torch


@pytest.mark.unit
def test_modelft_create_from_state_dict_round_trip(pdb_dir):
    from torchref.model import ModelFT

    cpu = torch.device("cpu")
    model = ModelFT()
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    model.to(cpu)
    sd = model.state_dict()

    restored = ModelFT.create_from_state_dict(sd, device=cpu, verbose=0)

    # Correct attribute name (adp, not the stale b) and same atom count.
    assert hasattr(restored, "adp")
    assert not hasattr(restored, "b")
    assert len(restored.pdb) == len(model.pdb)

    # ADP values survive the round trip.
    assert torch.allclose(model.adp().detach(), restored.adp().detach())
