"""Regression test: OccupancyTensor.set() must not corrupt collapsed-logit storage.

OccupancyTensor stores collapsed logits and overrides _set_values to convert
[0,1] -> logit -> collapse. MixedTensor.set() used to inline the base logic
(writing per-atom occupancies straight into the collapsed-logit buffer), so a
following forward() double-applied the sigmoid (e.g. set 0.5 -> read ~0.62).
set() now delegates to _set_values. See TORCHREF_AUDIT.md.
"""

import pytest
import torch

from torchref.model.parameter_wrappers import OccupancyTensor


@pytest.mark.unit
def test_occupancy_set_roundtrips_no_double_sigmoid():
    occ = OccupancyTensor(
        initial_values=torch.tensor([1.0, 1.0, 1.0, 1.0]), name="occupancy"
    )
    mask = torch.tensor([True, True, False, False])
    occ.set(torch.tensor([0.5, 0.7]), mask)

    out = occ().detach().cpu()
    # Pre-fix: sigmoid([0.5,0.7,...]) ~= [0.62, 0.67, ...]. Must be exact.
    assert torch.allclose(out, torch.tensor([0.5, 0.7, 1.0, 1.0]), atol=1e-4)


@pytest.mark.unit
def test_occupancy_set_matches_setitem():
    a = OccupancyTensor(
        initial_values=torch.tensor([0.9, 0.8, 0.7, 0.6]), name="occupancy"
    )
    b = OccupancyTensor(
        initial_values=torch.tensor([0.9, 0.8, 0.7, 0.6]), name="occupancy"
    )
    mask = torch.tensor([False, True, True, False])
    vals = torch.tensor([0.3, 0.4])

    a.set(vals, mask)
    b[mask] = vals  # the already-correct __setitem__ -> _set_values path

    assert torch.allclose(a().detach().cpu(), b().detach().cpu(), atol=1e-6)


@pytest.mark.unit
def test_occupancy_set_shared_group():
    occ = OccupancyTensor(
        initial_values=torch.tensor([0.8, 0.8, 0.9, 1.0]),
        sharing_groups=torch.tensor([0, 0, 1, 2]),
        name="occupancy",
    )
    mask = torch.tensor([True, True, False, False])  # the shared group
    occ.set(torch.tensor([0.4, 0.4]), mask)

    out = occ().detach().cpu()
    assert torch.allclose(out[:2], torch.tensor([0.4, 0.4]), atol=1e-4)
    assert torch.allclose(out[2:], torch.tensor([0.9, 1.0]), atol=1e-4)  # untouched
