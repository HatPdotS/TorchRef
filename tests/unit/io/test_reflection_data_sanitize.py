"""Regression test: sanitize_F must flag +/-Inf (not just NaN).

sanitize_F used torch.isnan(...) only, so +Inf/-Inf in F or F_sigma passed
through unmasked and poisoned downstream losses/gradients. It now uses
~torch.isfinite(...). See TORCHREF_AUDIT.md.
"""

import warnings

import pytest
import torch

from torchref.io import ReflectionData


@pytest.mark.unit
def test_sanitize_F_masks_nan_inf_and_nonpositive():
    data = ReflectionData()
    data.device = torch.device("cpu")
    data.verbose = 0
    data.masks = {}
    inf = float("inf")
    nan = float("nan")
    #               valid  +Inf   -Inf   NaN    neg    finite-F/Inf-sigma
    data.F = torch.tensor([1.0, inf, -inf, nan, -2.0, 5.0])
    data.F_sigma = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, inf])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-positive-F warning is expected
        data.sanitize_F()

    valid = data.masks["sanity_F"]
    # Only index 0 (finite F>0 with finite sigma) survives.
    assert valid.tolist() == [True, False, False, False, False, False]
    # Every invalid entry is zeroed in both F and F_sigma (no NaN/Inf leak).
    assert torch.isfinite(data.F).all()
    assert torch.isfinite(data.F_sigma).all()
    assert (data.F[~valid] == 0).all()
    assert (data.F_sigma[~valid] == 0).all()
