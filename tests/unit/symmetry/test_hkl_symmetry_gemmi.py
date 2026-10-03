"""Regression tests for reciprocal-space (Miller-index) symmetry against gemmi.

These guard the ``h' = h·R = Rᵀ·h`` reciprocal-space convention used by
``Symmetry.expand_reciprocal``. A previous bug applied the real-space transform
``R·h`` instead, which is only correct when the fractional rotation matrix is
symmetric. For space groups whose rotation matrices are non-symmetric
(trigonal, hexagonal, and permutation-type cubic operations) ``R·h`` produced
the wrong set of symmetry equivalents, corrupting the centric flags
(``Symmetry.is_centric``) and epsilon multiplicities (``Symmetry.epsilon``)
that feed French-Wilson intensity conversion and ML sigma_A weighting.

Ground truth comes from gemmi:
- transform:   ``gemmi.Op.apply_to_hkl``
- centric:     ``GroupOps.is_reflection_centric``
- epsilon:     ``GroupOps.epsilon_factor_without_centering`` (Friedel-doubled
               for centric reflections, matching the Friedel-aware count in
               ``Symmetry.epsilon``)
"""

import numpy as np
import pytest
import torch

from torchref.config import get_default_device, get_float_dtype, get_int_dtype

# Space groups whose rotation matrices are non-symmetric — these are the ones
# that regress if expand_reciprocal uses R·h instead of Rᵀ·h. Plus orthorhombic /
# tetragonal controls (symmetric matrices) that must remain correct either way.
_TRIGONAL_HEXAGONAL = ["P 32 2 1", "P 31 2 1", "P 61 2 2", "P 6 2 2", "P 3 1 2"]
_CONTROLS = ["P 21 21 21", "P 43 21 2", "P 1"]
_SPACE_GROUPS = _TRIGONAL_HEXAGONAL + _CONTROLS

# A spread of reflections including negative indices and axial/general cases.
_HKLS = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (2, 1, 0), (1, -1, 0),
    (1, 2, 3), (0, 1, 2), (3, 1, 0), (0, 0, 2), (1, 0, 1), (-1, 2, -3),
    (2, 2, 1), (1, 1, 1),
]


def _hkl_tensor():
    """Miller indices on the configured default device and integer dtype.

    Kept at the config defaults rather than a hardcoded float64/CPU so the test
    exercises the same precision and device production does, and stays
    MPS-compatible (MPS has no float64). The arithmetic here is exact in
    float32 regardless: fractional rotation matrices are integer-valued
    (gemmi's ``op.rot / 24``) and the Miller indices are small, so every
    product and sum is well inside the exactly-representable integer range.
    """
    return torch.tensor(_HKLS, dtype=get_int_dtype(), device=get_default_device())


@pytest.mark.unit
@pytest.mark.parametrize("sg_name", _SPACE_GROUPS)
def test_expand_reciprocal_matches_gemmi_transform(sg_name):
    """expand_reciprocal must reproduce gemmi's per-operation reciprocal transform.

    Compared as the *set* of equivalents per reflection so the test is
    insensitive to operation ordering between torchref and gemmi.
    """
    import gemmi

    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup(sg_name)
    hkl = _hkl_tensor()

    # torchref: (ops, N, 3) -> per-reflection set of equivalents
    out_int = sg.expand_reciprocal(hkl).cpu().numpy()  # (ops, N, 3)

    gemmi_ops = list(gemmi.SpaceGroup(sg_name).operations().sym_ops)

    for n, h in enumerate(_HKLS):
        got = {tuple(out_int[o, n, :]) for o in range(out_int.shape[0])}
        expected = {tuple(op.apply_to_hkl(h)) for op in gemmi_ops}
        assert got == expected, (
            f"{sg_name} hkl={h}: expand_reciprocal equivalents {sorted(got)} "
            f"!= gemmi {sorted(expected)}"
        )


@pytest.mark.unit
@pytest.mark.parametrize("sg_name", _SPACE_GROUPS)
def test_is_centric_matches_gemmi(sg_name):
    """Centric flags must match gemmi's is_reflection_centric."""
    import gemmi

    from torchref.symmetry import SpaceGroup

    ops = gemmi.SpaceGroup(sg_name).operations()
    hkl = _hkl_tensor()

    got = SpaceGroup(sg_name).is_centric(hkl).cpu().numpy().astype(bool)
    expected = np.array([ops.is_reflection_centric(h) for h in _HKLS], dtype=bool)

    assert np.array_equal(got, expected), (
        f"{sg_name}: centric {got.astype(int)} != gemmi {expected.astype(int)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("sg_name", _SPACE_GROUPS)
def test_epsilon_from_hkl_matches_gemmi(sg_name):
    """Friedel-aware epsilon must match gemmi's epsilon (doubled for centrics)."""
    import gemmi

    from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl
    from torchref.symmetry import SpaceGroup

    ops = gemmi.SpaceGroup(sg_name).operations()
    sg = SpaceGroup(sg_name)
    hkl = _hkl_tensor()

    got = epsilon_from_hkl(hkl, sg).cpu().numpy().round().astype(int)
    expected = np.array(
        [
            ops.epsilon_factor_without_centering(h)
            * (2 if ops.is_reflection_centric(h) else 1)
            for h in _HKLS
        ],
        dtype=int,
    )

    assert np.array_equal(got, expected), (
        f"{sg_name}: epsilon {got} != gemmi-derived {expected}"
    )


@pytest.mark.unit
def test_expand_reciprocal_is_transpose_not_plain_rotation():
    """Explicit guard on the exact bug: apply_to_hkl == Rᵀ·h, not R·h.

    Uses P3, whose 3-fold rotation matrix is non-symmetric, so R·h and Rᵀ·h
    give genuinely different results.
    """
    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup("P 3")
    # Use the SpaceGroup's own dtype (config default) -- no hardcoded float64,
    # so this runs on MPS as well as CPU/CUDA.
    mats = sg.matrices  # (ops, 3, 3)
    hkl = torch.tensor([[1, 2, 3]], dtype=mats.dtype, device=mats.device)

    out = sg.expand_reciprocal(hkl)  # (ops, 1, 3)

    # Correct reciprocal transform: Rᵀ·h
    expected_t = torch.einsum("oji,nj->oni", mats, hkl)
    # The buggy real-space transform: R·h
    plain_r = torch.einsum("oij,nj->oni", mats, hkl)

    assert torch.allclose(out.to(mats.dtype), expected_t), (
        "expand_reciprocal should compute Rᵀ·h"
    )
    # For P3 the two conventions must actually differ (non-symmetric matrix),
    # otherwise this test would not detect a regression.
    assert not torch.allclose(expected_t, plain_r), (
        "P3 rotation matrices should be non-symmetric; test cannot detect the bug"
    )
