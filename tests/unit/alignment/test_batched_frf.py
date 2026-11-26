"""The batched FRF expansion and maps against the single-system engine.

The batched path is selected automatically on CUDA float32, so it has to give the
single-system answer: the same coefficients as ``bessel_sh_expand`` for every set in
a batch, and the same maps as ``build_dense_map_per_beta``.

Agreement of the fused kernels is to single precision, not bit for bit: they evaluate
the Legendre factor at each reflection's own polar angle rather than at a cluster's
mean, and sum in a different order.
"""

import math

import pytest
import torch

from torchref.experimental.alignment.frf.batched import (
    ExpansionPlan,
    batched_bessel_sh_expand,
    batched_cross_correlate_xi,
    batched_rotation_maps,
    rotation_map_grid,
)
from torchref.experimental.alignment.frf.data_mr import (
    bessel_sh_expand,
    cross_correlate_xi,
)
from torchref.experimental.alignment.frf.sitelist_ang import build_dense_map_per_beta
from torchref.experimental.alignment.frf.types import BesselSHCoefficients

pytestmark = pytest.mark.unit

#: The fused path's agreement with the single-set expansion, as a fraction of the
#: largest coefficient. Measured below 5e-5 on 1DAW/3K7M/4BX9, and that is the
#: single-set path's own error: against a float64 expansion the fused path is the more
#: accurate of the two (its Bessel table normalises on j_0 or j_1, not j_0 alone). A
#: dropped or misplaced term is orders larger.
_TOL = 2e-4


def _box_set(n, a, seed):
    """A P1-box-like set: integer lattice points scaled by 1/a, so shells repeat."""
    g = torch.Generator().manual_seed(seed)
    r = torch.arange(-n, n + 1)
    h, k, l = torch.meshgrid(r, r, r, indexing="ij")
    hkl = torch.stack([h.flatten(), k.flatten(), l.flatten()], -1).to(torch.float32)
    s = hkl / a
    smag = s.norm(dim=-1)
    s = s[(smag > 0.02) & (smag < 0.25)]
    I = torch.randn(s.shape[0], generator=g, dtype=torch.float32)
    return s, I


def _random_set(m, seed):
    """Scattered directions and radii: every reflection its own shell."""
    g = torch.Generator().manual_seed(seed)
    smag = 0.03 + 0.2 * torch.rand(m, generator=g)
    v = torch.randn(m, 3, generator=g)
    s = v / v.norm(dim=-1, keepdim=True) * smag.unsqueeze(-1)
    return s, torch.randn(m, generator=g)


def _batch(device):
    sets = [_box_set(10, 60.0, 0), _random_set(3000, 1), _box_set(14, 90.0, 2)]
    s_list = [s.to(device) for s, _ in sets]
    I_list = [i.to(device) for _, i in sets]
    return s_list, I_list


@pytest.mark.cuda
@pytest.mark.parametrize("L", [5, 17, 34, 65])
def test_fused_expansion_matches_single_set(L):
    """Every set in a batch gets bessel_sh_expand's coefficients, m-filter included."""
    s_list, I_list = _batch("cuda")
    scales = [(L - 1) * 4.0, (L - 1) * 3.5, (L - 1) * 5.0]
    zsymm = [1, 2, 4]
    plan = ExpansionPlan(s_list, L=L, bessel_h_scale=scales)
    fused = plan.expand(I_list, zsymm=zsymm, force_portable=False)
    for k, (s, i) in enumerate(zip(s_list, I_list)):
        ref = bessel_sh_expand(
            s, i, L=L, bessel_h_scale=scales[k], zsymm=zsymm[k]
        ).coeffs
        err = float((fused[k] - ref).abs().max() / ref.abs().max())
        assert err < _TOL, f"set {k}, L={L}: {err:.2e}"


@pytest.mark.cuda
def test_plan_is_reusable_and_linear():
    """A plan expands new intensities on its geometry; the expansion is linear in I."""
    s_list, I_list = _batch("cuda")
    plan = ExpansionPlan(s_list, L=17, bessel_h_scale=60.0)
    c1 = plan.expand(I_list, force_portable=False)
    c2 = plan.expand([2.0 * i for i in I_list], force_portable=False)
    c12 = plan.expand([i + 2.0 * i for i in I_list], force_portable=False)
    torch.testing.assert_close(
        c2, 2.0 * c1, rtol=1e-5, atol=1e-6 * float(c1.abs().max())
    )
    torch.testing.assert_close(
        c12, c1 + c2, rtol=1e-4, atol=1e-5 * float(c12.abs().max())
    )
    one_shot = batched_bessel_sh_expand(
        s_list, I_list, L=17, bessel_h_scale=60.0, force_portable=False
    )
    torch.testing.assert_close(one_shot, c1)


def test_portable_path_is_the_single_set_expansion():
    """Off CUDA (or forced), each set is exactly bessel_sh_expand."""
    s_list, I_list = _batch("cpu")
    plan = ExpansionPlan(s_list, L=9, bessel_h_scale=[30.0, 31.0, 32.0])
    c = plan.expand(I_list, zsymm=[1, 3, 1], force_portable=True)
    for k in range(3):
        ref = bessel_sh_expand(
            s_list[k], I_list[k], L=9, bessel_h_scale=30.0 + k, zsymm=[1, 3, 1][k]
        ).coeffs
        assert torch.equal(c[k], ref)


def test_plan_rejects_mismatched_intensities():
    s_list, I_list = _batch("cpu")
    plan = ExpansionPlan(s_list, L=9, bessel_h_scale=30.0)
    with pytest.raises(ValueError):
        plan.expand(I_list[:2])
    with pytest.raises(ValueError):
        plan.expand([I_list[0], I_list[0], I_list[2]])


def test_maps_match_the_single_system_dense_map():
    """batched_rotation_maps is build_dense_map_per_beta, system by system (fp64)."""
    L = 9
    g = torch.Generator().manual_seed(3)
    c = torch.randn(4, 4, L, 2 * L - 1, generator=g, dtype=torch.complex128)
    c[:, :, 1::2] = 0.0
    c[:, :, 0] = 0.0
    xi = batched_cross_correlate_xi(c[:2], c[2:])
    maps, _ = batched_rotation_maps(xi, grid_sampling_deg=10.0)
    betas, N = rotation_map_grid(L, 10.0)
    assert maps.shape[-1] == N
    for b in range(2):
        xi_ref = cross_correlate_xi(
            BesselSHCoefficients(c[b], L, 0.0), BesselSHCoefficients(c[2 + b], L, 0.0)
        )
        torch.testing.assert_close(xi[b], xi_ref)
        ref = build_dense_map_per_beta(xi_ref, betas, N).real
        torch.testing.assert_close(
            maps[b], ref, rtol=1e-10, atol=1e-10 * float(ref.abs().max())
        )
    assert maps.shape[1] == math.ceil(180.0 / 10.0)


def test_xi_shares_observations_by_index():
    """obs_index lets several models share one observed expansion."""
    g = torch.Generator().manual_seed(4)
    c_obs = torch.randn(2, 3, 5, 9, generator=g, dtype=torch.complex128)
    c_calc = torch.randn(3, 3, 5, 9, generator=g, dtype=torch.complex128)
    idx = torch.tensor([1, 0, 1])
    xi = batched_cross_correlate_xi(c_obs, c_calc, obs_index=idx)
    for b in range(3):
        torch.testing.assert_close(
            xi[b], batched_cross_correlate_xi(c_obs[idx[b]][None], c_calc[b][None])[0]
        )
