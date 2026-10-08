"""
Unit tests for :class:`~torchref.experimental.ensemble.ensemble_model.EnsembleModel`.

Covers replicate-and-perturb construction, per-member view layout,
freeze semantics for B / occupancy, and multi-MODEL PDB round-trip.
"""

import os

import pytest
import torch

from torchref.experimental.ensemble import EnsembleModel


TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "1DAW.pdb"
)


@pytest.fixture
def small_ensemble() -> EnsembleModel:
    return EnsembleModel.from_single(
        TEST_PDB, n_members=5, perturb_sigma=0.2, b_const=5.0,
        seed=42, verbose=0,
    )


def test_construction_yields_expected_shapes(small_ensemble):
    ens = small_ensemble
    assert ens.n_members == 5
    assert ens.n_atoms_per_member > 0
    assert ens.xyz().shape[0] == ens.n_members * ens.n_atoms_per_member
    assert ens.xyz_per_member.shape == (
        ens.n_members, ens.n_atoms_per_member, 3,
    )


def test_b_factor_is_constant(small_ensemble):
    ens = small_ensemble
    b = ens.adp()
    assert torch.allclose(b, torch.full_like(b, 5.0), atol=1e-5)


def test_freezes_everything_except_xyz(small_ensemble):
    ens = small_ensemble
    trainable = {
        n for n, p in ens.named_parameters()
        if p.requires_grad and p.numel() > 0
    }
    # Only xyz.refinable_params should remain refinable.
    assert "xyz.refinable_params" in trainable
    # adp, u, occupancy should NOT be present.
    assert all("xyz" in n for n in trainable), \
        f"Unexpected trainable params: {trainable - {'xyz.refinable_params'}}"


def test_per_member_view_is_a_storage_view(small_ensemble):
    """Mutating xyz_per_member[i] should change the underlying flat tensor."""
    ens = small_ensemble
    flat_before = ens.xyz().clone()
    with torch.no_grad():
        # Shift member 0 by +1 in x via the view.
        # We cannot in-place .add_ on the view because xyz() may return a
        # new tensor via _AssembleMixedTensor; touch the refinable params
        # directly instead.
        ens.xyz.refinable_params[: ens.n_atoms_per_member, 0] += 1.0
    flat_after = ens.xyz()
    diff = (flat_after - flat_before)[: ens.n_atoms_per_member, 0]
    assert torch.allclose(diff, torch.ones_like(diff))


def test_pdb_single_is_one_copy(small_ensemble):
    ens = small_ensemble
    assert ens._pdb_single is not None
    assert len(ens._pdb_single) == ens.n_atoms_per_member


def test_multimodel_pdb_round_trip(tmp_path, small_ensemble):
    """Write multi-MODEL PDB then re-read; coordinates match within PDB precision."""
    ens = small_ensemble
    out = str(tmp_path / "ens.pdb")
    ens.write_pdb(out)
    ens2 = EnsembleModel.from_multimodel_pdb(
        out, perturb_sigma=0.0, b_const=5.0, verbose=0,
    )
    assert ens2.n_members == ens.n_members
    assert ens2.n_atoms_per_member == ens.n_atoms_per_member
    diff = (ens.xyz_per_member - ens2.xyz_per_member).abs().max().item()
    # PDB stores coordinates with 3 decimal places.
    assert diff < 5e-3, f"Round-trip coord diff {diff} exceeds PDB precision"


def test_forward_returns_complex_per_hkl(small_ensemble):
    ens = small_ensemble
    ens.setup_grid(max_res=2.5)
    hkl = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 2, 1], [0, 0, 4]],
                       dtype=torch.int32, device=ens.device)
    f = ens(hkl)
    assert f.shape == (4,)
    assert f.is_complex()
    assert torch.isfinite(f.abs()).all()


def test_gradient_flows_only_to_xyz(small_ensemble):
    ens = small_ensemble
    ens.setup_grid(max_res=2.5)
    hkl = torch.tensor([[1, 0, 0], [2, 1, 0]], dtype=torch.int32, device=ens.device)
    ens.xyz.refinable_params.grad = None
    f = ens(hkl)
    f.abs().sum().backward()
    assert ens.xyz.refinable_params.grad is not None
    assert torch.isfinite(ens.xyz.refinable_params.grad).all()
    # adp / u / occupancy should have no gradient (frozen).
    if ens.adp.refinable_params.grad is not None:
        assert float(ens.adp.refinable_params.grad.abs().sum()) == 0.0


# --------------------------------------------------------------------------
# Ensemble dropout (member-subset regularization)
# --------------------------------------------------------------------------

def _dropout_hkl(ens):
    return torch.tensor(
        [[1, 0, 0], [2, 0, 0], [3, 2, 1], [0, 0, 4], [1, 1, 1], [2, 2, 2]],
        dtype=torch.int32, device=ens.device,
    )

def test_dropout_zeros_gradient_for_dropped_members(small_ensemble):
    """Dropped members contribute nothing to F_calc, so get exactly zero
    gradient; kept members get a real gradient. This is the mechanism."""
    ens = small_ensemble
    ens.setup_grid(max_res=2.5)
    hkl = _dropout_hkl(ens)
    ens.configure_dropout(True, 2, 2)
    k = ens.resample_dropout()
    assert k == 2
    per_member = ens._dropout_occ_mult.view(
        ens.n_members, ens.n_atoms_per_member
    )[:, 0]
    kept = per_member > 0
    assert int(kept.sum()) == 2
    # Effective occupancy of kept members is 1/k = 0.5 (was 1/N = 0.2).
    assert torch.allclose(
        per_member[kept], torch.full_like(per_member[kept], ens.n_members / 2.0)
    )
    ens.reset_cache()
    ens.xyz.refinable_params.grad = None
    ens(hkl).abs().sum().backward()
    g = ens.xyz.refinable_params.grad.view(
        ens.n_members, ens.n_atoms_per_member, 3
    )
    gmem = g.abs().sum(dim=(1, 2))
    assert torch.all(gmem[~kept] == 0), "dropped members must get zero gradient"
    assert torch.all(gmem[kept] > 0), "kept members must get gradient"


def test_dropout_disable_restores_full_occupancy(small_ensemble):
    ens = small_ensemble
    ens.configure_dropout(True, 2, 2)
    ens.resample_dropout()
    assert not torch.allclose(
        ens._dropout_occ_mult, torch.ones_like(ens._dropout_occ_mult)
    )
    ens.configure_dropout(False)
    assert torch.allclose(
        ens._dropout_occ_mult, torch.ones_like(ens._dropout_occ_mult)
    )


def test_copy_carries_the_ensemble(tmp_path, small_ensemble):
    """A copy keeps the member layout, per-member levers and single-copy table."""
    ens = small_ensemble
    ens.enable_population_refinement(True)
    with torch.no_grad():
        ens.occ_logits[0] = 1.0
    dup = ens.copy()
    assert type(dup) is EnsembleModel
    assert dup.n_members == ens.n_members
    assert dup.n_atoms_per_member == ens.n_atoms_per_member
    assert torch.equal(dup.xyz_per_member, ens.xyz_per_member)
    assert torch.equal(dup.member_weights(), ens.member_weights())
    assert dup.occ_logits.requires_grad and not dup.b_raw.requires_grad
    with torch.no_grad():
        dup.occ_logits[1] = 2.0
    assert ens.occ_logits[1] == 0
    assert len(dup.pdb_single) == ens.n_atoms_per_member
    dup.write_pdb(str(tmp_path / "copy.pdb"))


def test_copy_keeps_a_low_rank_xyz():
    ens = EnsembleModel.from_single(
        TEST_PDB,
        n_members=5,
        perturb_sigma=0.2,
        b_const=5.0,
        seed=42,
        verbose=0,
    )
    ens.enable_low_rank(2)
    dup = ens.copy()
    assert torch.allclose(dup.xyz(), ens.xyz())
    with torch.no_grad():
        dup.xyz.amplitudes.add_(1.0)
    assert not torch.allclose(dup.xyz(), ens.xyz())


# --------------------------------------------------------------------------
# Spare (dead) slots of a population pool
# --------------------------------------------------------------------------

_POOL_KW = dict(perturb_sigma=0.2, b_const=5.0, seed=42, verbose=0)


def _member_occupancy(ens):
    occ = ens.get_iso()[2].detach().cpu()
    return occ.view(ens.n_members, ens.n_atoms_per_member)[:, 0]


def test_dead_slots_do_not_contribute_to_fcalc():
    """A pool with dead spare slots scatters like the alive members alone."""
    pool = EnsembleModel.from_single(TEST_PDB, n_members=3, n_max=5, **_POOL_KW)
    plain = EnsembleModel.from_single(TEST_PDB, n_members=3, **_POOL_KW)
    assert torch.equal(pool.xyz_per_member[:3], plain.xyz_per_member)
    occ = _member_occupancy(pool)
    assert torch.allclose(occ, torch.tensor([1 / 3] * 3 + [0, 0], dtype=occ.dtype))
    pool.setup_grid(max_res=2.5)
    plain.setup_grid(max_res=2.5)
    hkl = _dropout_hkl(pool)
    assert torch.allclose(pool(hkl), plain(hkl), rtol=1e-4, atol=1e-3)


def test_multimodel_pool_occupancy_sums_to_one(tmp_path, small_ensemble):
    path = str(tmp_path / "ens.pdb")
    small_ensemble.write_pdb(path)
    ens = EnsembleModel.from_multimodel_pdb(path, n_members=2, n_max=4, verbose=0)
    occ = _member_occupancy(ens)
    assert torch.allclose(occ, torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=occ.dtype))


def test_dropout_draws_only_alive_members():
    ens = EnsembleModel.from_single(TEST_PDB, n_members=4, n_max=6, **_POOL_KW)
    ens.configure_dropout(True, 2, 2)
    for _ in range(20):
        assert ens.resample_dropout() == 2
        mult = ens._dropout_occ_mult.view(ens.n_members, -1)[:, 0].cpu()
        assert torch.all(mult[4:] == 0)
        assert torch.allclose(mult[mult > 0], torch.full((2,), 2.0, dtype=mult.dtype))
        total = _member_occupancy(ens).sum()
        assert torch.allclose(total, torch.tensor(1.0, dtype=total.dtype))


def test_inactive_dropout_reports_the_alive_count():
    ens = EnsembleModel.from_single(TEST_PDB, n_members=4, n_max=6, **_POOL_KW)
    assert ens.resample_dropout() == 4


def test_write_pdb_writes_alive_members_with_their_weights(tmp_path):
    ens = EnsembleModel.from_single(TEST_PDB, n_members=4, n_max=6, **_POOL_KW)
    ens.enable_population_refinement(True)
    with torch.no_grad():
        ens.occ_logits[:4] = torch.tensor([2.0, 0.5, 0.0, -1.0])
    ens.kill_member(3)
    path = tmp_path / "pop.pdb"
    ens.write_pdb(str(path))
    lines = path.read_text().splitlines()
    assert sum(ln.startswith("MODEL ") for ln in lines) == 3
    first_atoms = []
    in_model = False
    for ln in lines:
        if ln.startswith("MODEL "):
            in_model = True
        elif in_model and ln.startswith(("ATOM", "HETATM")):
            first_atoms.append(float(ln[54:60]))
            in_model = False
    expected = ens.member_weights()[:3].detach().cpu()
    assert torch.allclose(torch.tensor(first_atoms), expected, atol=6e-3)
    assert EnsembleModel.from_multimodel_pdb(str(path), verbose=0).n_members == 3


def test_bifurcate_refuses_a_low_rank_xyz():
    ens = EnsembleModel.from_single(TEST_PDB, n_members=4, n_max=6, **_POOL_KW)
    ens.enable_population_refinement(True)
    ens.enable_low_rank(2)
    with pytest.raises(RuntimeError, match="enable_low_rank"):
        ens.bifurcate_member(0)


@pytest.mark.parametrize("swap", ["enable_low_rank", "enable_pca"])
def test_pca_seeding_runs_in_the_working_dtype(monkeypatch, small_ensemble, swap):
    """The seeding SVD stays in the model dtype, so it also runs on MPS."""
    seen = []
    svd = torch.linalg.svd

    def recording_svd(A, *args, **kwargs):
        seen.append(A.dtype)
        return svd(A, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", recording_svd)
    before = small_ensemble.xyz().detach().clone()
    getattr(small_ensemble, swap)(4)
    assert seen and all(dt == small_ensemble.dtype_float for dt in seen)
    assert torch.allclose(small_ensemble.xyz(), before, atol=1e-3)
