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

    # The anisotropic ``u`` must round-trip with the SAME parametrization as a
    # freshly-loaded model: CholeskyMixedTensor (positive-definite by
    # construction), not a plain MixedTensor. Guards the state-dict bug where
    # a restored model refined ``u`` in raw space and could go indefinite.
    # (Value equivalence is exercised on an ANISOU structure below; 1DAW is
    # isotropic so u() is all-NaN for both and only the type is meaningful.)
    from torchref.model import CholeskyMixedTensor

    assert isinstance(model.u, CholeskyMixedTensor)
    assert isinstance(restored.u, CholeskyMixedTensor)


@pytest.mark.unit
@pytest.mark.parametrize("cls_name", ["Model", "ModelFT"])
def test_create_from_state_dict_aniso_u_roundtrip(pdb_dir, cls_name):
    """Anisotropic ``u`` round-trips as a CholeskyMixedTensor with matching values.

    Uses an ANISOU structure (7L84) so ``u()`` is finite and meaningful. Before
    the fix, ``create_from_state_dict`` rebuilt ``u`` as a plain ``MixedTensor``,
    reinterpreting the stored Cholesky factors as raw U components and diverging
    from a freshly-loaded model.
    """
    import torchref.model as M

    cls = getattr(M, cls_name)
    cpu = torch.device("cpu")
    model = cls()
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    model.to(cpu)
    sd = model.state_dict()

    restored = cls.create_from_state_dict(sd, device=cpu, verbose=0)

    assert isinstance(model.u, M.CholeskyMixedTensor)
    assert isinstance(restored.u, M.CholeskyMixedTensor)

    u_fresh = model.u().detach()
    u_restored = restored.u().detach()
    # At least some atoms are anisotropic → finite, non-trivial u values present.
    assert torch.isfinite(u_fresh).any()
    assert torch.allclose(u_fresh, u_restored, equal_nan=True)


@pytest.mark.unit
def test_altloc_pairs_survive_state_dict_round_trip(pdb_dir):
    """``altloc_pairs`` must reach the state dict and come back.

    It lives on the model's context rather than the model, so a defensive
    ``hasattr(self, "altloc_pairs")`` in ``state_dict`` silently substituted an empty
    list -- losing the alternative-conformation grouping on every save without
    failing anything.
    """
    from torchref.model import ModelFT

    cpu = torch.device("cpu")
    model = ModelFT()
    model.load_pdb(str(pdb_dir / "7L84.pdb"))  # carries alternative conformations
    model.to(cpu)

    assert model.ctx.altloc_pairs, "fixture should have alternative conformations"

    sd = model.state_dict()
    assert sd["altloc_pairs"], "altloc groups must reach the state dict"

    restored = ModelFT.create_from_state_dict(sd, device=cpu, verbose=0)

    assert len(restored.ctx.altloc_pairs) == len(model.ctx.altloc_pairs)
    for got, want in zip(restored.ctx.altloc_pairs, model.ctx.altloc_pairs):
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert torch.equal(g, w)
