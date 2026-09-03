"""``set_adp_mode("field_aniso", mode_set=...)``: installing a displacement-mode field.

The wiring, not the payload arithmetic --- that is
``test_mode_covariance_payload.py``. What matters here is that a mode set reaches the
model through the existing switch, lands in the ``u`` slot like any anisotropic field,
and starts where the constant-U field starts, so entering the parametrisation is not
itself a change to the model.
"""

import pytest
import torch

from torchref.model.disorder_field import (
    MODE_SETS,
    DisorderFieldTensor,
    ModeCovariancePayload,
)
from torchref.model.model import Model


@pytest.fixture(scope="module")
def pdb_path(pdb_dir):
    return str(pdb_dir / "3GR5.pdb")


def _field_model(pdb_path, mode_set=None, n_nodes=8):
    model = Model(verbose=0)
    model.load_pdb(pdb_path)
    model.set_adp_mode(
        "field_aniso", n_nodes=n_nodes, k_neighbors=n_nodes, mode_set=mode_set
    )
    return model


@pytest.mark.unit
@pytest.mark.parametrize("mode_set", sorted(MODE_SETS))
def test_mode_set_installs_into_the_u_slot(pdb_path, mode_set):
    """A mode field is an anisotropic field: same slot, same downstream consumers."""
    model = _field_model(pdb_path, mode_set)
    assert model.adp_is_field
    assert isinstance(model.u, DisorderFieldTensor)
    assert isinstance(model.u.payload, ModeCovariancePayload)
    assert model.u.payload.mode_set == mode_set
    assert bool(model.aniso_flag.all())
    # The per-atom surface the structure-factor path uses.
    u6 = model.adp_u6()
    assert u6.shape == (len(model.pdb), 6)
    assert torch.isfinite(u6).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_set,per_node", [("constant", 10), ("rigid", 25), ("rigid_dilation", 32), ("affine", 82)]
)
def test_parameter_count_is_payload_plus_sigma_plus_offset(pdb_path, mode_set, per_node):
    """Storage is [payload | log sigma | 3 offset], so the ladder costs 10/25/32/82."""
    model = _field_model(pdb_path, mode_set, n_nodes=8)
    assert model.u.refinable_params.numel() == 8 * per_node


@pytest.mark.unit
@pytest.mark.parametrize("mode_set", sorted(MODE_SETS))
def test_entering_the_mode_starts_at_the_constant_u_field(pdb_path, mode_set):
    """Every rung seeds its translation block from the same solve and floors the rest.

    So a freshly installed mode field must give essentially the constant-U field's ADPs.
    That is what makes entering this parametrisation safe: the starting R-factor is one
    already known, and refinement can only move away from it.
    """
    base = _field_model(pdb_path, "constant").adp_u6().detach()
    got = _field_model(pdb_path, mode_set).adp_u6().detach()
    # Gradient modes start at the Cholesky floor, epsilon^2, which is ~1e-6 A^2 against
    # a U of order 0.2 -- present but far below anything observable.
    assert torch.allclose(got, base, atol=1e-5)


@pytest.mark.unit
def test_positive_definite_per_atom(pdb_path):
    """Every atom's U must be PD or the anisotropic B-matrix inverse blows up."""
    model = _field_model(pdb_path, "affine", n_nodes=6)
    u6 = model.adp_u6().detach()
    M = torch.zeros(u6.shape[0], 3, 3, dtype=u6.dtype)
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = u6[:, 0], u6[:, 1], u6[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = u6[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = u6[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = u6[:, 5]
    assert float(torch.linalg.eigvalsh(M).min()) > 0.0


@pytest.mark.unit
def test_gradient_reaches_the_node_parameters(pdb_path):
    """Through the zero-argument forward, which is the path refinement actually uses."""
    model = _field_model(pdb_path, "rigid")
    model.adp_u6().sum().backward()
    grad = model.u.refinable_params.grad
    assert grad is not None and float(grad.abs().sum()) > 0


@pytest.mark.unit
def test_copy_round_trips_a_mode_field(pdb_path):
    """``Model.copy`` shares no storage but must keep the payload and the accessor."""
    model = _field_model(pdb_path, "rigid_dilation")
    clone = model.copy()
    assert isinstance(clone.u.payload, ModeCovariancePayload)
    assert clone.u.payload.mode_set == "rigid_dilation"
    assert torch.allclose(clone.adp_u6().detach(), model.adp_u6().detach())
    assert clone.u.refinable_params is not model.u.refinable_params
    # The accessor must point at the COPY's coordinates, not the original's. The
    # perturbation has to be non-rigid: a field whose nodes are atom centroids is
    # translation-invariant by construction, so shifting every atom would change
    # nothing and prove nothing.
    original = model.adp_u6().detach().clone()
    with torch.no_grad():
        clone.xyz.refinable_params[: len(clone.pdb) // 2] += 3.0
    clone.u.reset_forward_cache()
    assert not torch.allclose(clone.adp_u6().detach(), original)
    # ...and the original must be untouched by it.
    model.u.reset_forward_cache()
    assert torch.allclose(model.adp_u6().detach(), original)


@pytest.mark.unit
def test_mode_set_is_rejected_on_the_isotropic_field(pdb_path):
    """There is no isotropic form of a displacement-mode covariance."""
    model = Model(verbose=0)
    model.load_pdb(pdb_path)
    with pytest.raises(ValueError, match="no isotropic form"):
        model.set_adp_mode("field", n_nodes=8, mode_set="rigid")


@pytest.mark.unit
def test_leaving_field_mode_materialises_per_atom(pdb_path):
    """The conversion out reads the per-atom U, so it works for any payload."""
    model = _field_model(pdb_path, "affine", n_nodes=6)
    before = model.adp_u6().detach().clone()
    model.set_adp_mode("anisotropic")
    assert not model.adp_is_field
    kept = model.aniso_flag
    assert torch.allclose(model.adp_u6().detach()[kept], before[kept], atol=1e-6)
