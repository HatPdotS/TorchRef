"""ADP restraint statistics describe the B values the restraints act on.

Once an atom is anisotropic its isotropic wrapper (``model.adp()``) stops being
refined, so on an anisotropic model the statistics, and the rigid-bond isotropic
proxy, must read B_eq from the unified U tensors, as the losses do.
"""

import math

import pytest
import torch

from torchref.base.targets.adp import u6_b_eq
from torchref.model.model import Model
from torchref.refinement.targets.adp import (
    ADPLocalityTarget,
    ADPSimilarityTarget,
    RigidBondTarget,
)


@pytest.fixture(scope="module")
def aniso_model(pdb_dir):
    """1DAW made anisotropic, with the U tensors stepped away from the initial B."""
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    model.set_adp_mode("anisotropic")
    params = model.u.refinable_params
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        params.add_(
            0.3 * torch.randn(params.shape, generator=generator, dtype=params.dtype)
        )
    model.u.reset_forward_cache()
    return model


def _b_eq(model: Model) -> torch.Tensor:
    return u6_b_eq(model.adp_u6()).detach()


@pytest.mark.unit
def test_simu_stats_report_b_eq_differences(aniso_model):
    target = ADPSimilarityTarget(aniso_model)
    pairs = target._get_pair_indices()
    b = _b_eq(aniso_model)
    delta = b[pairs[:, 0]] - b[pairs[:, 1]]

    expected = torch.sqrt((delta**2).mean()).item()
    assert target.stats()["rms_delta_b"].value == pytest.approx(expected, rel=1e-5)


@pytest.mark.unit
def test_locality_stats_report_log_b_eq_differences(aniso_model):
    target = ADPLocalityTarget(aniso_model)
    stats = target.stats()
    log_b = torch.log(_b_eq(aniso_model).clamp(min=1e-3))
    diff = log_b.unsqueeze(1) - log_b[target._neighbor_indices]

    expected = torch.sqrt((diff**2).mean()).item()
    assert stats["rms_deviation_log"].value == pytest.approx(expected, rel=1e-5)


@pytest.mark.unit
def test_rigid_bond_proxy_reads_b_eq(aniso_model):
    """Both the Δz statistics and the ``use_aniso=False`` loss use ΔB_eq / 8π²."""
    target = RigidBondTarget(aniso_model, use_aniso=False)
    pairs = target._bond_pairs()
    b = _b_eq(aniso_model)
    delta_z = (b[pairs[:, 0]] - b[pairs[:, 1]]) / (8.0 * math.pi**2)

    expected_rms = torch.sqrt((delta_z**2).mean()).item()
    assert target.get_delta_z_stats()["rms"] == pytest.approx(expected_rms, rel=1e-5)

    sigma = target.sigma
    nll = 0.5 * (delta_z / sigma) ** 2 + math.log(sigma) + 0.5 * math.log(2.0 * math.pi)
    assert target().item() == pytest.approx(nll.sum().item(), rel=1e-5)
