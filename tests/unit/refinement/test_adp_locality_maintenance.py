"""ADPLocalityTarget keeps its k-NN list current through ``maintenance()``.

LossState calls ``maintenance()`` after every optimizer step, and drivers that never
collect statistics rely on it alone, so moved coordinates or a changed ``k_neighbors``
must rebuild the list there, while unchanged coordinates must leave it alone.
"""

import pytest
import torch

from torchref.model.model import Model
from torchref.refinement.targets.adp import ADPLocalityTarget


@pytest.mark.unit
def test_maintenance_rebuilds_the_neighbour_list_after_a_step(pdb_dir):
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    target = ADPLocalityTarget(model)
    target()

    params = model.xyz.refinable_params
    generator = torch.Generator().manual_seed(0)
    step = torch.randn(params.shape, generator=generator, dtype=params.dtype)
    step = step.to(params.device)
    with torch.no_grad():
        params.add_(step / step.norm(dim=-1, keepdim=True))
    model.xyz.reset_forward_cache()

    target.maintenance()

    fresh = ADPLocalityTarget(model)
    fresh()
    assert torch.equal(target._neighbor_indices, fresh._neighbor_indices)
    assert torch.equal(target._neighbor_distances, fresh._neighbor_distances)

    built = target._neighbor_indices
    target.maintenance()
    target.stats()
    assert target._neighbor_indices is built


@pytest.mark.unit
def test_k_neighbors_setter_rebuilds_the_neighbour_list(pdb_dir):
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    target = ADPLocalityTarget(model)
    target.stats()
    target.k_neighbors = 10

    stats = target.stats()
    assert target._neighbor_indices.shape[1] == 10
    assert stats["k_neighbors"].value == 10
