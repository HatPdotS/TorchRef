"""The batched translation-function maps against the single-candidate map.

``batched_translation_maps`` is selected automatically on CUDA float32, so for every
system and orientation in a batch it has to return the map
``fast_translation_function`` returns for that candidate alone, on either a shared
grid or each system's own. The CUDA path accumulates with float32 atomics, so
agreement is to float32 rounding.
"""

from types import SimpleNamespace

import pytest
import torch

from torchref.experimental.alignment.translation import (
    _grid_sizes,
    fast_translation_function,
)
from torchref.experimental.alignment.translation_batched import (
    batched_translation_maps,
    rotated_indices,
    shared_grid_shape,
)
from torchref.symmetry.cell import Cell
from torchref.symmetry.spacegroup import SpaceGroup

pytestmark = pytest.mark.unit

#: float32 accumulation order (atomics vs index_add_), as a fraction of the map's
#: largest value. Measured below 2e-6 on 1DAW/3K7M/4BX9.
_TOL = 1e-5


def _system(sg, cell, n, R, seed, device):
    g = torch.Generator().manual_seed(seed)
    cell = Cell(cell, device="cpu")
    spacegroup = SpaceGroup(sg, device="cpu")
    r = torch.arange(-8, 9)
    h, k, l = torch.meshgrid(r, r, r, indexing="ij")
    hkl = torch.stack([h.flatten(), k.flatten(), l.flatten()], -1)
    hkl = hkl[hkl.abs().sum(-1) > 0]
    hkl = hkl[torch.randperm(hkl.shape[0], generator=g)[:n]]
    S = int(spacegroup.matrices.shape[0])
    G = torch.complex(
        torch.randn(R, S, n, generator=g), torch.randn(R, S, n, generator=g)
    )
    coeff = torch.randn(n, generator=g)
    hR = rotated_indices(hkl, spacegroup)
    return {
        "G": G.to(device),
        "hR": hR.to(device),
        "coeff": coeff.to(device),
        "cell": cell,
    }


def _systems(device):
    return [
        _system("P 21 21 21", [41.0, 52.0, 63.0, 90, 90, 90], 900, 3, 0, device),
        _system("C 1 2 1", [70.0, 30.0, 25.0, 90, 103.5, 90], 700, 2, 1, device),
        _system("P 43 21 2", [50.0, 50.0, 90.0, 90, 90, 90], 800, 3, 2, device),
    ]


def _reference(sysd, r, spacing):
    """fast_translation_function on the system's own grid at that spacing.

    The inputs stay on the system's device, which need not be the default device
    the function computes on.
    """
    obs = SimpleNamespace(coeff=sysd["coeff"])
    cand = SimpleNamespace(G=sysd["G"][r], h_R=sysd["hR"].to(torch.float32))
    ref, _ = fast_translation_function(
        obs, cand, sysd["cell"], grid_spacing_A=spacing, n_peaks=1
    )
    return ref


@pytest.mark.parametrize(
    "device", [pytest.param("cuda", marks=pytest.mark.cuda), "cpu"]
)
def test_own_grids_match_single_candidate_map(device):
    """Per-system grids: every (system, orientation) map is the single-candidate one."""
    systems = _systems(device)
    spacing = 2.0
    grids = [_grid_sizes(s["cell"], spacing) for s in systems]
    maps = batched_translation_maps(
        [s["G"] for s in systems],
        [s["hR"] for s in systems],
        [s["coeff"] for s in systems],
        grids,
    )
    for s, m, grid in zip(systems, maps, grids):
        assert tuple(m.shape) == (s["G"].shape[0], *grid)
        for r in range(s["G"].shape[0]):
            ref = _reference(s, r, spacing).to(m.device)
            err = float((m[r] - ref).abs().max() / ref.abs().max())
            assert err < _TOL, f"{err:.2e}"


@pytest.mark.cuda
def test_shared_grid_and_padding():
    """One grid for the batch; a system with fewer orientations is zero-padded."""
    systems = _systems("cuda")
    grid = shared_grid_shape([s["cell"] for s in systems], 2.0)
    args = (
        [s["G"] for s in systems],
        [s["hR"] for s in systems],
        [s["coeff"] for s in systems],
        grid,
    )
    fused = batched_translation_maps(*args, force_portable=False)
    port = batched_translation_maps(*args, force_portable=True)
    assert tuple(fused.shape) == (3, 3, *grid)
    assert torch.all(fused[1, 2] == 0)
    err = float((fused - port).abs().max() / port.abs().max())
    assert err < _TOL, f"{err:.2e}"


def test_rejects_inconsistent_shapes():
    s = _systems("cpu")[0]
    with pytest.raises(ValueError):
        batched_translation_maps([s["G"]], [s["hR"][:, :-1]], [s["coeff"]], (8, 8, 8))
    with pytest.raises(ValueError):
        batched_translation_maps(
            [s["G"]], [s["hR"]], [s["coeff"]], [(8, 8, 8), (8, 8, 8)]
        )
