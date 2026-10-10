"""Non-bonded scoring of image pairs: pair positions, the eager kernel, and Triton.

A pair ``(i, j, symop, offset)`` is scored at the distance from atom ``i`` to the
``(symop, offset)`` image of atom ``j``, whatever the operation. In P1 every crystal
contact is a pure lattice translation under symop 0, so 5BOV (P 1) checks that the
offset is applied when no pair has another operation, and 1DAW (C 1 2 1) checks a list
mixing operations. Distances are compared with gemmi's own orthogonalization. The
CUDA-only Triton comparison skips without a CUDA device.
"""

import math

import gemmi
import numpy as np
import pytest
import torch

from torchref.base.coordinates import is_symmetry_image
from torchref.base.targets._dispatch import use_triton
from torchref.base.targets.nonbonded import (
    _nonbonded_heavy_math_eager,
    nonbonded_heavy_math,
    nonbonded_pair_positions,
)
from torchref.config import get_float_dtype
from torchref.symmetry import SpaceGroup
from torchref.symmetry.cell import Cell
from torchref.topology import nonbonded as nb

pytestmark = pytest.mark.unit

CUTOFF = 6.0
# The production NonBondedTarget defaults: sigma 0.3 Å, r_exp 4, no buffer.
_SIGMA, _R_EXP = 0.3, 4.0
_C_REP = 1.0 / (_R_EXP * _SIGMA**_R_EXP)


def _image_pairs(path):
    """Image pairs of one deposited model from builder steps 1-4, with radius sums."""
    st = gemmi.read_structure(str(path))
    atoms = [a for ch in st[0] for r in ch for a in r]
    # On CPU whatever the configured device: the distances are compared with gemmi's
    # on the host, and the Triton test moves this host copy to CUDA itself.
    cpu = torch.device("cpu")
    xyz = torch.tensor(
        [[a.pos.x, a.pos.y, a.pos.z] for a in atoms], dtype=get_float_dtype()
    )
    c = st.cell
    cell = Cell([c.a, c.b, c.c, c.alpha, c.beta, c.gamma], device=cpu)
    sg = SpaceGroup(st.spacegroup_hm, device=cpu)
    ops, offsets = nb.prefilter_symop_offsets(cell, sg, xyz, CUTOFF)
    identity = (~is_symmetry_image(ops, offsets)).nonzero()[0].item()
    grid_dims = torch.clamp(
        (torch.stack([cell.a, cell.b, cell.c]) / CUTOFF).long(), min=1
    )
    _, atom_idx, combo_idx, cart = nb.assign_to_grid(
        xyz, cell, sg, ops, offsets, grid_dims
    )
    i, j, combo = nb.find_pairs_kdtree(cart, atom_idx, combo_idx, CUTOFF, identity)
    image = combo != identity
    i, j, combo = i[image], j[image], combo[image]
    radii = torch.as_tensor(nb.vdw_radii_for_elements([a.element.name for a in atoms]))
    return {
        "st": st,
        "xyz": xyz,
        "indices": torch.stack([i, j], dim=1),
        "symop_indices": ops[combo],
        "cell_offsets": offsets[combo],
        "min_distances": (radii[i] + radii[j]).to(get_float_dtype()),
        "tables": (
            sg.matrices,
            sg.translations,
            cell.fractional_matrix,
            cell.inv_fractional_matrix,
        ),
    }


def _gemmi_distances(pairs):
    """Image distance of every pair through gemmi's own operations and cell, float64.

    TorchRef's :class:`SpaceGroup` takes its operations in gemmi's order, so a stored
    ``symop`` indexes gemmi's list directly.
    """
    st = pairs["st"]
    atoms = [a for ch in st[0] for r in ch for a in r]
    ops = list(st.find_spacegroup().operations())
    out = []
    for (i, j), s, n in zip(
        pairs["indices"].tolist(),
        pairs["symop_indices"].tolist(),
        pairs["cell_offsets"].tolist(),
    ):
        f = st.cell.fractionalize(atoms[j].pos)
        x, y, z = ops[s].apply_to_xyz([f.x, f.y, f.z])
        image = gemmi.Fractional(x + n[0], y + n[1], z + n[2])
        out.append(atoms[i].pos.dist(st.cell.orthogonalize(image)))
    return np.array(out)


@pytest.fixture(scope="module", params=["5BOV.pdb", "1DAW.pdb"])
def pairs(request, pdb_dir):
    out = _image_pairs(pdb_dir / request.param)
    out["name"] = request.param
    return out


def _args(pairs):
    return (
        pairs["xyz"],
        pairs["indices"],
        pairs["symop_indices"],
        pairs["cell_offsets"],
        *pairs["tables"],
    )


def test_image_pairs_are_placed_at_gemmi_distances(pairs):
    pos1, pos2 = nonbonded_pair_positions(*_args(pairs))
    got = (pos2 - pos1).norm(dim=1).double().numpy()
    want = _gemmi_distances(pairs)
    assert np.abs(got - want).max() < 1e-3
    assert want.max() < CUTOFF
    if pairs["name"] == "5BOV.pdb":
        assert bool((pairs["symop_indices"] == 0).all()), "P1: lattice images only"


def test_loss_scores_every_image_pair(pairs):
    xyz = pairs["xyz"]
    one = torch.ones((), dtype=xyz.dtype)
    loss = nonbonded_heavy_math(
        xyz,
        pairs["indices"],
        pairs["min_distances"],
        pairs["symop_indices"],
        pairs["cell_offsets"],
        *pairs["tables"],
        _C_REP * one,
        _R_EXP * one,
        0.0,
        _SIGMA * one,
    )
    # The kernel's sqrt epsilon matters here: 1DAW has a water on the 2-fold axis
    # whose image under that axis is itself, at distance 0.
    distance = np.sqrt(_gemmi_distances(pairs) ** 2 + 1e-8)
    overlap = np.clip(pairs["min_distances"].double().numpy() - distance, 0.0, None)
    assert np.count_nonzero(overlap) > 0, "the fixture must contain clashing mates"
    n = len(overlap)
    want = _C_REP * (overlap**_R_EXP).sum() + n * (
        math.log(_SIGMA) + 0.5 * math.log(2.0 * math.pi)
    )
    assert float(loss) == pytest.approx(want, rel=1e-5, abs=1e-2)


@pytest.mark.cuda
@pytest.mark.parametrize("pdb", ["5BOV.pdb", "3E98.pdb"])
def test_triton_matches_eager_on_image_pairs(pdb, pdb_dir):
    """The Triton kernel images every pair as the eager path does, offsets included.

    5BOV (P 1) holds lattice images only, all under symop 0; 3E98 (P 1 21 1) mixes
    them with screw-axis images. Neither has an atom on a special position, whose
    self-image at distance ~0 would make its gradient direction float32 noise.
    """
    from tests.helpers.grad_asserts import assert_grads_agree

    host = _image_pairs(pdb_dir / pdb)
    dev = torch.device("cuda")
    # float32 throughout, whatever the configured dtype: that is the Triton contract.
    xyz = host["xyz"].to(dev, torch.float32)
    min_distances = host["min_distances"].to(dev, torch.float32)
    tables = [t.to(dev, torch.float32) for t in host["tables"]]
    indices = host["indices"].to(dev)
    symop_indices = host["symop_indices"].to(dev)
    cell_offsets = host["cell_offsets"].to(dev)
    one = torch.ones((), device=dev, dtype=torch.float32)
    scalars = (_C_REP * one, _R_EXP * one, 0.0, _SIGMA * one)
    assert use_triton(xyz), "the Triton arm would compare eager against eager"

    def run(fn, offsets):
        x = xyz.clone().requires_grad_(True)
        loss = fn(x, indices, min_distances, symop_indices, offsets, *tables, *scalars)
        (grad,) = torch.autograd.grad(loss, x)
        return loss.detach(), grad

    loss_t, grad_t = run(nonbonded_heavy_math, cell_offsets)
    loss_e, grad_e = run(_nonbonded_heavy_math_eager, cell_offsets)
    loss_unshifted, _ = run(_nonbonded_heavy_math_eager, torch.zeros_like(cell_offsets))

    # Non-vacuity: dropping the offsets must move the loss far past the tolerance.
    assert abs(float(loss_unshifted - loss_e)) > 1.0
    torch.testing.assert_close(loss_t, loss_e, rtol=1e-5, atol=1e-2)
    assert_grads_agree([grad_t], [grad_e], min_cos=0.9999, ratio_tol=1e-3, ctx="vdw ")


@pytest.mark.cuda
def test_triton_weighs_each_pair_as_eager_does(pdb_dir):
    """Per-pair weights scale a pair's whole NLL, constant included, and its gradient,
    in the Triton kernel as in the eager one. The weights alternate 1/2 and 1, the two
    the pair lists carry, so a kernel that dropped or misplaced them would differ."""
    from tests.helpers.grad_asserts import assert_grads_agree

    host = _image_pairs(pdb_dir / "3E98.pdb")
    dev = torch.device("cuda")
    xyz = host["xyz"].to(dev, torch.float32)
    min_distances = host["min_distances"].to(dev, torch.float32)
    tables = [t.to(dev, torch.float32) for t in host["tables"]]
    pair_args = (
        host["indices"].to(dev),
        min_distances,
        host["symop_indices"].to(dev),
        host["cell_offsets"].to(dev),
        *tables,
    )
    one = torch.ones((), device=dev, dtype=torch.float32)
    scalars = (_C_REP * one, _R_EXP * one, 0.0, _SIGMA * one)
    alternate = torch.arange(len(min_distances), device=dev) % 2 == 0
    weights = torch.where(alternate, 0.5, 1.0).to(torch.float32)
    assert use_triton(xyz), "the Triton arm would compare eager against eager"

    def run(fn, w):
        x = xyz.clone().requires_grad_(True)
        loss = fn(x, *pair_args, *scalars, w)
        (grad,) = torch.autograd.grad(loss, x)
        return loss.detach(), grad

    loss_t, grad_t = run(nonbonded_heavy_math, weights)
    loss_e, grad_e = run(_nonbonded_heavy_math_eager, weights)
    loss_unweighted, _ = run(_nonbonded_heavy_math_eager, None)

    assert abs(float(loss_unweighted - loss_e)) > 1.0
    torch.testing.assert_close(loss_t, loss_e, rtol=1e-5, atol=1e-2)
    assert_grads_agree([grad_t], [grad_e], min_cos=0.9999, ratio_tol=1e-3, ctx="vdw ")
