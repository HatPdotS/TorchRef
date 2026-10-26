"""Batched splats: a stack of members must equal the members splatted one at a time.

The batch contract (:mod:`torchref.base.electron_density.batching`) adds a leading member
axis to the density map and shifts each atom's writes into its member's slab. Nothing
about the per-atom arithmetic changes, so the reference for a batched call is the same
kernel run once per member and stacked -- not an independent oracle. Accuracy of the
unbatched kernels themselves is what ``test_forward.py`` / ``test_gradients.py`` cover.

Kernels that accumulate in a fixed order must match bitwise: every CPU forward, and the
fused CPU backward, which sums each atom's gradient on its own. The portable backward
regroups atoms into different chunks when batched, and the GPU kernels accumulate with
atomics, so those are held to a rounding tolerance.
"""

from __future__ import annotations

import pytest
import torch

from torchref.base.electron_density.batching import (
    check_batch,
    member_index,
    normalize_atoms,
)
from torchref.base.electron_density.radius_policy import (
    per_atom_radius_aniso,
    per_atom_radius_iso,
)
from torchref.config import get_sigma_cutoff_ed

from . import helpers as H
from .conftest import DEVICE_DTYPE_KERNELS

pytestmark = pytest.mark.unit

#: Member atom counts, including an empty member, so the slab arithmetic is tested across
#: a gap and members of unequal size.
_MEMBER_SIZES = (9, 0, 6)


def _exact(device: torch.device, name: str, grad: bool) -> bool:
    """Whether batching leaves this kernel's summation order, and so its bits, unchanged."""
    if grad:
        return name == "cpu_sphere"
    return device.type == "cpu"


def _tol(dtype: torch.dtype) -> float:
    """Max-abs tolerance, relative to the map's peak, for atomic accumulation order."""
    return 1e-12 if dtype is torch.float64 else 2e-6


def _members(device, dtype):
    """Scenes sharing the first one's cell, one per entry of ``_MEMBER_SIZES``."""
    return [
        H.synthetic_scene(n_atoms=n, seed=10 + i).to(device, dtype)
        for i, n in enumerate(_MEMBER_SIZES)
    ]


def _radius(scene, third, aniso):
    n_sigma = get_sigma_cutoff_ed()
    if aniso:
        return per_atom_radius_aniso(scene.B, third, n_sigma=n_sigma)
    return per_atom_radius_iso(third, scene.B, n_sigma=n_sigma)


def _splat(fn, cell_scene, dims, members, leaves, aniso, batched):
    """Run ``fn`` per member (stacked) or once batched; ``leaves`` are per-member
    ``(xyz, occ, third)``."""
    if not batched:
        maps = []
        for sc, (xyz, occ, third) in zip(members, leaves):
            z = torch.zeros(dims, dtype=xyz.dtype, device=xyz.device)
            maps.append(fn(
                z, xyz, third, occ, sc.A, sc.B, cell_scene.inv_frac_matrix,
                cell_scene.frac_matrix, _radius(sc, third, aniso),
            ))
        return torch.stack(maps)
    cat = lambda ts: torch.cat(list(ts))  # noqa: E731
    xyz, occ, third = (cat(t[i] for t in leaves) for i in range(3))
    radius = cat(_radius(sc, t[2], aniso) for sc, t in zip(members, leaves))
    sizes = [len(sc.xyz) for sc in members]
    offsets = torch.tensor([0] + sizes).cumsum(0)
    z = torch.zeros((len(members),) + tuple(dims), dtype=xyz.dtype, device=xyz.device)
    return fn(
        z, xyz, third, occ, cat(sc.A for sc in members), cat(sc.B for sc in members),
        cell_scene.inv_frac_matrix, cell_scene.frac_matrix, radius,
        batch_offsets=offsets,
    )


def _assert_close(got, ref, device, dtype, name, what, grad=False):
    if _exact(device, name, grad):
        assert torch.equal(got, ref), f"{what}: max diff {(got - ref).abs().max():.3e}"
        return
    scale = ref.abs().max().clamp(min=1e-30)
    err = float((got - ref).abs().max() / scale)
    assert err < _tol(dtype), f"{what}: rel max diff {err:.3e}"


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,name", DEVICE_DTYPE_KERNELS)
def test_batched_forward_matches_per_member(device, dtype, name, kind):
    aniso = kind == "aniso"
    members = _members(device, dtype)
    dims = H._grid_dims(members[0])
    fn = H.splat_kernel(name, aniso)
    leaves = [sc.leaves(aniso=aniso, requires_grad=False) for sc in members]
    ref = _splat(fn, members[0], dims, members, leaves, aniso, batched=False)
    got = _splat(fn, members[0], dims, members, leaves, aniso, batched=True)
    assert got.shape == (len(members),) + tuple(dims)
    assert ref.abs().max() > 0
    _assert_close(got, ref, device, dtype, name, "density")


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,name", DEVICE_DTYPE_KERNELS)
def test_batched_gradients_match_per_member(device, dtype, name, kind):
    """Each member's gradient sees only its own slab of the upstream gradient."""
    aniso = kind == "aniso"
    members = _members(device, dtype)
    dims = H._grid_dims(members[0])
    fn = H.splat_kernel(name, aniso)
    g = torch.Generator().manual_seed(3)
    weight = torch.randn((len(members),) + tuple(dims), generator=g, dtype=torch.float64)
    weight = weight.to(device=device, dtype=dtype)

    grads = {}
    for batched in (False, True):
        leaves = [sc.leaves(aniso=aniso) for sc in members]
        out = _splat(fn, members[0], dims, members, leaves, aniso, batched=batched)
        (out * weight).sum().backward()
        grads[batched] = [t.grad if t.grad is not None else torch.zeros_like(t)
                          for lv in leaves for t in lv]
    for i, (got, ref) in enumerate(zip(grads[True], grads[False])):
        if ref.numel():
            _assert_close(got, ref, device, dtype, name, f"grad leaf {i}", grad=True)


@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_dense_input_equals_ragged(kind):
    """``(B, N, ...)`` with shared ``A``/``B`` flattens to the ragged form."""
    aniso = kind == "aniso"
    sc = H.synthetic_scene(n_atoms=7, seed=4)
    third = sc.u6 if aniso else sc.adp
    shift = torch.tensor([0.3, -0.2, 0.1], dtype=sc.xyz.dtype)
    xyz = torch.stack([sc.xyz, sc.xyz + shift])
    flat = normalize_atoms(
        xyz, torch.stack([third, third]), torch.stack([sc.occ, sc.occ]), sc.A, sc.B
    )
    assert flat[0].shape == (14, 3) and flat[3].shape == (14, 5)
    assert flat[5].tolist() == [0, 7, 14]
    assert torch.equal(flat[0][7:], sc.xyz + shift)
    # Per-member A/B is accepted too, and gives the same flat tensors.
    per_member = normalize_atoms(
        xyz, torch.stack([third, third]), torch.stack([sc.occ, sc.occ]),
        torch.stack([sc.A, sc.A]), torch.stack([sc.B, sc.B]),
    )
    for a, b in zip(flat, per_member):
        assert torch.equal(a, b)


def test_unbatched_input_passes_through():
    sc = H.synthetic_scene(n_atoms=4, seed=1)
    out = normalize_atoms(sc.xyz, sc.adp, sc.occ, sc.A, sc.B)
    assert out[5] is None and out[0] is sc.xyz


def test_member_index():
    offsets = torch.tensor([0, 3, 3, 5])
    assert member_index(offsets, 5).tolist() == [0, 0, 0, 2, 2]


@pytest.mark.parametrize(
    "shape,offsets,match",
    [
        ((2, 4, 4, 4), None, "needs batch_offsets"),
        ((2, 4, 4, 4), [0, 1, 2, 3], "describe 3 members"),
        ((4, 4, 4), [0, 1, 3], "describe 2 members"),
        ((3, 4, 4, 4), [0, 2, 1, 3], "non-decreasing"),
        ((2, 4, 4, 4), [0, 1, 2], "from 0 to the atom count 3"),
        ((4, 4), None, "must be"),
    ],
)
def test_check_batch_refuses_inconsistent_calls(shape, offsets, match):
    off = None if offsets is None else torch.tensor(offsets)
    with pytest.raises(ValueError, match=match):
        check_batch(torch.zeros(shape), 3, off)


def test_dense_offsets_are_implied():
    sc = H.synthetic_scene(n_atoms=2, seed=1)
    with pytest.raises(ValueError, match="dense"):
        normalize_atoms(
            sc.xyz[None], sc.adp[None], sc.occ[None], sc.A, sc.B,
            batch_offsets=torch.tensor([0, 2]),
        )


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("name", ["portable", "cpu_sphere"])
def test_batched_second_order_matches_per_member(name, kind):
    """Double backward re-derives through the portable splat with the same batch."""
    aniso = kind == "aniso"
    device, dtype = torch.device("cpu"), torch.float64
    members = _members(device, dtype)
    dims = H._grid_dims(members[0])
    fn = H.splat_kernel(name, aniso)
    g = torch.Generator().manual_seed(5)
    weight = torch.randn((len(members),) + tuple(dims), generator=g, dtype=dtype)
    probes = [torch.randn(sc.xyz.shape, generator=g, dtype=dtype) for sc in members]

    hvps = {}
    for batched in (False, True):
        leaves = [sc.leaves(aniso=aniso) for sc in members]
        xyz = [lv[0] for lv in leaves]
        out = _splat(fn, members[0], dims, members, leaves, aniso, batched=batched)
        grads = torch.autograd.grad(
            ((out * weight) ** 2).sum(), xyz, create_graph=True, allow_unused=True
        )
        dot = sum((gr * p).sum() for gr, p in zip(grads, probes) if gr is not None)
        hvps[batched] = torch.autograd.grad(dot, xyz, allow_unused=True)
    for got, ref, sc in zip(hvps[True], hvps[False], members):
        if sc.n_atoms:
            _assert_close(got, ref, device, dtype, "portable", "hvp", grad=True)


def test_metal_refuses_a_batch_past_32_bit_indices():
    """The shaders index in ``int``, so a batch of 2**31 voxels must raise, not wrap."""
    from torchref.base.electron_density.kernels.mps.variable_radius import _member

    huge = torch.empty((2, 1024, 1024, 1024), device="meta")
    with pytest.raises(ValueError, match="32 bits"):
        _member(huge, torch.empty((0, 3), device="meta"), torch.tensor([0, 0, 0]))
