"""Batched splats: a stack of members must equal the members splatted one at a time.

The batch contract (:mod:`torchref.base.electron_density.batching`) adds a leading member
axis to the density map and shifts each atom's writes into its member's slab. Nothing
about the per-atom arithmetic changes, so the reference for a batched call is the same
kernel run once per member and stacked -- not an independent oracle. Accuracy of the
unbatched kernels themselves is what ``test_forward.py`` / ``test_gradients.py`` cover.

The fused CPU kernel accumulates in a fixed order, so it must match bitwise, forward and
backward. The portable splat regroups atoms into different ``scatter_add`` chunks when
batched, which changes its summation order on some hosts, and the GPU kernels accumulate
with atomics, so those are held to a rounding tolerance.
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
    return name == "cpu_sphere"


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
@pytest.mark.parametrize("device,dtype,name", DEVICE_DTYPE_KERNELS)
def test_batched_second_order_matches_per_member(device, dtype, name, kind):
    """A batched HVP along every leaf equals the members' HVPs, stacked."""
    aniso = kind == "aniso"
    members = _members(device, dtype)
    dims = H._grid_dims(members[0])
    fn = H.splat_kernel(name, aniso)
    g = torch.Generator().manual_seed(5)
    rand = lambda shape: torch.randn(shape, generator=g).to(device, dtype)  # noqa: E731
    weight = rand((len(members),) + tuple(dims))
    probes = [rand(t.shape) for sc in members for t in sc.leaves(aniso=aniso)]

    hvps = {}
    for batched in (False, True):
        leaves = [sc.leaves(aniso=aniso) for sc in members]
        flat = [t for lv in leaves for t in lv]
        out = _splat(fn, members[0], dims, members, leaves, aniso, batched=batched)
        grads = torch.autograd.grad(
            ((out * weight) ** 2).sum(), flat, create_graph=True, allow_unused=True
        )
        dot = sum((gr * p).sum() for gr, p in zip(grads, probes) if gr is not None)
        hvps[batched] = torch.autograd.grad(dot, flat, allow_unused=True)
    for got, ref in zip(hvps[True], hvps[False]):
        if ref is not None and ref.numel():
            # cpu_sphere takes its second derivative through the portable splat, so it is
            # not bitwise here.
            _assert_close(got, ref, device, dtype, "portable", "hvp", grad=True)


def test_metal_refuses_a_batch_past_32_bit_indices():
    """The shaders index in ``int``, so a batch of 2**31 voxels must raise, not wrap."""
    from torchref.base.electron_density.kernels.mps.variable_radius import _member

    huge = torch.empty((2, 1024, 1024, 1024), device="meta")
    with pytest.raises(ValueError, match="32 bits"):
        _member(huge, torch.empty((0, 3), device="meta"), torch.tensor([0, 0, 0]))


# ---------------------------------------------------------------------------
# Base functions above the kernels
# ---------------------------------------------------------------------------
def _build_kw(scene, dims):
    return dict(
        grid_shape=dims, device=scene.xyz.device, inv_frac_matrix=scene.inv_frac_matrix,
        frac_matrix=scene.frac_matrix, dtype=scene.xyz.dtype,
    )


def test_build_electron_density_batched_matches_loop():
    """Dense isotropic atoms beside ragged anisotropic ones, one member empty of the
    latter."""
    from torchref.base.electron_density.main import build_electron_density

    a, b = (H.synthetic_scene(n_atoms=6, seed=s) for s in (20, 21))
    c = H.synthetic_scene(n_atoms=4, seed=22)
    dims = H._grid_dims(a)
    kw = _build_kw(a, dims)

    got = build_electron_density(
        xyz_iso=torch.stack([a.xyz, b.xyz]), adp_iso=torch.stack([a.adp, b.adp]),
        occ_iso=torch.stack([a.occ, b.occ]), A_iso=a.A, B_iso=torch.stack([a.B, b.B]),
        xyz_aniso=c.xyz, u_aniso=c.u6, occ_aniso=c.occ, A_aniso=c.A, B_aniso=c.B,
        batch_offsets_aniso=torch.tensor([0, 4, 4]), **kw,
    )
    ref0 = build_electron_density(
        xyz_iso=a.xyz, adp_iso=a.adp, occ_iso=a.occ, A_iso=a.A, B_iso=a.B,
        xyz_aniso=c.xyz, u_aniso=c.u6, occ_aniso=c.occ, A_aniso=c.A, B_aniso=c.B, **kw,
    )
    # Member 1 shares member 0's A (passed once), but has its own B.
    ref1 = build_electron_density(
        xyz_iso=b.xyz, adp_iso=b.adp, occ_iso=b.occ, A_iso=a.A, B_iso=b.B, **kw,
    )
    assert got.shape == (2,) + tuple(dims)
    assert torch.equal(got, torch.stack([ref0, ref1]))


def test_build_electron_density_refuses_a_half_batched_call():
    from torchref.base.electron_density.main import build_electron_density

    a = H.synthetic_scene(n_atoms=3, seed=1)
    kw = _build_kw(a, H._grid_dims(a))
    with pytest.raises(ValueError, match="one atom set is batched"):
        build_electron_density(
            xyz_iso=a.xyz[None], adp_iso=a.adp[None], occ_iso=a.occ[None],
            A_iso=a.A, B_iso=a.B,
            xyz_aniso=a.xyz, u_aniso=a.u6, occ_aniso=a.occ, A_aniso=a.A, B_aniso=a.B,
            **kw,
        )
    with pytest.raises(ValueError, match="different batch sizes"):
        build_electron_density(
            xyz_iso=a.xyz[None], adp_iso=a.adp[None], occ_iso=a.occ[None],
            A_iso=a.A, B_iso=a.B,
            xyz_aniso=a.xyz, u_aniso=a.u6, occ_aniso=a.occ, A_aniso=a.A, B_aniso=a.B,
            batch_offsets_aniso=torch.tensor([0, 1, 3]), **kw,
        )


@pytest.mark.parametrize("shape", [(12, 16, 20), (11, 15, 21)], ids=["direct", "interp"])
def test_symmetrize_map_batched_matches_loop(shape):
    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup("P 21 21 21", dtype=torch.float64, device=torch.device("cpu"))
    assert sg.can_index_directly(shape) == (shape == (12, 16, 20))
    g = torch.Generator().manual_seed(0)
    maps = torch.rand((3,) + shape, generator=g, dtype=torch.float64)
    got = sg.symmetrize_map(maps)
    ref = torch.stack([sg.symmetrize_map(m) for m in maps])
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-14)
    mates = sg.expand_map_to_P1(maps)
    assert mates.shape == (sg.n_ops, 3) + shape


def test_reciprocal_extractor_batched_matches_loop():
    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup("P 21 21 21", dtype=torch.float64, device=torch.device("cpu"))
    shape = (12, 16, 20)
    hkl = torch.tensor([[1, 2, 3], [0, 0, 2], [3, 1, 0], [-2, 4, 5]])
    ex = sg.reciprocal_extractor(hkl, shape)
    g = torch.Generator().manual_seed(1)
    grids = torch.randn((3,) + shape, generator=g, dtype=torch.complex128)
    grids.requires_grad_(True)
    got = ex.extract_from_grid(grids)
    ref = torch.stack([ex.extract_from_grid(m) for m in grids])
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-14)
    (gb,) = torch.autograd.grad(got.abs().sum(), grids)
    (gl,) = torch.autograd.grad(ref.abs().sum(), grids)
    torch.testing.assert_close(gb, gl, rtol=0, atol=1e-14)


def _absorptive(scene, n):
    """A small imaginary (f'') term on the scene's first ``n`` atoms, isotropic only."""
    z = scene.xyz.new_zeros
    return (scene.xyz[:n], scene.adp[:n], scene.occ[:n], 0.1 * scene.A[:n],
            scene.B[:n], z(0, 3), z(0, 6), z(0), z(0, 5), z(0, 5))


@pytest.mark.parametrize("late", [True, False], ids=["late_symmetry", "early_symmetry"])
@pytest.mark.parametrize("with_imaginary", [False, True], ids=["real", "imaginary"])
def test_sf_fft_batched_matches_loop(late, with_imaginary):
    """Dense isotropic atoms, ragged anisotropic ones and (optionally) a ragged
    imaginary part, through both symmetry routes."""
    a, b = (H.synthetic_scene(n_atoms=5, seed=s, max_refl=60) for s in (30, 31))
    c = H.synthetic_scene(n_atoms=3, seed=32)
    sf = H.sf_fft_for(a, spacegroup="P 21 21 21")
    sf.use_late_symmetry = late
    sf.ensure_grid()
    assert sf.late_symmetry_compatible

    def one(iso, aniso, imag):
        return sf.compute_structure_factors(
            a.hkl, *iso, *aniso, imaginary=imag if with_imaginary else None
        )[0]

    refs = [
        one((s.xyz, s.adp, s.occ, s.A, s.B), aniso, _absorptive(s, k))
        for s, aniso, k in (
            (a, (c.xyz, c.u6, c.occ, c.A, c.B), 2),
            (b, (None,) * 5, 3),
        )
    ]
    ab = lambda f: torch.stack([f(a), f(b)])  # noqa: E731
    imag = None
    imag_offsets = None
    if with_imaginary:
        ia, ib = _absorptive(a, 2), _absorptive(b, 3)
        imag = tuple(torch.cat([x, y]) for x, y in zip(ia, ib))
        imag_offsets = (torch.tensor([0, 2, 5]), torch.tensor([0, 0, 0]))
    got, density = sf.compute_structure_factors(
        a.hkl,
        ab(lambda s: s.xyz), ab(lambda s: s.adp), ab(lambda s: s.occ),
        ab(lambda s: s.A), ab(lambda s: s.B),
        c.xyz, c.u6, c.occ, c.A, c.B,
        imaginary=imag,
        batch_offsets_aniso=torch.tensor([0, 3, 3]),
        imaginary_batch_offsets=imag_offsets,
    )
    assert got.shape == (2, a.n_refl)
    assert density.shape[0] == 2
    torch.testing.assert_close(got, torch.stack(refs), rtol=1e-12, atol=1e-10)


def _sf_fft_pair():
    """Two 5-atom members sharing a P212121 cell, and an SfFFT on their grid."""
    a, b = (H.synthetic_scene(n_atoms=5, seed=s, max_refl=60) for s in (30, 31))
    sf = H.sf_fft_for(a, spacegroup="P 21 21 21")
    sf.ensure_grid()
    return a, b, sf


def _iso(*scenes, dense):
    """The scenes' isotropic atoms, stacked (dense) or concatenated (ragged)."""
    join = torch.stack if dense else torch.cat
    return [join([getattr(s, k) for s in scenes]) for k in ("xyz", "adp", "occ", "A", "B")]


@pytest.mark.parametrize("ragged", [False, True], ids=["dense", "ragged"])
@pytest.mark.parametrize("late", [True, False], ids=["late_symmetry", "early_symmetry"])
def test_sf_fft_empty_batch(late, ragged):
    """A batch of no members gives no structure factors, not an FFT error."""
    a, _, sf = _sf_fft_pair()
    sf.use_late_symmetry = late
    atoms = [t[:0] if ragged else t[None][:0] for t in _iso(a, dense=False)]
    F, density = sf.compute_structure_factors(
        a.hkl, *atoms, batch_offsets_iso=torch.tensor([0]) if ragged else None
    )
    assert F.shape == (0, a.n_refl)
    assert density.shape == (0, *sf.grid_shape)


def test_sf_fft_empty_unbatched_imaginary_spans_the_batch():
    a, b, sf = _sf_fft_pair()
    real, _ = sf.compute_structure_factors(a.hkl, *_iso(a, b, dense=True))
    got, _ = sf.compute_structure_factors(
        a.hkl, *_iso(a, b, dense=True), imaginary=_absorptive(a, 0)
    )
    assert got.shape == (2, a.n_refl)
    torch.testing.assert_close(got, real, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize(
    "case,match",
    [
        ("unbatched_imaginary", "batched and the other is not"),
        ("unbatched_real", "batched and the other is not"),
        ("one_member_imaginary", r"different batch sizes \[1, 2\]"),
    ],
)
def test_sf_fft_refuses_mismatched_imaginary_batch(case, match):
    """The imaginary atoms must describe the real atoms' batch; ``torch.complex``
    would otherwise broadcast one imaginary map over every member."""
    a, b, sf = _sf_fft_pair()
    ia, ib = _absorptive(a, 2), _absorptive(b, 3)
    pair = torch.tensor([0, 5, 10])
    kw = {}
    if case == "unbatched_imaginary":
        real, kw["batch_offsets_iso"] = _iso(a, b, dense=False), pair
        imag = tuple(torch.cat([x, y]) for x, y in zip(ia, ib))
    elif case == "unbatched_real":
        real = _iso(a, dense=False)
        imag = tuple(torch.cat([x, y]) for x, y in zip(ia, ib))
        kw["imaginary_batch_offsets"] = (torch.tensor([0, 2, 5]), None)
    else:
        real = _iso(a, b, dense=True)
        imag = tuple(t[None] if i < 5 else t for i, t in enumerate(ia))
    with pytest.raises(ValueError, match=match):
        sf.compute_structure_factors(a.hkl, *real, imaginary=imag, **kw)
