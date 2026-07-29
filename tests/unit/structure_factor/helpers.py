"""Shared scene construction, oracle wrappers and comparison metrics.

Two scene sources:

* :func:`synthetic_scene` -- a small random P1 scene, for the ``gradcheck`` /
  ``gradgradcheck`` links where the cost is O(n_params) forward evaluations and the
  scene size is what makes the difference between milliseconds and minutes.
* :func:`gemmi_scene` -- a real deposited structure, with the torchref scene built
  **from** the gemmi structure so both sides are driven by the same atoms.

That second point is load-bearing. If the two sides are populated in parallel from the
same file by two different readers, the test silently measures reader agreement instead
of structure-factor agreement, and a single differing atom shows up as a tolerance
problem rather than as the bug it is. So there is exactly one traversal of the gemmi
model, and torchref's tensors are filled from it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields
from typing import Optional, Sequence

import torch

from torchref.base.direct_summation.dispatch import _eager_aniso, _eager_iso
from torchref.base.reciprocal import get_scattering_vectors, reciprocal_basis_matrix
from torchref.base.scattering.scattering_table import get_scattering_params_by_z
from torchref.symmetry.cell import Cell

__all__ = [
    "Scene",
    "GRID_FINENESS",
    "synthetic_scene",
    "gemmi_scene",
    "gemmi_sf",
    "ds_iso_oracle",
    "ds_aniso_oracle",
    "sf_fft_for",
    "fft_sf",
    "splat_kernel",
    "device_supports_dtype",
    "kernels_for",
    "splat_direct",
    "ds_kernel",
    "ds_kernels_for",
    "ds_direct",
    "density_to_F",
    "synthetic_obs",
    "ls_target",
    "best_fit_scale",
    "rel_l2",
    "max_rel",
]

# Grids are built at ``max_res = d_min / GRID_FINENESS``, i.e. a voxel spacing of
# ``d_min / (3 * GRID_FINENESS)`` given ``NYQUIST_OVERSAMPLING = 3.0``.
#
# ``GRID_FINENESS = 1.0`` is the **production** configuration: ``max_res = d_min``, voxel
# spacing ``d_min / 3``. These tests default to it, because a gate on a finer grid than
# production ships would not bound anything that matters.
#
# Measured on ``scene_fine`` (60 atoms, a = 24 A, d_min 1.6), float64, against the DS
# oracle. Amplitudes are rel L2 on complex F; derivatives are of ``ls_target``:
#
#   fineness  spacing    gridsize        amplitude   g_xyz      g_xyz cos   HVP        HVP cos
#   0.667     d_min/2    (30, 36, 27)    1.04e-01    1.25e+00    0.4207     2.77e-01   0.9622
#   1.000     d_min/3    (45, 50, 40)    4.11e-03    4.30e-02    0.999077   2.06e-02   0.999814
#   1.300     d_min/3.9  (60, 64, 54)    8.01e-04    5.86e-03    0.999983   1.16e-03   0.999999
#   1.600     d_min/4.8  (72, 80, 64)    8.03e-04    5.87e-03    0.999983   1.16e-03   0.999999
#   2.200     d_min/6.6  (100,108, 90)   7.98e-04    6.00e-03    0.999982   1.16e-03   0.999999
#
# Three things follow.
#
# 1. **Bare Nyquist is unusable**, which is why ``NYQUIST_OVERSAMPLING`` is 3 and not 2. At
#    oversampling 2 the xyz gradient cosine against the analytic answer collapses to 0.42
#    and amplitudes are 10% out. The factor of 3 is buying a great deal.
# 2. **Production sits one step before convergence.** Everything is converged from fineness
#    1.3. At production the residuals are ~5x larger in amplitude and ~7x in the xyz
#    gradient, but direction stays excellent (cos 0.999) so the residual is predominantly
#    magnitude. On a *real* structure the production numbers are better still -- 7L84 gives
#    amplitude 2.28e-03 and xyz gradient 1.04e-02 -- because derivative aliasing cancels
#    across atoms as ~1/sqrt(N). See the gate constants in ``__init__.py``; absolute
#    accuracy gates are calibrated there rather than here.
# 3. **The sigma cutoff is not the binding constraint at production.** Sweeping n_sigma at
#    fineness 1.0 moves the amplitude residual 5.43e-3 -> 4.11e-3 -> 4.05e-3 and then
#    flatlines; grid sampling dominates. Tests that mean to exercise the cutoff therefore
#    pass an explicit finer ``fineness`` -- see
#    ``test_forward.py::test_nsigma_reduces_truncation_error``.
GRID_FINENESS = 1.0


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scene:
    """Everything both routes need, in one immutable bundle.

    ``xyz`` is Cartesian (what ``SfFFT`` / ``SfDS`` take) and ``xyz_frac`` is
    fractional (what the ``direct_summation`` functions take). Both are carried rather
    than converted at the call site, because getting that conversion wrong in one place
    produces a plausible-looking wrong answer instead of an error.
    """

    cell: Cell
    spacegroup: str
    d_min: float
    hkl: torch.Tensor  # (R, 3) float
    hkl_list: tuple  # (R,) tuples of int, for gemmi
    s: torch.Tensor  # (R,)  |s| = 1/d
    s_vec: torch.Tensor  # (R, 3)
    xyz: torch.Tensor  # (N, 3) Cartesian
    xyz_frac: torch.Tensor  # (N, 3) fractional
    occ: torch.Tensor  # (N,)
    adp: torch.Tensor  # (N,)   B convention, A^2
    u6: torch.Tensor  # (N, 6) [U11,U22,U33,U12,U13,U23], A^2
    A: torch.Tensor  # (N, 5) ITC92
    B: torch.Tensor  # (N, 5) ITC92
    frac_matrix: torch.Tensor  # (3, 3) fractional -> Cartesian
    inv_frac_matrix: torch.Tensor  # (3, 3) Cartesian -> fractional

    @property
    def n_atoms(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def n_refl(self) -> int:
        return int(self.hkl.shape[0])

    def leaves(self, *, aniso: bool = False, requires_grad: bool = True):
        """Fresh differentiable copies of ``(xyz, occ, adp_or_u6)``."""
        third = self.u6 if aniso else self.adp
        return tuple(
            t.clone().requires_grad_(requires_grad) for t in (self.xyz, self.occ, third)
        )

    def to(self, device=None, dtype=None) -> "Scene":
        """A new :class:`Scene` with every tensor moved and cast.

        Scenes are built CPU/float64 and stay that way, because the oracle is computed
        from them and a reference must not inherit the precision of the thing it judges.
        Only the *candidate* side is moved, so an MPS float32 kernel and a CPU float64
        oracle describe the same physical structure.

        ``hkl_list`` is untouched -- it is Python ints for gemmi, not a tensor.
        """
        if device is None and dtype is None:
            return self
        real = {"hkl", "s", "s_vec", "xyz", "xyz_frac", "occ", "adp", "u6", "A", "B",
                "frac_matrix", "inv_frac_matrix"}
        moved = {}
        for f in fields(self):
            v = getattr(self, f.name)
            moved[f.name] = (
                v.to(device=device, dtype=dtype) if f.name in real else v
            )
        return Scene(**moved)


def _hkl_within(cell: Cell, d_min: float, dtype: torch.dtype, cap: Optional[int]):
    """Every integer hkl inside the ``d_min`` shell, ordered, ``F(000)`` excluded.

    ``F(000)`` is dropped because the FFT route cannot reproduce it: the grid carries a
    truncated density, so the zeroth term is the truncated electron count rather than
    the true one.
    """
    a, b, c = (float(cell.data[i]) for i in range(3))
    lim = [int(v / d_min) + 1 for v in (a, b, c)]
    recB = reciprocal_basis_matrix(cell.data)
    cand = [
        h
        for h in itertools.product(
            range(-lim[0], lim[0] + 1),
            range(-lim[1], lim[1] + 1),
            range(-lim[2], lim[2] + 1),
        )
        if any(h)
    ]
    hkl = torch.tensor(cand, dtype=dtype)
    s = get_scattering_vectors(hkl, cell.data, recB).norm(dim=1)
    keep = (s > 0) & (s <= 1.0 / d_min)
    hkl = hkl[keep]
    if cap is not None and hkl.shape[0] > cap:
        # Even stride, so the kept set still spans the full resolution range rather
        # than clustering at low angle where agreement is easiest.
        step = hkl.shape[0] // cap + 1
        hkl = hkl[::step]
    return hkl


def _finish(
    cell: Cell,
    spacegroup: str,
    d_min: float,
    hkl: torch.Tensor,
    z: torch.Tensor,
    xyz_frac: torch.Tensor,
    occ: torch.Tensor,
    adp: torch.Tensor,
    u6: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Scene:
    recB = reciprocal_basis_matrix(cell.data)
    s_vec = get_scattering_vectors(hkl, cell.data, recB)
    frac_matrix = cell.fractional_matrix.to(dtype)
    inv_frac_matrix = cell.inv_fractional_matrix.to(dtype)
    A, B = get_scattering_params_by_z(z, dtype=dtype)
    to = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
    return Scene(
        cell=cell,
        spacegroup=spacegroup,
        d_min=d_min,
        hkl=to(hkl),
        hkl_list=tuple(tuple(int(v) for v in row) for row in hkl),
        s=to(s_vec.norm(dim=1)),
        s_vec=to(s_vec),
        xyz=to(xyz_frac @ frac_matrix.T),
        xyz_frac=to(xyz_frac),
        occ=to(occ),
        adp=to(adp),
        u6=to(u6),
        A=to(A),
        B=to(B),
        frac_matrix=to(frac_matrix),
        inv_frac_matrix=to(inv_frac_matrix),
    )


def synthetic_scene(
    *,
    n_atoms: int = 10,
    a: float = 24.0,
    beta: float = 90.0,
    d_min: float = 1.6,
    seed: int = 0,
    max_refl: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Scene:
    """A small random P1 scene.

    Two degeneracies are deliberately avoided, both of which once survived in two test
    files at the same time:

    * ``occ`` is never exactly 1.0. The kernels recover ``d/d_occ`` by dividing the
      accumulated gradient by ``occ``, and at ``occ == 1`` that division is a no-op that
      hides a wrong scaling.
    * the ADP off-diagonals are non-zero **and signed**. Zero off-diagonals mean every
      ellipsoid is axis-aligned, which leaves the cross-term arithmetic completely
      uncovered -- the ``p01``/``p02``/``p12`` entries of the inverted 3x3, and the
      backward's off-diagonal U gradients, which carry a ``4*pi^2`` factor where the
      diagonal ones carry ``2*pi^2``. Magnitudes stay well below the diagonal so every U
      remains comfortably positive-definite, since the Metal shader inverts ``M_g``
      analytically with no positive-definiteness guard.

    (Recorded here rather than in ``tests/helpers/kernel_cases.py``, which documented them
    for the two accelerator test files this package replaced and has been removed with
    them. ``test_dispatch.py::test_aniso_scene_exercises_off_diagonal_u`` asserts the
    second one rather than trusting this docstring.)
    """
    g = torch.Generator().manual_seed(seed)
    cell = Cell(
        torch.tensor([a, a * 1.1, a * 0.9, 90.0, beta, 90.0], dtype=torch.float64)
    )
    hkl = _hkl_within(cell, d_min, torch.float64, max_refl)

    z = torch.tensor(([6, 7, 8, 16, 6, 8, 7, 26] * ((n_atoms // 8) + 1))[:n_atoms])
    # Keep atoms off the cell walls so a wrapped-vs-unwrapped bug in the map route
    # cannot be confused with a truncation residual.
    xyz_frac = 0.15 + 0.7 * torch.rand(n_atoms, 3, generator=g, dtype=torch.float64)
    occ = 0.55 + 0.4 * torch.rand(n_atoms, generator=g, dtype=torch.float64)
    adp = 8.0 + 22.0 * torch.rand(n_atoms, generator=g, dtype=torch.float64)

    u6 = torch.zeros(n_atoms, 6, dtype=torch.float64)
    u6[:, :3] = (adp[:, None] / (8.0 * torch.pi**2)) * (
        0.75 + 0.5 * torch.rand(n_atoms, 3, generator=g, dtype=torch.float64)
    )
    off = 0.35 * u6[:, :3].min(dim=1).values
    u6[:, 3:] = off[:, None] * (
        2.0 * torch.rand(n_atoms, 3, generator=g, dtype=torch.float64) - 1.0
    )
    return _finish(
        cell, "P 1", d_min, hkl, z, xyz_frac, occ, adp, u6, dtype, device
    )


def gemmi_scene(
    pdb_path,
    *,
    p1: bool = True,
    d_min: float = 3.0,
    max_refl: Optional[int] = 200,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
):
    """Build a :class:`Scene` **from** a gemmi structure; return ``(scene, structure)``.

    Hydrogens are removed on the gemmi side before the traversal, so both routes see
    the identical atom list -- ``remove_hydrogens`` after extraction would leave
    torchref summing over atoms gemmi no longer has.

    ``p1=True`` rewrites the spacegroup to ``P 1`` and rebuilds ``cell.images``, which
    is the mechanism gemmi's calculator uses for symmetry: ``calculate_sf_from_model``
    sums each atom over ``cell.images``, so an empty image list is a true P1 sum.
    Verified: forcing P1 takes ``len(cell.images)`` from 11 to 0 on a P 65 2 2 entry.
    """
    import gemmi

    st = gemmi.read_structure(str(pdb_path))
    st.setup_entities()
    st.remove_hydrogens()
    if p1:
        st.spacegroup_hm = "P 1"
    st.setup_cell_images()

    zs, frac, occ, adp, u6 = [], [], [], [], []
    for chain in st[0]:
        for res in chain:
            for at in res:
                f = st.cell.fractionalize(at.pos)
                an = at.aniso
                zs.append(at.element.atomic_number)
                frac.append([f.x, f.y, f.z])
                occ.append(at.occ)
                adp.append(at.b_iso)
                u6.append([an.u11, an.u22, an.u33, an.u12, an.u13, an.u23])

    c = st.cell
    cell = Cell(
        torch.tensor([c.a, c.b, c.c, c.alpha, c.beta, c.gamma], dtype=torch.float64)
    )
    hkl = _hkl_within(cell, d_min, torch.float64, max_refl)
    scene = _finish(
        cell,
        st.spacegroup_hm,
        d_min,
        hkl,
        torch.tensor(zs),
        torch.tensor(frac, dtype=torch.float64),
        torch.tensor(occ, dtype=torch.float64),
        torch.tensor(adp, dtype=torch.float64),
        torch.tensor(u6, dtype=torch.float64),
        dtype,
        device,
    )
    return scene, st


def gemmi_sf(structure, hkl_list: Sequence, *, dtype=torch.complex128) -> torch.Tensor:
    """``F(hkl)`` from gemmi's X-ray structure-factor calculator.

    ``addends`` is left untouched (zero), so there is no anomalous term to account for
    on the torchref side.
    """
    import gemmi

    calc = gemmi.StructureFactorCalculatorX(structure.cell)
    return torch.tensor(
        [complex(calc.calculate_sf_from_model(structure[0], h)) for h in hkl_list],
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------
def ds_iso_oracle(scene: Scene, xyz=None, occ=None, adp=None) -> torch.Tensor:
    """Isotropic ``F(hkl)`` from the pure-torch eager path.

    ``_eager_iso`` and not ``ds_iso``: the public entry points wrap ``_CheckpointedSF``,
    whose backward omits ``create_graph=True``, so they raise on double backward. See
    the package docstring.

    ``xyz`` is **Cartesian** here, matching :attr:`Scene.xyz`; it is fractionalized
    inside so a differentiable leaf stays a leaf in Cartesian space, which is what the
    FFT route differentiates with respect to.
    """
    xyz = scene.xyz if xyz is None else xyz
    occ = scene.occ if occ is None else occ
    adp = scene.adp if adp is None else adp
    return _eager_iso(
        scene.hkl,
        scene.s,
        xyz @ scene.inv_frac_matrix.T,
        occ,
        adp,
        scene.A,
        scene.B,
        None,
    )


def ds_aniso_oracle(scene: Scene, xyz=None, occ=None, u6=None) -> torch.Tensor:
    """Anisotropic ``F(hkl)`` from the pure-torch eager path."""
    xyz = scene.xyz if xyz is None else xyz
    occ = scene.occ if occ is None else occ
    u6 = scene.u6 if u6 is None else u6
    return _eager_aniso(
        scene.hkl,
        scene.s_vec,
        xyz @ scene.inv_frac_matrix.T,
        occ,
        u6,
        scene.A,
        scene.B,
        None,
    )


def sf_fft_for(
    scene: Scene,
    dtype: torch.dtype = torch.float64,
    *,
    fineness: float = GRID_FINENESS,
    spacegroup: str = "P 1",
):
    """An ``SfFFT`` with its grid already set up, at the fineness policy above.

    Pass ``fineness=1.0`` for a deliberately under-sampled grid; see
    :data:`GRID_FINENESS` for why that is the sampling-limited regime.
    """
    from torchref.model.sf_fft import SfFFT

    sf = SfFFT(
        cell=scene.cell,
        spacegroup=spacegroup,
        max_res=scene.d_min / fineness,
        dtype_float=dtype,
        device=torch.device("cpu"),
    )
    sf.setup_grid()
    return sf


# ---------------------------------------------------------------------------
# Direct kernel access
# ---------------------------------------------------------------------------
# Every production splat is called *directly* here rather than through
# ``build_electron_density`` + ``use_engine``. Two reasons:
#
# 1. **No vacuity risk.** Under ``Engine.AUTO`` a failed accelerator kernel silently
#    falls back to the portable splat (``main.py`` catches and falls through), so a
#    dispatch-driven test can pass while measuring a different kernel than the one it
#    names. Calling the kernel directly settles that by construction.
# 2. **No global-config coupling.** ``SfFFT`` builds its grid through ``get_real_grid``,
#    which reads the *global* ``dtypes.float`` and takes no dtype argument -- so an MPS
#    ``SfFFT`` under this package's float64 pin would try to allocate float64 on MPS and
#    fail. ``ifft`` and ``extract_structure_factor_from_grid`` read no global config at
#    all, so :func:`density_to_F` needs no config switching.
#
# The dispatch ladder is a separate concern, tested in ``test_dispatch.py``.

#: ``name -> (fn, kind, devices, dtypes)``. All six wrappers share one signature,
#: ``(density_map, xyz, adp_or_u, occ, A, B, inv_frac, frac, radius_per_atom)``, so
#: :func:`splat_direct` needs no per-kernel adapter. They did not before the
#: standardization -- the Metal pair took extra unused arguments in a different order,
#: and CUDA had no wrapper at all.
_KERNEL_SPECS = (
    # name                device   dtypes                        kind
    ("cpu_sphere",        "cpu",   (torch.float32, torch.float64), "both"),
    ("portable",          "any",   (torch.float32, torch.float64), "both"),
    ("mps_metal",         "mps",   (torch.float32,),               "both"),
    ("cuda_triton",       "cuda",  (torch.float32,),               "both"),
)


def splat_kernel(name: str, aniso: bool):
    """Resolve a kernel name to its wrapper. Imports are local per backend.

    Backend imports stay inside the function: the Metal module loads MSL source and the
    CUDA one imports Triton, neither of which a CPU-only host should pay for at
    collection time.
    """
    if name == "cpu_sphere":
        from torchref.base.electron_density.kernels.cpu import sphere_splat as m

        return m.add_anisotropic_cpu_sphere_var if aniso else m.add_isotropic_cpu_sphere_var
    if name == "portable":
        from torchref.base.electron_density.kernels.cpu import variable_radius as m

        return m.add_anisotropic_plain_var if aniso else m.add_isotropic_plain_var
    if name == "mps_metal":
        from torchref.base.electron_density.kernels.mps import variable_radius as m

        return m.add_anisotropic_mps_var if aniso else m.add_isotropic_mps_var
    if name == "cuda_triton":
        from torchref.base.electron_density.kernels.cuda import variable_radius as m

        return m.add_anisotropic_cuda_var if aniso else m.add_isotropic_cuda_var
    raise ValueError(f"unknown kernel {name!r}")


def device_supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether ``device`` can hold ``dtype`` at all.

    MPS has no float64 — it is a *device* limitation, not a kernel one, so it belongs
    here and not in the per-kernel table. The portable splat is float64-capable and
    device-agnostic, and without this check it would be offered for ``(mps, float64)``
    and fail inside ``Scene.to``.
    """
    if device.type == "mps" and dtype is torch.float64:
        return False
    return True


def kernels_for(device: torch.device, dtype: torch.dtype):
    """Kernel names that actually run on this ``(device, dtype)``.

    Filtering here rather than skipping inside the test keeps a wrong entry visible: an
    unsupported combination produces no test rather than a passing one.
    """
    if not device_supports_dtype(device, dtype):
        return []
    out = []
    for name, dev, dtypes_ok, _ in _KERNEL_SPECS:
        if dtype not in dtypes_ok:
            continue
        if dev != "any" and dev != device.type:
            continue
        out.append(name)
    return out


def splat_direct(scene: Scene, name: str, xyz=None, occ=None, third=None, *, aniso=False):
    """Call one splat kernel directly and return the density map.

    ``scene`` must already be on the target device and dtype (see :meth:`Scene.to`).
    The per-atom radius is computed here because the kernels take it as an argument --
    ``build_electron_density`` normally does this, and bypassing the dispatch means the
    caller owns it. Same policy call the dispatch makes, so the truncation contract is
    unchanged.
    """
    from torchref.base.electron_density.radius_policy import (
        per_atom_radius_aniso,
        per_atom_radius_iso,
    )
    from torchref.config import get_sigma_cutoff_ed

    xyz = scene.xyz if xyz is None else xyz
    occ = scene.occ if occ is None else occ
    if third is None:
        third = scene.u6 if aniso else scene.adp

    n_sigma = get_sigma_cutoff_ed()
    radius = (
        per_atom_radius_aniso(scene.B, third, n_sigma=n_sigma)
        if aniso
        else per_atom_radius_iso(third, scene.B, n_sigma=n_sigma)
    )
    dims = _grid_dims(scene)
    density_map = torch.zeros(*dims, dtype=xyz.dtype, device=xyz.device)
    fn = splat_kernel(name, aniso)
    return fn(
        density_map, xyz, third, occ, scene.A, scene.B,
        scene.inv_frac_matrix, scene.frac_matrix, radius,
    )


# ---------------------------------------------------------------------------
# Direct-summation kernels, also called directly
# ---------------------------------------------------------------------------
#: ``name -> (device, dtypes)``. ``eager`` is the oracle and is excluded from the
#: candidate list by :func:`ds_kernels_for`; it is registered here so the signature
#: adapter has one place to live.
#:
#: There is **no Metal/MPS direct-summation kernel**: ``Engine.METAL`` returns False from
#: the Triton gate and an MPS device fails ``t.is_cuda``, so DS on MPS *is*
#: ``_checkpointed_*`` running on-device. That is a real production path and it is what the
#: ``mps`` leg of ``checkpointed`` covers.
_DS_KERNEL_SPECS = (
    ("eager",        "any",  (torch.float32, torch.float64)),
    ("checkpointed", "any",  (torch.float32, torch.float64)),
    ("ds_triton",    "cuda", (torch.float32,)),
)


def ds_kernel(name: str, aniso: bool):
    """Resolve a direct-summation kernel name to its function.

    The Triton import is function-local: it pulls in ``triton``, which a CPU-only host
    should not need at collection time.
    """
    from torchref.base.direct_summation import dispatch as D

    if name == "eager":
        return D._eager_aniso if aniso else D._eager_iso
    if name == "checkpointed":
        return D._checkpointed_aniso if aniso else D._checkpointed_iso
    if name == "ds_triton":
        from torchref.base.direct_summation.triton_ds import (
            ds_aniso_triton,
            ds_iso_triton,
        )

        return ds_aniso_triton if aniso else ds_iso_triton
    raise ValueError(f"unknown DS kernel {name!r}")


def ds_kernels_for(device: torch.device, dtype: torch.dtype):
    """Candidate DS kernels on this ``(device, dtype)``. Excludes the oracle."""
    if not device_supports_dtype(device, dtype):
        return []
    return [
        name
        for name, dev, dtypes_ok in _DS_KERNEL_SPECS
        if name != "eager"
        and dtype in dtypes_ok
        and (dev == "any" or dev == device.type)
    ]


def ds_direct(scene: Scene, name: str, xyz_frac=None, occ=None, third=None, *, aniso=False):
    """Call one direct-summation kernel directly and return complex ``F(hkl)``.

    Coordinates are **fractional** here -- unlike the splat kernels, which take Cartesian.
    Carrying both on :class:`Scene` is what keeps that from being a silent error.

    ``ds_iso_triton`` takes no ``max_memory_gb``; the eager and checkpointed paths do. That
    is the only signature difference, and it is absorbed here rather than at call sites.
    """
    xyz_frac = scene.xyz_frac if xyz_frac is None else xyz_frac
    occ = scene.occ if occ is None else occ
    if third is None:
        third = scene.u6 if aniso else scene.adp
    geom = scene.s_vec if aniso else scene.s
    fn = ds_kernel(name, aniso)
    args = (scene.hkl, geom, xyz_frac, occ, third, scene.A, scene.B)
    if name == "ds_triton":
        return fn(*args)
    return fn(*args, None)


def _grid_dims(scene: Scene, fineness: float = GRID_FINENESS):
    """Grid dimensions matching what ``SfFFT.setup_grid`` would choose.

    Derived arithmetically instead of by constructing an ``SfFFT``, so no global dtype is
    read and nothing float64 is allocated on a device that cannot hold it. Rounded to
    even numbers, which is all the kernels care about (they take the shape from
    ``density_map``); FFT-friendliness only matters for speed here.
    """
    spacing = scene.d_min / (3.0 * fineness)
    lengths = [float(scene.cell.data[i]) for i in range(3)]
    return tuple(max(4, 2 * int(round(L / spacing / 2.0))) for L in lengths)


def density_to_F(scene: Scene, density_map: torch.Tensor) -> torch.Tensor:
    """``F(hkl)`` from a density map, without ``SfFFT``.

    ``ifft`` applies the ``V_cell / N`` voxel-volume scaling that makes the result
    directly comparable to a direct-summation atom sum -- no scale factor, no offset.
    Neither this nor ``extract_structure_factor_from_grid`` reads the global dtype
    config, which is what lets an accelerator candidate run while the package is pinned
    to float64 for the oracle.
    """
    from torchref.base.fourier.fft import ifft
    from torchref.base.reciprocal.grid_operations import (
        extract_structure_factor_from_grid,
    )

    volume = float(scene.cell.volume)
    return extract_structure_factor_from_grid(
        ifft(density_map, volume), scene.hkl
    )


def fft_sf(scene: Scene, sf_fft, xyz=None, occ=None, third=None, *, aniso=False):
    """``F(hkl)`` through the density-splat + FFT route, in P1.

    ``apply_symmetry=False`` on both calls: P1 isolates the truncation and sampling
    budget, and a symmetric comparison would cancel the symmetry algebra anyway, since
    both routes call the same ``compute_symmetry_equivalent_hkls`` /
    ``compute_translation_phases``. Symmetry is validated against gemmi instead, in
    ``test_forward.py``.
    """
    xyz = scene.xyz if xyz is None else xyz
    occ = scene.occ if occ is None else occ
    if third is None:
        third = scene.u6 if aniso else scene.adp

    # Match the grid's dtype. Scenes are built in float64 so the DS oracle keeps full
    # precision, but production runs float32 and ``SfFFT`` derives its cell matrices from
    # ``dtype_float`` -- feeding float64 atoms into a float32 grid raises inside
    # ``_canonical_setup``. Casting is differentiable, so gradient tests still reach the
    # original float64 leaf.
    target = sf_fft.dtype_float
    if xyz.dtype != target:
        xyz, occ, third = xyz.to(target), occ.to(target), third.to(target)
    A, B = scene.A.to(target), scene.B.to(target)

    empty1 = xyz.new_zeros(0)
    empty3 = xyz.new_zeros(0, 3)
    empty5 = A.new_zeros(0, 5)
    if aniso:
        dm = sf_fft.build_density_map(
            empty3,
            empty1,
            empty1,
            empty5,
            empty5,
            xyz_aniso=xyz,
            u_aniso=third,
            occ_aniso=occ,
            A_aniso=A,
            B_aniso=B,
            apply_symmetry=False,
        )
    else:
        dm = sf_fft.build_density_map(xyz, third, occ, A, B, apply_symmetry=False)
    return sf_fft.map_to_structure_factors(dm, scene.hkl, apply_symmetry=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def synthetic_obs(
    F_ref: torch.Tensor, *, rel_sigma: float = 0.10, seed: int = 101
) -> torch.Tensor:
    """Pseudo-observed amplitudes: ``|F_ref|`` perturbed by ``rel_sigma`` relative noise.

    This exists to make :func:`ls_target` well-posed. A least-squares residual taken
    directly against ``|F_ref|`` has its minimum exactly where the two routes agree, so
    near convergence the gradient comparison becomes partly self-referential -- the
    quantity being differentiated is the very difference under test. Offsetting the
    target by a fixed random 10% breaks that: the residual is O(0.1|F|), dominated by
    synthetic misfit rather than by the FFT-vs-DS discrepancy, which is the regime a
    refinement actually operates in.

    Deterministic by seed, and generated once from the oracle, so the candidate and the
    oracle are scored against the *same* target. Multiplicative rather than additive
    noise, so weak reflections are not swamped.
    """
    g = torch.Generator().manual_seed(seed)
    amp = F_ref.abs()
    noise = torch.randn(amp.shape, generator=g, dtype=amp.dtype)
    return (amp * (1.0 + rel_sigma * noise)).clamp_min(0.0)


def ls_target(
    F: torch.Tensor, f_obs: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Least-squares amplitude residual ``sum(w * (|F| - f_obs)^2)``.

    The quantitative target for the gradient and HVP gates. Unweighted by default, so it
    is dominated by strong reflections in the way an unweighted crystallographic LS
    residual is.

    Well-conditioned in a way :func:`phase_sensitive_loss` is not: the per-reflection
    residuals are random-signed at O(0.1|F|), so the sum grows like sqrt(N) instead of
    cancelling, and the relative-error metric measures accuracy rather than the
    conditioning of a near-cancelled sum.
    """
    r = F.abs().to(f_obs.dtype) - f_obs
    return (r * r).sum() if weights is None else (weights * r * r).sum()


# Why there is no phase-sensitive loss here
# -----------------------------------------
# An earlier draft used ``(F.real**2 + 2*F.imag).sum()`` -- asymmetric in the imaginary
# part -- on the theory that a squared target cannot detect a kernel that conjugates F.
# Both premises were wrong.
#
# It is not testable this way. A crystallographic amplitude target is phase-blind by
# construction, and so are its derivatives:
#
#     d|F|/dx = Re(F* dF/dx) / |F|
#
# Conjugate the whole calculation -- F -> F*, dF/dx -> (dF/dx)* -- and
# Re(F* dF/dx) -> Re(F (dF/dx)*) = Re((F* dF/dx)*) = Re(F* dF/dx), unchanged. A global
# phase convention is an unobservable gauge choice for such a target, so asserting a
# particular imaginary sign through it tests nothing physical.
#
# And it actively harmed the measurement. A linear term in F.imag sums signed quantities
# that largely cancel, so its gradient is a small difference of large numbers and every
# per-reflection error is amplified in relative terms. Measured on the same FFT-vs-DS xyz
# gradient comparison: 2.7e-01 under the asymmetric functional against ~6e-03 under
# ``ls_target`` -- a 40x artefact of conditioning, not of accuracy.
#
# Phase *convention* still matters, because anomalous scattering and difference maps are
# phase-sensitive and because the two routes must agree with each other. It is gated where
# it is observable: on the forward complex F, against gemmi and between routes, in
# ``test_forward.py``. See ``test_ls_target_is_phase_blind`` there, which pins the
# invariance above so this reasoning cannot quietly rot.


def rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).norm() / ref.norm())


def max_rel(got: torch.Tensor, ref: torch.Tensor, *, significance: float = 1e-3) -> float:
    """Worst per-reflection relative error, over *significant* reflections only.

    Reflections with ``|F_ref| <= significance * max|F_ref|`` are excluded. Without that
    filter this metric is dominated by systematic absences: in ``P 65 2 2`` the
    extinguished reflections have ``|F|`` at round-off, so a ratio there reports ~6e-3
    while the rel-L2 over the same set is 4e-8. That is a property of dividing by zero,
    not a disagreement, and gating on it would mean either a meaningless tolerance or
    deleting the symmetry test.

    Reported alongside :func:`rel_l2` rather than instead of it -- rel L2 can hide a
    single badly wrong strong reflection, and this cannot.
    """
    scale = ref.abs().max()
    keep = ref.abs() > significance * scale
    if not bool(keep.any()):
        return 0.0
    return float(((got[keep] - ref[keep]).abs() / ref[keep].abs()).max())


def best_fit_scale(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Least-squares scale ``k`` minimizing ``||k*|ref| - |got|||``.

    Compared on amplitudes so a phase convention difference does not leak into the
    scale estimate; phase is gated separately.
    """
    g, r = got.abs().reshape(-1), ref.abs().reshape(-1)
    return float((r * g).sum() / (r * r).sum())
