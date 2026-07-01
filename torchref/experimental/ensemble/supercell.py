"""
Supercell layout + OpenMM System replication for quasi-crystal ensemble Amber.

.. warning::

   Experimental — part of ``torchref.experimental.ensemble``. The API and
   behaviour may change or be removed without notice.

Extracted from :mod:`torchref.experimental.targets.amber_target` so the
single-molecule :class:`~torchref.experimental.targets.amber_target.AmberTarget`
base carries no crystal-specific code. Only
:class:`~torchref.experimental.ensemble.quasi_crystal_amber.QuasiCrystalAmberTarget`
uses these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Supercell layout
# ---------------------------------------------------------------------------
#
# For ensemble refinement, the disorder copies are tiled along axis a in a
# k × 1 × 1 supercell, with the spacegroup's full sym expansion applied
# within each small cell. The two "filling axes" of the ensemble are:
#
#   - sym mate j ∈ [0, N_sym): the crystallographic symmetry copies within
#     one small cell (1 to N_sym = order of the spacegroup).
#   - disorder copy d ∈ [0, k): the supercell tile (the same molecule
#     conformer at a shifted small-cell position).
#
# Member index m maps to (d, j) by ``m = d * N_sym + j`` so members are
# block-ordered: the first N_sym members fill tile d=0, the next N_sym fill
# tile d=1, etc. This matches the layout of ``EnsembleModel.xyz_per_member``.
#
# Total members ``N = k * N_sym``.


@dataclass
class SupercellLayout:
    """k × 1 × 1 supercell tiled along axis a, with sym expansion per small cell.

    The class is just a container + a position-transform method — no OpenMM
    state lives here. ``compute_member_positions`` is a pure differentiable
    tensor op so gradients flow from supercell positions back to the model's
    per-member xyz tensor.

    Parameters
    ----------
    cell : torch.Tensor, (3, 3)
        Orthogonalization matrix ``B`` such that ``cart_col = B @ frac_col``
        (column convention). Columns are lattice vectors. This matches
        :attr:`torchref.symmetry.Cell.fractional_matrix`.
    sym_rotations : torch.Tensor, (n_sym, 3, 3)
        Spacegroup rotation matrices in fractional coordinates (from
        :attr:`torchref.symmetry.SpaceGroup.matrices`).
    sym_translations : torch.Tensor, (n_sym, 3)
        Spacegroup translations in fractional coordinates.
    n_disorder : int
        Number of disorder copies tiled along axis a (k ≥ 1).

    Notes
    -----
    For member m = (d, j) (where ``d = m // n_sym`` and ``j = m % n_sym``)
    and atom a with Cartesian ASU coords ``r_a`` (one row of the
    ``model_xyz`` argument to :meth:`compute_member_positions`), the
    supercell-Cartesian position is::

        r_sym_cart = R_cart_j @ r_a + t_cart_j        (apply sym op j)
        r_supercell = r_sym_cart + d * cell[:, 0]     (shift to tile d)

    where ``R_cart_j = B @ R_j @ B^-1`` and ``t_cart_j = B @ t_j``
    (rotation and translation conjugated into Cartesian frame).
    """

    cell: torch.Tensor
    sym_rotations: torch.Tensor
    sym_translations: torch.Tensor
    n_disorder: int

    def __post_init__(self) -> None:
        if tuple(self.cell.shape) != (3, 3):
            raise ValueError(
                f"cell must be (3, 3); got {tuple(self.cell.shape)}"
            )
        if (
            self.sym_rotations.ndim != 3
            or tuple(self.sym_rotations.shape[-2:]) != (3, 3)
        ):
            raise ValueError(
                f"sym_rotations must be (n_sym, 3, 3); got "
                f"{tuple(self.sym_rotations.shape)}"
            )
        n_sym = int(self.sym_rotations.shape[0])
        if tuple(self.sym_translations.shape) != (n_sym, 3):
            raise ValueError(
                f"sym_translations must be ({n_sym}, 3); got "
                f"{tuple(self.sym_translations.shape)}"
            )
        if int(self.n_disorder) < 1:
            raise ValueError(
                f"n_disorder must be >= 1; got {self.n_disorder}"
            )

    @property
    def n_sym(self) -> int:
        return int(self.sym_rotations.shape[0])

    @property
    def n_members(self) -> int:
        return int(self.n_disorder) * self.n_sym

    @property
    def supercell_vectors(self) -> torch.Tensor:
        """(3, 3) cell matrix for the ``k·a × b × c`` supercell (same column
        convention as :attr:`cell`)."""
        a_vec, b_vec, c_vec = self.cell.unbind(dim=1)
        return torch.stack(
            [a_vec * int(self.n_disorder), b_vec, c_vec], dim=1
        )

    def compute_member_positions(
        self, model_xyz: torch.Tensor
    ) -> torch.Tensor:
        """Map per-member ASU coords to supercell Cartesian positions.

        Parameters
        ----------
        model_xyz : torch.Tensor, (N, n_atoms, 3)
            Per-member Cartesian coordinates in the ASU frame.
            ``N`` must equal :attr:`n_members`.

        Returns
        -------
        torch.Tensor, (N, n_atoms, 3)
            Cartesian positions in the supercell frame (Å). Differentiable
            in ``model_xyz``.
        """
        if model_xyz.ndim != 3 or model_xyz.shape[-1] != 3:
            raise ValueError(
                f"model_xyz must be (N, n_atoms, 3); got "
                f"{tuple(model_xyz.shape)}"
            )
        N = int(model_xyz.shape[0])
        if N != self.n_members:
            raise ValueError(
                f"model_xyz N={N} does not match layout n_members="
                f"{self.n_members}"
            )

        # Move cell + sym ops to the model's device/dtype, treat as constants.
        device = model_xyz.device
        dtype = model_xyz.dtype
        B = self.cell.to(device=device, dtype=dtype).detach()
        R_frac = self.sym_rotations.to(device=device, dtype=dtype).detach()
        t_frac = self.sym_translations.to(device=device, dtype=dtype).detach()

        # Cartesian sym ops: R_cart_j = B @ R_frac_j @ B^-1; t_cart_j = B @ t_frac_j.
        B_inv = torch.linalg.inv(B)
        R_cart = B @ R_frac @ B_inv  # (n_sym, 3, 3)
        # t_frac is (n_sym, 3) row vectors; t_cart row = (B @ t_col).T = t_row @ B.T.
        t_cart = t_frac @ B.T  # (n_sym, 3)

        # Per-member (d, j) decomposition.
        n_sym = self.n_sym
        member_idx = torch.arange(N, device=device)
        j_idx = member_idx % n_sym
        d_idx = member_idx // n_sym

        R_per_member = R_cart.index_select(0, j_idx)  # (N, 3, 3)
        t_per_member = t_cart.index_select(0, j_idx)  # (N, 3)

        # Apply rotation: r_sym[n, a, i] = R[n, i, k] * r[n, a, k] (column conv).
        rotated = torch.einsum("nik,nak->nai", R_per_member, model_xyz)
        rotated = rotated + t_per_member.unsqueeze(1)  # broadcast over atoms

        # Tile shift: d * a_vec where a_vec = B[:, 0].
        a_vec = B[:, 0]
        tile_offset = (
            d_idx.to(dtype).unsqueeze(-1) * a_vec.unsqueeze(0)
        )  # (N, 3)
        return rotated + tile_offset.unsqueeze(1)


def _reduce_box_vectors_for_openmm(box: np.ndarray) -> np.ndarray:
    """Canonicalize a (3, 3) column-major lattice matrix to OpenMM's
    reduced-form periodic box.

    OpenMM requires:
      a = (a_x, 0, 0), a_x > 0
      b = (b_x, b_y, 0), b_y > 0
      c = (c_x, c_y, c_z), c_z > 0
      |b_x| < a_x/2,  |c_x| < a_x/2,  |c_y| < b_y/2 (strict — float-precision-tight)

    The crystallographic cell starts in this frame already (a along x, etc.)
    but real data trips the strict check via two paths:

    1. tiny float-noise non-zero entries (e.g. ``c_x ≈ 1e-5`` because the
       supplied ``cos(90°)`` isn't exactly 0). Fix: zero entries below
       ``1e-5 × cell scale``.
    2. ``b_x`` sitting on the ``|b_x| = a_x/2`` boundary (hexagonal γ=120°
       lattices like 3GR5: a=90.67, γ=120 → b_x = -45.335 = -a_x/2). Fix:
       clamp ``b_x`` to just inside the boundary (a 1e-9 relative shift is
       sub-µÅ on an Å-scale cell — undetectable in any force calc).

    We don't do general lattice reduction (subtract n·a from b for vectors
    far outside the reduced form) because every real crystallographic cell
    already arrives in the standard reduced frame; only the boundary float-
    precision case needs cleanup.
    """
    import math as _math  # noqa: PLC0415

    a = box[:, 0].astype(np.float64).copy()
    b = box[:, 1].astype(np.float64).copy()
    c = box[:, 2].astype(np.float64).copy()

    # 1) Drop sub-µÅ noise in entries that should be exactly zero.
    scale = max(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c))
    tol = 1e-5 * scale
    if abs(a[1]) < tol: a[1] = 0.0
    if abs(a[2]) < tol: a[2] = 0.0
    if abs(b[2]) < tol: b[2] = 0.0
    if abs(c[0]) < tol: c[0] = 0.0
    if abs(c[1]) < tol: c[1] = 0.0

    # 2) Clamp boundary cases inward by a tiny fraction so OpenMM's strict
    #    "< a_x/2" passes. copysign keeps the sign of the original component.
    def _clamp_in(v: float, half: float) -> float:
        if abs(v) >= half * (1 - 1e-12):
            return _math.copysign(half * (1.0 - 1e-9), v)
        return v

    if a[0] > 0:
        b[0] = _clamp_in(b[0], a[0] / 2.0)
        c[0] = _clamp_in(c[0], a[0] / 2.0)
    if b[1] > 0:
        c[1] = _clamp_in(c[1], b[1] / 2.0)

    return np.stack([a, b, c], axis=1)


def _replicate_to_supercell_system(
    template_system,
    layout: SupercellLayout,
    pme_cutoff_nm: float = 1.0,
    ewald_error_tolerance: float = 5e-4,
):
    """Build a unified OpenMM ``System`` for ``layout``'s k·N_sym replicas.

    Walks the template (single-molecule) System's forces and replicates each
    for every member m, offsetting atom indices by ``m * N_template``. The
    NonbondedForce is rebuilt for PME on the supercell PBC; per-member 1-2/
    1-3/1-4 exceptions are replicated. **No cross-member exceptions** — those
    interactions are the crystal-contact forces that do the regularization.

    Parameters
    ----------
    template_system : openmm.System
        Single-molecule System (as produced by AmberTarget's existing
        ``_build_omm_system`` path).
    layout : SupercellLayout
        Disorder + sym-expansion description; supplies ``n_members`` and
        ``supercell_vectors`` (Å, column-vector convention).
    pme_cutoff_nm : float
        Nonbonded cutoff in nm (10 Å = 1.0 nm is typical for protein PME).
    ewald_error_tolerance : float
        PME accuracy tolerance; 5e-4 is OpenMM's default for production work.

    Returns
    -------
    openmm.System
        Supercell System with ``layout.n_members * template_system
        .getNumParticles()`` particles, PBC set to ``layout.supercell_vectors``,
        PME NonbondedForce.

    Raises
    ------
    NotImplementedError
        If the template contains a force type the replicator doesn't handle
        (so unsupported forces surface immediately rather than being silently
        dropped).
    """
    import openmm  # noqa: PLC0415
    import openmm.unit as u_omm  # noqa: PLC0415

    n_template = template_system.getNumParticles()
    n_members = layout.n_members

    new_system = openmm.System()

    # ---- Particles ----
    for _m in range(n_members):
        for i in range(n_template):
            new_system.addParticle(template_system.getParticleMass(i))

    # ---- Periodic box (Å → nm), canonicalized to OpenMM reduced form ----
    sv_ang_raw = layout.supercell_vectors.detach().cpu().numpy()
    sv_ang = _reduce_box_vectors_for_openmm(sv_ang_raw)
    sv_nm = sv_ang / 10.0
    new_system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(float(sv_nm[0, 0]), float(sv_nm[1, 0]), float(sv_nm[2, 0]))
        * u_omm.nanometer,
        openmm.Vec3(float(sv_nm[0, 1]), float(sv_nm[1, 1]), float(sv_nm[2, 1]))
        * u_omm.nanometer,
        openmm.Vec3(float(sv_nm[0, 2]), float(sv_nm[1, 2]), float(sv_nm[2, 2]))
        * u_omm.nanometer,
    )

    # ---- Forces ----
    for force in template_system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            new_force = openmm.HarmonicBondForce()
            for m in range(n_members):
                off = m * n_template
                for k in range(force.getNumBonds()):
                    p1, p2, length, kK = force.getBondParameters(k)
                    new_force.addBond(p1 + off, p2 + off, length, kK)
            new_force.setUsesPeriodicBoundaryConditions(True)
            new_system.addForce(new_force)

        elif isinstance(force, openmm.HarmonicAngleForce):
            new_force = openmm.HarmonicAngleForce()
            for m in range(n_members):
                off = m * n_template
                for k in range(force.getNumAngles()):
                    p1, p2, p3, theta, kK = force.getAngleParameters(k)
                    new_force.addAngle(
                        p1 + off, p2 + off, p3 + off, theta, kK
                    )
            new_force.setUsesPeriodicBoundaryConditions(True)
            new_system.addForce(new_force)

        elif isinstance(force, openmm.PeriodicTorsionForce):
            new_force = openmm.PeriodicTorsionForce()
            for m in range(n_members):
                off = m * n_template
                for k in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, kK = (
                        force.getTorsionParameters(k)
                    )
                    new_force.addTorsion(
                        p1 + off, p2 + off, p3 + off, p4 + off,
                        periodicity, phase, kK,
                    )
            new_force.setUsesPeriodicBoundaryConditions(True)
            new_system.addForce(new_force)

        elif isinstance(force, openmm.NonbondedForce):
            new_force = openmm.NonbondedForce()
            for _m in range(n_members):
                for i in range(n_template):
                    q, sigma, eps = force.getParticleParameters(i)
                    new_force.addParticle(q, sigma, eps)
            for m in range(n_members):
                off = m * n_template
                for k in range(force.getNumExceptions()):
                    i, j, q_prod, sigma, eps = force.getExceptionParameters(k)
                    new_force.addException(
                        i + off, j + off, q_prod, sigma, eps,
                    )
            new_force.setNonbondedMethod(openmm.NonbondedForce.PME)
            new_force.setCutoffDistance(pme_cutoff_nm * u_omm.nanometer)
            new_force.setEwaldErrorTolerance(float(ewald_error_tolerance))
            new_force.setUseDispersionCorrection(True)
            new_system.addForce(new_force)

        elif isinstance(force, openmm.CMMotionRemover):
            # Strip — per-atom forces are required for autograd; the template
            # path already removes it on the original System, but be robust.
            continue

        else:
            raise NotImplementedError(
                f"Force type {type(force).__name__} not yet supported in "
                f"_replicate_to_supercell_system"
            )

    return new_system
