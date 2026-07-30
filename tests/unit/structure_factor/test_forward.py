"""Forward structure factors: gemmi -> direct summation -> the FFT/splat route.

Read the package docstring first for the oracle chain and, in particular, for what the
gemmi comparison does and does not prove -- the ITC92 table was generated from gemmi, so
``f(s)`` is shared and cancels. :func:`test_stored_table_matches_gemmi` is the only test
here that constrains the form factors themselves.
"""

from __future__ import annotations

import pytest
import torch

from torchref.model.sf_ds import SfDS

from . import (
    COS_MIN_DS,
    RTOL_DS_F32,
    RTOL_DS_F64,
    ATOL_TABLE_VS_GEMMI,
    COS_MIN,
    MAXREL_VS_GEMMI,
    RTOL_AMPLITUDE,
    RTOL_BACKEND_F32,
    RTOL_BACKEND_F64,
    RTOL_VS_GEMMI,
    SCALE_TOL,
    SCALE_TOL_GEMMI,
)
from . import helpers as H
from .conftest import DEVICE_DTYPE_KERNELS, DS_DEVICE_DTYPE_KERNELS

pytestmark = pytest.mark.unit


def _lbl(pin):
    """Label for the backend a pin parametrization selected."""
    return "portable" if pin else "default"

# float32 first: it is the production dtype, so it is the case that matters and the one
# these gates are calibrated on. float64 separates truncation error from float32 noise.
_DTYPES = [torch.float32, torch.float64]


def _complex_cos(got, ref):
    """Cosine of two complex F vectors, treating them as real 2R-vectors.

    Catches a conjugated result, which an amplitude comparison cannot.
    """
    g = torch.view_as_real(got.reshape(-1)).reshape(-1).double()
    r = torch.view_as_real(ref.reshape(-1)).reshape(-1).double()
    return float((g @ r) / (g.norm() * r.norm()).clamp_min(1e-30))


def _report(tag, got, ref):
    """Print the three metrics, so a widened tolerance always has a number behind it."""
    r, m, k = H.rel_l2(got, ref), H.max_rel(got, ref), H.best_fit_scale(got, ref)
    print(f"  {tag:34s} relL2 {r:.3e}  maxrel {m:.3e}  scale {k:.9f}")
    return r, m, k


# ---------------------------------------------------------------------------
# The one place gemmi is an independent reference for f(s)
# ---------------------------------------------------------------------------
def test_stored_table_matches_gemmi():
    """The stored ITC92 ``.pt`` must still equal gemmi's live table.

    ``torchref/data/itc92_scattering_factors.pt`` is generated from gemmi by
    ``torchref/scripts/generate_scattering_table.py``, then used at runtime so gemmi is
    not a runtime dependency. Nothing re-checked the two after generation, so a
    corrupted regeneration or a gemmi release that revised coefficients would go
    unnoticed -- and would be invisible to every other test in this package, because
    those compare torchref against gemmi using the *same* coefficients on both sides.

    Also pins the storage convention: torchref keeps gemmi's four Gaussians plus the
    constant ``c`` folded in as a fifth with ``B = 0``, so ``exp(0) = 1`` reproduces the
    additive term.

    Compared in **float32**, and required to be bit-exact there. The stored table is
    float32 (``(104, 5)``), while gemmi's Python ``it92`` returns doubles -- so a
    float64 comparison can only ever agree to float32 rounding, measured at 5.9e-08
    worst case over Z=1..103, which is half of float32 eps. Asserting exact equality
    after rounding gemmi's doubles to float32 is the stronger statement: it permits the
    intended precision loss and nothing else, where a 1e-7 rtol would also permit a
    genuinely wrong coefficient that happened to land nearby.

    Note this caps the oracle: asking ``get_scattering_params_by_z`` for float64 widens
    float32 values rather than recovering the doubles, so ``f(s)`` in the "float64"
    oracle carries ~6e-8 relative error. That is the floor behind ``RTOL_VS_GEMMI`` and
    it is the correct trade for a float32 production path.
    """
    gemmi = pytest.importorskip("gemmi")
    from torchref.base.scattering.scattering_table import get_scattering_params_by_z

    checked = 0
    for z in range(1, 104):
        element = gemmi.Element(z)
        coef = element.it92
        if coef is None:
            continue
        A, B = get_scattering_params_by_z(torch.tensor([z]), dtype=torch.float32)
        assert A.shape == (1, 5) and B.shape == (1, 5), f"Z={z} unexpected table shape"

        want_a = torch.tensor(list(coef.a), dtype=torch.float32)
        want_b = torch.tensor(list(coef.b), dtype=torch.float32)
        want_c = torch.tensor(coef.c, dtype=torch.float32)

        assert torch.equal(A[0, :4], want_a), (
            f"Z={z} ({element.name}): 'a' drifted from gemmi -- "
            f"stored {A[0, :4].tolist()} vs gemmi {want_a.tolist()}"
        )
        assert torch.equal(B[0, :4], want_b), (
            f"Z={z} ({element.name}): 'b' drifted from gemmi -- "
            f"stored {B[0, :4].tolist()} vs gemmi {want_b.tolist()}"
        )
        assert torch.equal(A[0, 4], want_c), (
            f"Z={z} ({element.name}): constant term c not in slot 4 -- "
            f"stored {float(A[0, 4])} vs gemmi {float(want_c)}"
        )
        assert float(B[0, 4]) == 0.0, (
            f"Z={z} ({element.name}): slot 4 must have B=0 so exp(0)=1 gives +c"
        )
        checked += 1

    assert checked >= 90, f"only {checked} elements checked -- table lookup is not working"


def test_ls_target_is_phase_blind(scene_small):
    """An amplitude target and its gradients are invariant under conjugating ``F``.

    This is why phase convention is gated on forward complex values -- here and against
    gemmi -- and *not* through the derivative tests. Since

        d|F|/dx = Re(F* dF/dx) / |F|

    sending ``F -> F*`` and ``dF/dx -> (dF/dx)*`` leaves ``Re(F* dF/dx)`` unchanged, so a
    global phase convention is an unobservable gauge choice for such a target. An earlier
    draft tried to catch a conjugated kernel with a functional linear in ``F.imag``; that
    cannot work for a physical target and cost a 40x conditioning artefact in the relative
    error (2.7e-01 vs ~6e-03 for the same comparison). See the note in ``helpers.py``.

    Phase still matters -- anomalous scattering and difference maps are phase-sensitive,
    and the two routes must agree with each other -- which is exactly what
    :func:`test_ds_matches_gemmi_iso_p1` and the complex-cosine assertion in
    :func:`test_fft_matches_ds_amplitudes` cover.
    """
    s = scene_small
    with torch.no_grad():
        F = H.ds_iso_oracle(s)
    obs = H.synthetic_obs(F)

    def target(x, conjugate):
        Fx = H.ds_iso_oracle(s, x, s.occ, s.adp)
        return H.ls_target(Fx.conj() if conjugate else Fx, obs)

    x1 = s.xyz.clone().requires_grad_(True)
    x2 = s.xyz.clone().requires_grad_(True)
    v1, v2 = target(x1, False), target(x2, True)
    (g1,) = torch.autograd.grad(v1, x1)
    (g2,) = torch.autograd.grad(v2, x2)

    assert v1.item() == pytest.approx(v2.item(), rel=1e-12), (
        "conjugating F changed the amplitude target value"
    )
    assert H.rel_l2(g2, g1) < 1e-12, (
        "conjugating F changed the amplitude target's gradient, so this target is not "
        "phase-blind after all -- the derivative tests could then gate phase convention "
        "and the reasoning in helpers.py needs revisiting"
    )


# ---------------------------------------------------------------------------
# gemmi -> DS
# ---------------------------------------------------------------------------
def test_ds_matches_gemmi_iso_p1(gemmi_iso_p1):
    """Isotropic P1. Constrains the exponential sum, the 2*pi factors and the B
    convention, with symmetry taken out of the picture."""
    scene, structure = gemmi_iso_p1
    F_gemmi = H.gemmi_sf(structure, scene.hkl_list)
    with torch.no_grad():
        F_ds = H.ds_iso_oracle(scene)

    print(f"\n{scene.n_atoms} atoms, {scene.n_refl} reflections, P1")
    rel, mrel, scale = _report("DS vs gemmi (iso, P1)", F_ds, F_gemmi)
    assert rel < RTOL_VS_GEMMI
    assert mrel < MAXREL_VS_GEMMI
    assert abs(scale - 1.0) < SCALE_TOL_GEMMI


def test_ds_matches_gemmi_aniso_p1(gemmi_aniso_p1):
    """Anisotropic P1 on a structure where every atom carries ANISOU.

    Compares complex ``F``, not ``|F|``: an amplitude-only check passes a phase-sign
    error, and the aniso Debye-Waller factor is real, so a sign slip elsewhere would
    hide behind it.
    """
    scene, structure = gemmi_aniso_p1
    assert (scene.u6.abs().sum(dim=1) > 0).all(), (
        "every atom must carry ANISOU or this test degenerates to the isotropic case"
    )
    F_gemmi = H.gemmi_sf(structure, scene.hkl_list)
    with torch.no_grad():
        F_ds = H.ds_aniso_oracle(scene)

    print(f"\n{scene.n_atoms} ANISOU atoms, {scene.n_refl} reflections, P1")
    rel, mrel, scale = _report("DS vs gemmi (aniso, P1)", F_ds, F_gemmi)
    assert rel < RTOL_VS_GEMMI
    assert mrel < MAXREL_VS_GEMMI
    assert abs(scale - 1.0) < SCALE_TOL_GEMMI


@pytest.mark.parametrize(
    "perm,label",
    [((0, 1, 2, 4, 3, 5), "U12<->U13"), ((0, 1, 2, 5, 4, 3), "reversed off-diagonals")],
)
def test_aniso_ordering_is_actually_constrained(gemmi_aniso_p1, perm, label):
    """Non-vacuity guard for the test above.

    A tight tolerance is only meaningful if a wrong convention would breach it. Permuting
    the ADP off-diagonals must move the residual far outside ``RTOL_VS_GEMMI`` -- measured
    1.35e-02 for the ``U12<->U13`` swap and 6.08e-03 for the reversal, five orders of
    magnitude clear. Without this, ``[U11,U22,U33,U12,U13,U23]`` would be an assumption
    rather than a verified fact.
    """
    scene, structure = gemmi_aniso_p1
    F_gemmi = H.gemmi_sf(structure, scene.hkl_list)
    with torch.no_grad():
        F_perm = H.ds_aniso_oracle(scene, u6=scene.u6[:, list(perm)])

    rel = H.rel_l2(F_perm, F_gemmi)
    print(f"\n  permuted {label:24s} relL2 {rel:.3e}  (gate is {RTOL_VS_GEMMI:.0e})")
    assert rel > 100 * RTOL_VS_GEMMI, (
        f"permuting {label} changed the result by only {rel:.2e}; the aniso comparison "
        "does not constrain the off-diagonal ordering"
    )


def test_sfds_matches_gemmi_with_symmetry(gemmi_iso_symmetry):
    """Symmetry algebra against a second implementation, in a hexagonal group.

    ``P 65 2 2`` is chosen because ``h.R`` and ``R.h`` agree for symmetric rotation
    matrices and differ for trigonal/hexagonal ones -- so an orthorhombic or tetragonal
    entry could not detect the transpose error that
    ``tests/unit/symmetry/test_hkl_symmetry_gemmi.py`` documents.

    This is also the only symmetric comparison in the package. A DS-vs-FFT check cannot
    validate symmetry, because both routes call the same
    ``compute_symmetry_equivalent_hkls`` / ``compute_translation_phases`` and the shared
    algebra cancels; gemmi does not share it.
    """
    scene, structure = gemmi_iso_symmetry
    assert len(structure.cell.images) > 0, "structure was not set up with symmetry"

    F_gemmi = H.gemmi_sf(structure, scene.hkl_list)
    ds = SfDS(
        cell=scene.cell,
        spacegroup=scene.spacegroup,
        dtype_float=torch.float64,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        F_sym, _ = ds.compute_structure_factors(
            scene.hkl, scene.xyz, scene.adp, scene.occ, scene.A, scene.B,
            apply_symmetry=True,
        )

    print(f"\n{scene.spacegroup}, {len(structure.cell.images) + 1} operations")
    rel, mrel, scale = _report("SfDS(sym) vs gemmi", F_sym, F_gemmi)
    assert rel < RTOL_VS_GEMMI
    assert mrel < MAXREL_VS_GEMMI
    assert abs(scale - 1.0) < SCALE_TOL_GEMMI


# ---------------------------------------------------------------------------
# DS -> FFT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pin", [False, True], ids=["default", "portable"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_fft_matches_ds_amplitudes(scene_fine, oracle_fine, kind, dtype, pin):
    """The headline gate: the map route reproduces the analytic answer.

    The DS reference stays in float64 whatever the candidate's dtype -- an oracle should
    not inherit the precision of the thing it is judging.
    """
    aniso = kind == "aniso"
    sf = H.sf_fft_for(scene_fine, dtype)
    with H.maybe_portable(pin), torch.no_grad():
        F_fft = H.fft_sf(scene_fine, sf, aniso=aniso).to(torch.complex128)
    F_ds = oracle_fine[f"{kind}_F"]

    print(f"\ngrid {tuple(int(v) for v in sf.gridsize)}, {kind}, {dtype}, {_lbl(pin)}")
    rel, mrel, scale = _report(f"FFT vs DS ({kind})", F_fft, F_ds)
    assert rel < RTOL_AMPLITUDE, f"{kind}/{dtype}/{_lbl(pin)}: rel L2 {rel:.3e}"
    cos = float(
        (F_fft.conj() * F_ds).real.sum()
        / (F_fft.abs().norm() * F_ds.abs().norm()).clamp_min(1e-30)
    )
    assert cos > COS_MIN, f"phase disagreement: cos {cos:.6f}"


def test_fft_absolute_scale(scene_fine, oracle_fine):
    """Absolute scale, which nothing in the suite checked before.

    The FFT route picks up a voxel-volume factor from ``ifft``
    (``torchref/base/fourier/fft.py``) while DS is a bare atom sum. A factor of ``V``,
    of ``N`` voxels, or of ``V/N`` would leave every ratio-, cosine- and parity-based
    gate in the repo perfectly happy -- correlation stays 1.0 under a global rescale.
    """
    sf = H.sf_fft_for(scene_fine)
    with torch.no_grad():
        F_fft = H.fft_sf(scene_fine, sf)
    scale = H.best_fit_scale(F_fft, oracle_fine["iso_F"])
    print(f"\n  best-fit scale {scale:.9f}")
    assert abs(scale - 1.0) < SCALE_TOL, (
        f"absolute scale off by {abs(scale - 1.0):.3e}; suspect a voxel-volume or "
        f"grid-size normalization factor"
    )


def test_fft_matches_ds_monoclinic(scene_monoclinic, oracle_monoclinic):
    """beta = 115 deg. A truncation that is spherical in fractional rather than
    Cartesian space, or centred on a grid node rather than the atom, diverges most in a
    strongly non-orthogonal metric -- measured 5.0e-3 rel L2 for the node-centred
    variant at this angle, against 9.9e-4 orthorhombic."""
    sf = H.sf_fft_for(scene_monoclinic)
    with torch.no_grad():
        F_fft = H.fft_sf(scene_monoclinic, sf)
    rel, _, _ = _report("FFT vs DS (beta=115)", F_fft, oracle_monoclinic["iso_F"])
    assert rel < RTOL_AMPLITUDE


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_backend_parity_auto_vs_eager(scene_fine, kind, dtype):
    """AUTO and EAGER implement the same truncation contract, so they must agree to
    float noise -- far tighter than either agrees with DS. This separates a kernel bug
    from the shared truncation error that ``test_fft_matches_ds_amplitudes`` bounds."""
    aniso = kind == "aniso"
    sf = H.sf_fft_for(scene_fine, dtype)
    with torch.no_grad():
        with H.maybe_portable(False):
            auto = H.fft_sf(scene_fine, sf, aniso=aniso).to(torch.complex128)
        with H.maybe_portable(True):
            eager = H.fft_sf(scene_fine, sf, aniso=aniso).to(torch.complex128)
    tol = RTOL_BACKEND_F32 if dtype is torch.float32 else RTOL_BACKEND_F64
    rel = H.rel_l2(auto, eager)
    print(f"\n  AUTO vs EAGER ({kind}, {dtype}): relL2 {rel:.3e}  gate {tol:.0e}")
    assert rel < tol


# ---------------------------------------------------------------------------
# Truncation budget
# ---------------------------------------------------------------------------
def test_nsigma_reduces_truncation_error(scene_fine, oracle_fine, sigma_cutoff):
    """Raising the cutoff must monotonically improve agreement with DS.

    Asserted as a trend rather than an absolute number, for two reasons: it puts a test
    behind the N_sigma claim at ``torchref/config.py:196`` which until now existed only
    as a comment, and it stays valid whichever default the cutoff ends up at, so this
    test does not have to be retuned alongside that decision.
    """
    # Explicitly finer than production: at d_min/3 the cutoff is not the binding
    # constraint (5.89e-3 -> 5.86e-3 then flat), so a monotonicity assertion there
    # would be measuring grid sampling. See GRID_FINENESS in helpers.py.
    sf = H.sf_fft_for(scene_fine, fineness=1.6)
    residuals = []
    for n_sigma in (2.5, 3.0, 3.5, 4.5):
        sigma_cutoff(n_sigma)
        with torch.no_grad():
            F_fft = H.fft_sf(scene_fine, sf)
        residuals.append((n_sigma, H.rel_l2(F_fft, oracle_fine["iso_F"])))

    print()
    for n_sigma, rel in residuals:
        print(f"  n_sigma {n_sigma:4.1f}   relL2 vs DS {rel:.3e}")

    values = [r for _, r in residuals]
    assert values == sorted(values, reverse=True), (
        f"truncation error is not monotone in n_sigma: {residuals}"
    )
    assert values[0] > values[-1] * 2, (
        "tightening the cutoff barely changed the residual, so this scene is "
        "sampling-limited rather than truncation-limited and does not test the cutoff"
    )


def test_coarse_grid_is_measurably_worse(scene_coarse, scene_fine, oracle_fine):
    """Non-vacuity guard for ``RTOL_AMPLITUDE``.

    If an under-sampled grid also passed the gate, the gate would be measuring nothing.
    This pins the sampling-dominated regime separately so a regression that degraded
    grid sampling cannot hide behind the fine-grid tolerance.
    """
    # Bare Nyquist (oversampling 2 rather than production's 3).
    sf_coarse = H.sf_fft_for(scene_fine, fineness=2.0 / 3.0)
    with torch.no_grad():
        F_coarse = H.fft_sf(scene_fine, sf_coarse)
    rel = H.rel_l2(F_coarse, oracle_fine["iso_F"])
    print(
        f"\n  coarse grid {tuple(int(v) for v in sf_coarse.gridsize)}: "
        f"relL2 {rel:.3e}  (fine-grid gate is {RTOL_AMPLITUDE:.0e})"
    )
    assert rel > RTOL_AMPLITUDE, (
        "an under-sampled grid passes the amplitude gate, so the gate is not "
        "constraining grid sampling at all"
    )


# ---------------------------------------------------------------------------
# Every production kernel, on every device it ships on
# ---------------------------------------------------------------------------
# These call each splat kernel **directly** rather than through
# ``build_electron_density``. Under the default a failed accelerator
# kernel silently falls back to the portable splat, so a dispatch-driven test can pass
# while measuring a different kernel than the one it names; a direct call settles that by
# construction. The dispatch ladder is tested separately in ``test_dispatch.py``.
#
# The oracle stays CPU float64 whatever the candidate is -- a reference must not inherit
# the precision of the thing it judges.
#
# Measured when written (60-atom synthetic scene, worst case over the whole matrix):
#
#   amplitude rel L2   5.68e-03      gradient rel   9.51e-02
#   amplitude cos      0.999990      gradient cos   0.995544
#
# and, notably, **every device agrees to the printed digits**: Metal float32, CPU fused
# float32/float64 and the portable splat on either device all land on the same residual
# against the oracle. That is the truncation-contract standardization showing up as a
# measurement -- what is left is shared discretization error, not per-kernel divergence.
# Hence no accelerator-specific tolerances: the CPU constants already bound it.


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DEVICE_DTYPE_KERNELS)
def test_kernel_matches_ds_amplitudes(scene_fine, oracle_fine, device, dtype, kernel, kind):
    """Forward amplitudes from one production kernel against the DS oracle."""
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    with torch.no_grad():
        got = H.density_to_F(
            scene, H.splat_direct(scene, kernel, aniso=aniso)
        ).cpu().to(torch.complex128)
    ref = oracle_fine[f"{kind}_F"]

    rel, cos = H.rel_l2(got, ref), _complex_cos(got, ref)
    print(f"\n  {device.type}/{dtype}/{kernel}/{kind}: relL2 {rel:.3e}  cos {cos:.8f}")
    assert rel < RTOL_AMPLITUDE, f"{device.type}/{dtype}/{kernel}/{kind}: rel {rel:.3e}"
    assert cos > COS_MIN, f"{device.type}/{dtype}/{kernel}/{kind}: cos {cos:.6f}"


@pytest.mark.parametrize("device,dtype,kernel", DEVICE_DTYPE_KERNELS)
def test_kernel_absolute_scale(scene_fine, oracle_fine, device, dtype, kernel):
    """Absolute scale per kernel.

    Runs on every backend because the voxel-volume factor comes from ``ifft``, which is
    shared, but the *density* each kernel accumulates is not -- a kernel that normalized
    its Gaussian differently would show up only here. Correlation and cosine are both
    invariant to a global rescale.
    """
    scene = scene_fine.to(device=device, dtype=dtype)
    with torch.no_grad():
        got = H.density_to_F(scene, H.splat_direct(scene, kernel)).cpu().to(torch.complex128)
    scale = H.best_fit_scale(got, oracle_fine["iso_F"])
    print(f"\n  {device.type}/{dtype}/{kernel}: scale {scale:.9f}")
    assert abs(scale - 1.0) < SCALE_TOL, (
        f"{device.type}/{dtype}/{kernel}: absolute scale off by {abs(scale - 1.0):.3e}"
    )


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DEVICE_DTYPE_KERNELS)
def test_kernels_agree_with_each_other(scene_fine, device, dtype, kernel, kind):
    """Every kernel on a given ``(device, dtype)`` must agree with the portable splat.

    Complements the oracle comparison rather than duplicating it. The oracle gate is
    ``1e-2``, loose enough to hide a kernel that is subtly wrong in the same direction as
    the discretization error; this one is tight, because two kernels implementing the same
    truncation contract on the same inputs have nothing to disagree about beyond float
    accumulation order.
    """
    if kernel == "portable":
        pytest.skip("portable is the reference for this comparison")
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    with torch.no_grad():
        got = H.splat_direct(scene, kernel, aniso=aniso)
        ref = H.splat_direct(scene, "portable", aniso=aniso)
    tol = RTOL_BACKEND_F32 if dtype is torch.float32 else RTOL_BACKEND_F64
    rel = H.rel_l2(got.cpu().to(torch.float64), ref.cpu().to(torch.float64))
    print(f"\n  {device.type}/{dtype}/{kernel} vs portable ({kind}): relL2 {rel:.3e}")
    assert rel < tol, f"{device.type}/{dtype}/{kernel}/{kind}: vs portable rel {rel:.3e}"


# ---------------------------------------------------------------------------
# Direct-summation kernels vs the eager oracle
# ---------------------------------------------------------------------------
# Restores coverage lost when ``tests/integration/test_ds_triton_vs_eager.py`` was deleted,
# and extends it: that file only ran on CUDA, so ``_checkpointed_*`` -- what every CPU, MPS
# and float64 call actually executes -- was compared against eager on CPU only, and DS on
# MPS had no coverage at all.
#
# No grid and no truncation on either side, so the only source of disagreement is float
# arithmetic and these gate near precision. See ``RTOL_DS_F32`` / ``RTOL_DS_F64``.


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DS_DEVICE_DTYPE_KERNELS)
def test_ds_kernel_matches_eager_oracle(scene_fine, device, dtype, kernel, kind):
    """Forward ``F(hkl)`` from one DS kernel against the pure-torch eager oracle."""
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    with torch.no_grad():
        got = H.ds_direct(scene, kernel, aniso=aniso).cpu().to(torch.complex128)
        ref = H.ds_direct(scene_fine, "eager", aniso=aniso)

    tol = RTOL_DS_F32 if dtype is torch.float32 else RTOL_DS_F64
    rel, cos = H.rel_l2(got, ref), _complex_cos(got, ref)
    print(f"\n  {device.type}/{dtype}/{kernel}/{kind}: relL2 {rel:.3e}  cos {cos:.9f}")
    assert rel < tol, f"{device.type}/{dtype}/{kernel}/{kind}: rel {rel:.3e}"
    assert cos > COS_MIN_DS, f"{device.type}/{dtype}/{kernel}/{kind}: cos {cos:.7f}"
