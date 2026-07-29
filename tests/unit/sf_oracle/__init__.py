"""Structure-factor correctness, gated against an oracle at every derivative order.

Every accuracy gate in this package traces back to a reference that is *independent*
of the code under test. That is the whole point: before this package existed, the
density and SF coverage was

* kernel-A-vs-kernel-B parity at ``rel < 2e-2`` (``test_variable_radius_{gpu,mps}.py``),
  which cannot detect an error the two kernels share;
* splat-vs-brute-force-splat (``test_canonical_sphere_cpu.py``), which validates the
  truncation *contract* rather than the physics;
* finite differences of the map route against itself, which is self-consistency.

Nothing checked that the FFT route reproduces an analytic ``F(hkl)``, and nothing
checked its absolute scale at all -- every gate was ratio-, cosine- or parity-based, so
a factor-of-volume slip passed the entire suite.

The chain
---------
::

            gemmi  ->  direct summation  ->  the FFT/splat route     (amplitudes)
    finite differences  ->  direct summation  ->  the FFT/splat route (grad, HVP)

**Direct summation is the oracle.** It evaluates ``F(h) = sum_j occ_j f_j(s) DW_j
exp(2 pi i h.x_j)`` analytically -- no grid, no truncation radius -- so finite differences
are a valid reference for it (measured 1.19e-10). It is then an *independent* reference for
the map route, which is the part finite differences cannot supply: FD differentiates the
same discretized function the kernel does, so it confirms only that autograd is correct,
never that the kernel is. Measured, the map route's HVP agrees with itself to 5e-04 while
sitting 2.3e-02 from the analytic answer -- see
``test_second_order.py::test_finite_differences_cannot_detect_map_route_error``.

**gemmi anchors the forward values.** It is a separate codebase, is a pip dependency
(so it runs in CI, unlike conda-only cctbx), and exposes no derivatives -- hence FD
remains the derivative link.

What the gemmi comparison does and does not prove
-------------------------------------------------
torchref's ITC92 table was *generated from gemmi*
(``torchref/scripts/generate_scattering_table.py`` writes
``torchref/data/itc92_scattering_factors.pt``). Verified here: torchref stores gemmi's
four ITC92 Gaussians plus the constant ``c`` folded in as a fifth with ``B = 0``, and
``f(0)`` agrees to every printed digit.

So ``f(s)`` is a **shared** input and cancels out of a DS-vs-gemmi comparison. That test
does **not** validate the form factors. What it does validate is everything layered on
top of them, which is where the error-prone conventions live: the ``2 pi`` factors, the
phase sign, Debye-Waller in both the B and U conventions, the
``[U11,U22,U33,U12,U13,U23]`` ordering, occupancy weighting, and -- in the non-P1 case --
the symmetry algebra against a second implementation.

Two consequences, both acted on:

1. **The gate is tight, not loose.** With the coefficients shared there is no table
   discrepancy to absorb. See the measured numbers on ``RTOL_VS_GEMMI`` below.
2. **The table needs its own check**, since the SF comparison cannot provide one.
   ``test_forward.py::test_stored_table_matches_gemmi`` compares the stored ``.pt``
   against gemmi's live table element by element -- the one place gemmi *is* an
   independent reference for ``f(s)``, and the only thing that would catch table drift,
   a corrupted regeneration, or a gemmi release that changed coefficients.

Why the oracle is ``_eager_*`` and never ``ds_iso``/``SfDS``
------------------------------------------------------------
``ds_iso``, ``ds_aniso`` and ``SfDS`` all route through ``_CheckpointedSF``, whose
``backward`` calls ``torch.autograd.grad`` **without** ``create_graph=True`` on detached
copies (``torchref/base/direct_summation/dispatch.py:197``). They are exact at first
order -- measured agreement with the eager path is 2.3e-16 -- but a second derivative
through them raises ``element 0 of tensors does not require grad``. ``Engine.EAGER``
does not help; it only steers away from Triton and still lands on ``_CheckpointedSF``.

So the oracle is ``_eager_iso`` / ``_eager_aniso``, which are pure torch and therefore
compose to any order. ``test_second_order.py`` asserts both facts -- that the eager path
``gradgradcheck``s and that the public API raises -- so the constraint is documented
behaviour rather than a trap.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Outer link: gemmi -> DS. Forward only; gemmi exposes no derivatives.
# ---------------------------------------------------------------------------
# Measured on first calibration (float64, CPU), rel L2 on complex F:
#
#   iso,   P1,       3GR5 (1329 atoms, 240 refl)        3.28e-08
#   aniso, P1,       7L84 (1209 ANISOU atoms, 200 refl) 2.91e-08
#   iso,   P 65 2 2, 3GR5 via SfDS (12 ops, 200 refl)   3.22e-08
#
# The floor is the **stored table's float32 precision**. ``torchref/data/
# itc92_scattering_factors.pt`` holds float32 ``(104, 5)`` tensors, while gemmi's Python
# ``it92`` returns doubles; the worst relative coefficient difference over Z=1..103 is
# 5.9e-08, which is half of float32 eps. Asking ``get_scattering_params_by_z`` for
# float64 widens those float32 values rather than recovering gemmi's doubles, so even the
# "float64" oracle carries ~6e-8 relative error in ``f(s)``.
#
# That is the correct trade for a float32 production path, not a defect -- but it means
# tightening below ~1e-7 would gate on table storage precision rather than on any
# structure-factor logic. ``test_stored_table_matches_gemmi`` pins the rounding exactly
# (bit-exact in float32) so this floor cannot drift unnoticed.
#
# Non-vacuity, measured with the same harness: permuting the ADP off-diagonals to
# swap U12<->U13 moves rel L2 to 1.35e-02, and reversing them to 6.08e-03 -- five
# orders of magnitude clear of the gate. The comparison genuinely constrains the
# convention.
RTOL_VS_GEMMI = 1e-6  # rel L2 on complex F;   30x margin on the worst measurement
MAXREL_VS_GEMMI = 1e-4  # max per-reflection;    48x margin (worst was 2.06e-06)
SCALE_TOL_GEMMI = 1e-6  # |least-squares scale - 1|; worst measured 1.3e-08

# Stored table vs gemmi's live table. Both hold the same float32-rounded coefficients,
# so this is expected to be exact; a nonzero result is a finding, not a tolerance to
# widen.
ATOL_TABLE_VS_GEMMI = 0.0

# ---------------------------------------------------------------------------
# Oracle soundness: FD -> DS. DS is analytic and smooth, so gate hard.
# ---------------------------------------------------------------------------
EPS_GRADCHECK = 1e-6
ATOL_GRADCHECK = 1e-5
RTOL_GRADCHECK = 1e-3
RTOL_DS_HVP_VS_FD = 1e-5  # measured 1.19e-10 on a 6-atom scene

# ---------------------------------------------------------------------------
# The FFT/splat route vs the DS oracle.
# ---------------------------------------------------------------------------
# These gates bound a real discretization error rather than eliminating it: the map route
# carries both a truncation and a grid-sampling residual, and gradients are markedly more
# sensitive to both than amplitudes are, because each xyz derivative brings a factor
# ~``2*pi*h`` and so re-weights the comparison toward high-resolution reflections -- the
# ones the grid resolves worst.
#
# **Production runs in float32.** So float32 is the load-bearing case throughout this
# package, not a variant bolted on beside float64 -- every parametrization lists it
# first, and these gates are calibrated on it. float64 is kept because it separates a
# genuine truncation residual from float32 accumulation noise: if a float32 result misses
# a gate and float64 passes it comfortably, the cause is precision, not the kernel. The
# oracle itself always stays in float64 regardless of the candidate's dtype -- a
# reference should not inherit the precision of the thing it is judging.
# Measured at the **production** grid (``max_res = d_min``, spacing ``d_min/3``) with the
# default 3.0 sigma cutoff, differentiating ``ls_target`` against pseudo-observations
# offset from the oracle by 10% relative noise. Identical to 3 significant figures across
# float32/float64 and AUTO/EAGER, so these are properties of the discretization, not of
# precision or of a kernel:
#
#            amplitude   g_xyz      g_occ      g_adp/g_U   HVP        HVP cos
#   iso      5.89e-03    8.51e-02   2.68e-02   4.45e-02    2.26e-02   0.99980
#   aniso    5.20e-03    8.17e-02   1.20e-01   2.24e-01    1.39e-02   0.99991
#
# The anisotropic ADP gradient is the outlier at 2.2e-01 -- it converges to 5.6e-03 by
# fineness 1.6, so it is grid-sampling error rather than a defect in the U derivative, but
# it does not fit inside 1e-1 at production sampling. Rather than raise the shared gate
# (which would stop it constraining anything) or silently test on a finer grid than
# production, aniso ADP gets its own documented constant.
# **Authoritative gates are calibrated on a real structure, not a synthetic scene.**
# Measured on 7L84 forced to P1 -- 1209 atoms all carrying ANISOU, 799 reflections to
# 1.5 A -- at production oversampling 3, float32 candidate against the float64 oracle:
#
#   amplitude rel L2      2.28e-03
#   xyz gradient          1.35e-02   cos 0.999909
#   U gradient            2.62e-02   cos 0.999657   gradnorm ratio 0.9997
#
# Synthetic small-cell scenes are markedly *pessimistic* here and must not be used to set
# absolute gates. Aliasing contamination in the derivative cancels across atoms roughly as
# 1/sqrt(N), because with few atoms each reflection's F is dominated by a handful of
# coherent contributions while with many the inter-atom phases are quasi-random. Measured
# deviatoric-U gradnorm ratio at production oversampling, all else equal:
#
#   10 atoms    1.43 - 1.70   (independent of how anisotropic U actually is)
#   60 atoms    0.972
#   300 atoms   0.997
#   7L84, 1209  0.9995
#
# An earlier revision of this package calibrated the aniso gates on a 10-atom scene and
# recorded a "systematic 54% deviatoric ADP gradient bias" as a production finding. It was
# an atom-count artifact and does not exist on real structures. The synthetic scenes are
# retained for cheap parametrized coverage -- dtype, engine, cell angle -- where the
# comparison is *between* backends and scene realism is irrelevant.
RTOL_AMPLITUDE = 1e-2  # 7L84 2.28e-03; synthetic worst 5.50e-03
RTOL_GRADIENT = 1e-1  # 7L84 worst leaf 2.62e-02
RTOL_HVP = 1e-1  # synthetic worst 2.22e-02
COS_MIN = 0.999  # forward complex F
COS_MIN_GRADIENT = 0.999  # 7L84 worst 0.999657
SCALE_TOL = 1e-3  # |best-fit scale - 1|; worst measured 3.7e-05

# Synthetic-scene coverage sweeps only, never as an accuracy statement about production.
# Small cells with few atoms overstate the discretization residual and which leaf is worst
# moves around with atom count (10 atoms: aniso U; 60 atoms: iso adp at 1.36e-01).
RTOL_GRADIENT_SYNTHETIC = 2e-1
COS_MIN_GRADIENT_SYNTHETIC = 0.99

# ---------------------------------------------------------------------------
# Cross-backend parity. Same truncation contract on every path, so near-exact.
# ---------------------------------------------------------------------------
RTOL_BACKEND_F32 = 2e-4
RTOL_BACKEND_F64 = 1e-12

# Gradients need their own float32 constant. The fused C++ kernel's hand-written backward
# and the portable splat's autograd accumulate in different orders, and float32 does not
# forgive that the way the forward pass does. Measured AUTO-vs-EAGER on ``scene_fine``
# (60 atoms), float32:
#
#            xyz        occ        adp/U
#   iso      2.07e-04   7.63e-04   1.43e-03
#   aniso    1.95e-04   6.79e-04   6.29e-04
#
# so ~7x above the 2e-04 that suffices for forward values, worst case on the iso ADP
# gradient. float64 stays exact to 1e-12, which is the evidence that this is accumulation
# order rather than a genuine difference in what the two kernels compute.
#
# Unlike the accuracy gates, this one is *comparative* -- two backends on identical inputs
# -- so it is legitimately calibrated on a synthetic scene: the atom-count effect that
# makes small scenes unrepresentative for DS comparisons cancels when both sides share it.
RTOL_BACKEND_GRAD_F32 = 5e-3
RTOL_BACKEND_GRAD_F64 = 1e-10

__all__ = [
    "RTOL_VS_GEMMI",
    "MAXREL_VS_GEMMI",
    "SCALE_TOL_GEMMI",
    "ATOL_TABLE_VS_GEMMI",
    "EPS_GRADCHECK",
    "ATOL_GRADCHECK",
    "RTOL_GRADCHECK",
    "RTOL_DS_HVP_VS_FD",
    "RTOL_AMPLITUDE",
    "RTOL_GRADIENT",
    "RTOL_GRADIENT_SYNTHETIC",
    "RTOL_HVP",
    "COS_MIN",
    "COS_MIN_GRADIENT",
    "COS_MIN_GRADIENT_SYNTHETIC",
    "SCALE_TOL",
    "RTOL_BACKEND_F32",
    "RTOL_BACKEND_F64",
    "RTOL_BACKEND_GRAD_F32",
    "RTOL_BACKEND_GRAD_F64",
]
