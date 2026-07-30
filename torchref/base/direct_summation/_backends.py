"""The direct-summation dispatch policy, as one table.

Companion to ``electron_density/_backends.py``; see :mod:`torchref.utils.backends` for what
each field means.

Two things about this table are worth reading before changing it.

**There is no Metal direct-summation kernel.** ``Engine.METAL`` selects the Metal *density
splat* and nothing else, so at this site it has to mean "run eager" -- which is why
``checkpointed`` lists it. That was previously an early ``return False`` buried in a
predicate; here it is a declared fact, and the table refuses to be constructed if some
engine ends up handled by nothing. DS on MPS is therefore ``_checkpointed_*`` running
on-device, a real production path.

**The float32 requirement here is policy, not capability**, and the policy is mild. The
Triton kernel casts every input to float32 itself (``triton_ds._cols_f32``), so it would
happily consume float64; the gate declines instead, so that a float64 configuration is not
silently served at float32 precision. That is a consistency rule, not a rescue: measured on
a 40-atom / 60-reflection scene, downcasting the real-valued inputs moves ``F`` by 7.2e-06
relative to ``mean|F|``, three orders of magnitude inside the 1e-2 amplitude tolerance this
package works to.

It matters even less than that suggests, because the interesting case was never reachable.
Under a float64 configuration ``xyz_frac`` is float64, and the previous gate probed
``xyz_frac`` -- so it already declined. The only thing widening the probe set catches is
hand-mixed dtypes (float32 coordinates beside a float64 ``s`` or ``adp``), which no caller in
the repo produces. Treat this as making the gate check what its kernel actually consumes,
consistent with the density table, rather than as a fix for anything observed.

``hkl`` is deliberately **not** probed, and probing it would be wrong. Miller indices are
integers -- ``int32`` from the MTZ reader, and integer-valued even in a float dtype, since
symmetry maps ``h -> h.R`` with integer ``R``. The f64->f32 round-trip is bit-exact for any
\\|h\\| < 2**24 (real structures reach a few hundred), and measurably so: downcasting ``hkl``
alone changes ``F`` by exactly 0.0. Probing it would only manufacture a false negative.

The dtype rule exempts integer tensors for the same reason -- without that, the ``int32``
production dtype would read as a capability failure and disable the kernel outright.
"""

from __future__ import annotations

import torch

from torchref.utils.backends import Backend, BackendTable
from torchref.utils.triton_dispatch import Engine, triton_available

_THIS = "torchref.base.direct_summation._backends"


def why_unavailable():
    """``None`` if the direct-summation Triton kernels can run, else why they cannot.

    Two separable questions, and both have to be asked. ``triton_available()`` answers "is
    the ``triton`` package importable"; this additionally answers "does ``triton_ds``
    itself import", which can fail on its own -- the module evaluates
    ``tl.constexpr(...)`` at import time and that is version-sensitive. The density
    backend has the same split.
    """
    if not triton_available():
        return "triton is not importable"
    try:
        from torchref.base.direct_summation import triton_ds  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure is a reason, not a crash
        return (
            "the direct-summation Triton kernels could not be imported "
            f"({type(exc).__name__}: {exc})"
        )
    return None


# ---------------------------------------------------------------------------
# Signature adapters
# ---------------------------------------------------------------------------
# The two backends do not share a signature: the checkpointed path takes a trailing
# ``max_memory_gb`` reflection-chunk budget and the Triton kernel has nothing to bound --
# it forms only a ``(BLOCK_H, N)`` tile in registers. Rather than add a parameter the
# kernel ignores (a lie in a public signature, and the exact drift just removed from the
# Metal wrappers), the budget is dropped in a named adapter. Named, not a lambda, so it
# appears in a traceback and can be patched.
def _ds_iso_triton(hkl, s, xyz_frac, occ, adp, A, B, max_memory_gb):
    """Isotropic Triton DS; ``max_memory_gb`` is not applicable and is dropped here."""
    from torchref.base.direct_summation.triton_ds import ds_iso_triton

    return ds_iso_triton(hkl, s, xyz_frac, occ, adp, A, B)


def _ds_aniso_triton(hkl, s_vec, xyz_frac, occ, U, A, B, max_memory_gb):
    """Anisotropic Triton DS; ``max_memory_gb`` is not applicable and is dropped here."""
    from torchref.base.direct_summation.triton_ds import ds_aniso_triton

    return ds_aniso_triton(hkl, s_vec, xyz_frac, occ, U, A, B)


DS_BACKENDS = BackendTable(
    name="direct-summation structure factors",
    backends=(
        Backend(
            name="ds_triton",
            kernel=(_THIS, "_ds_iso_triton", "_ds_aniso_triton"),
            engines=frozenset({Engine.AUTO, Engine.TRITON}),
            device="cuda",
            dtypes=(torch.float32,),
            # Every argument except ``hkl`` (position 0), whose dtype provably costs
            # nothing -- see the module docstring.
            probes=(1, 2, 3, 4, 5, 6),
            probe=(_THIS, "why_unavailable"),
            expect_available="cuda",
            on_failure="degrade",
            second_order=False,
        ),
        Backend(
            name="checkpointed",
            kernel=(
                "torchref.base.direct_summation.dispatch",
                "_checkpointed_iso",
                "_checkpointed_aniso",
            ),
            # METAL is here because there is no Metal DS kernel; see the module docstring.
            # TRITON is absent, which is what makes that engine strict.
            engines=frozenset({Engine.AUTO, Engine.EAGER, Engine.METAL}),
            expect_available="always",
            on_failure="raise",
            # First-order only: the backward replays each chunk under ``enable_grad`` but
            # without ``create_graph``, so a second derivative raises rather than returning
            # something wrong. ``_eager_*`` is the double-differentiable reference and is
            # deliberately not in this table -- it is not a production dispatch target.
            second_order=False,
        ),
    ),
)
