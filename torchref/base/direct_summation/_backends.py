"""The direct-summation dispatch policy, as one table.

Companion to ``electron_density/_backends.py``; see :mod:`torchref.utils.backends` for what
each field means.

There is no Metal direct-summation kernel -- the Metal shader is a *density splat* only --
so DS on MPS is ``_checkpointed_*`` running on-device, which the table states by having no
MPS row.

The float32 requirement is policy, not capability: ``triton_ds._cols_f32`` would happily
downcast a float64 input, so the gate declines instead rather than serve a float64
configuration silently at float32. ``hkl`` is deliberately *not* probed (Miller indices are
integer-valued, so their f64->f32 round-trip is bit-exact), and the dtype rule exempts
integer tensors -- without that the ``int32`` production dtype would read as a capability
failure and disable the kernel outright.
"""

from __future__ import annotations

import torch

from torchref.utils.backends import Backend, BackendTable
from torchref.utils.backends import triton_available

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
# it forms only a ``(BLOCK_H, N)`` tile in registers. The budget is dropped in a named
# adapter (named, not a lambda, so it shows up in a traceback and can be patched).
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
