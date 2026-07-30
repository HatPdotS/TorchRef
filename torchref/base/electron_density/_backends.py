"""The electron-density splat dispatch policy, as one table.

Every criterion for choosing a density kernel is a field in a row below: which device,
which dtypes, how availability is probed, whether a runtime failure
may degrade, and whether the kernel composes to second order. Reading this file answers
"which kernel runs for MPS + float64?" without tracing an if/elif ladder
through three modules.

All four wrappers share one signature --
``(density_map, xyz, adp_or_u, occ, A, B, inv_frac, frac, radius_per_atom)`` -- which is
what lets the dispatch site be a single call rather than a per-kernel adapter.

Ordering within the table is not load-bearing. The three non-base backends are pairwise
device-disjoint (CUDA / CPU / MPS), so at most one can ever match; the base case matches
everything and is what selection falls through to.

See :mod:`torchref.utils.backends` for what each field means and why it exists.
"""

from __future__ import annotations

import torch

from torchref.utils.backends import Backend, BackendTable

_CUDA = "torchref.base.electron_density.kernels.cuda.variable_radius"
_MPS = "torchref.base.electron_density.kernels.mps.variable_radius"
_SPHERE = "torchref.base.electron_density.kernels.cpu.sphere_splat"
_PORTABLE = "torchref.base.electron_density.kernels.cpu.variable_radius"

#: Argument positions carrying the device/dtype contract:
#: ``density_map, xyz, adp_or_u, occ, A, B``. The trailing three -- the two cell matrices
#: and the per-atom radius -- are excluded because no gate has ever probed them, and
#: widening the contract to cover them would make two currently-working backends stricter
#: for no demonstrated bug. They *are* read as raw pointers by the C++ and CUDA kernels, so
#: that remains an open question rather than a settled one.
_ATOM_ARGS = (0, 1, 2, 3, 4, 5)

DENSITY_BACKENDS = BackendTable(
    name="electron-density splat",
    backends=(
        Backend(
            name="cuda_triton",
            kernel=(_CUDA, "add_isotropic_cuda_var", "add_anisotropic_cuda_var"),
            device="cuda",
            dtypes=(torch.float32,),
            probes=_ATOM_ARGS,
            probe=(_CUDA, "why_unavailable"),
            expect_available="cuda",
            # A failed launch on an available GPU is a capability miss worth degrading
            # from; the kernel clones the density map before accumulating, so the
            # fallback cannot double-count.
            on_failure="degrade",
            second_order=False,
        ),
        Backend(
            name="mps_metal",
            kernel=(_MPS, "add_isotropic_mps_var", "add_anisotropic_mps_var"),
            device="mps",
            dtypes=(torch.float32,),
            probes=_ATOM_ARGS,
            probe=("torchref.base.electron_density.kernels.mps.compile",
                   "why_unavailable"),
            expect_available="mps",
            on_failure="degrade",
            second_order=False,
        ),
        Backend(
            name="cpu_sphere",
            kernel=(_SPHERE, "add_isotropic_cpu_sphere_var",
                    "add_anisotropic_cpu_sphere_var"),
            device="cpu",
            dtypes=(torch.float32, torch.float64),
            # Uniformity, not membership: the kernel picks one ``scalar_t`` from the output
            # map and then reads every other tensor through a raw pointer of that type, so a
            # float64 map beside float32 atoms would be a 2x out-of-bounds read.
            require_uniform_dtype=True,
            probes=_ATOM_ARGS,
            probe=(_SPHERE, "why_unavailable"),
            # A pure C++ build with no hardware requirement, so it must work everywhere. The
            # expectation is what makes a broken build a CI failure instead of a skip.
            expect_available="always",
            # Deliberately not "degrade". The extension having failed to *build* already
            # yields a reason from the probe, so this governs only a runtime throw from a
            # kernel that compiled -- a genuine bug. Falling back would turn wrong results
            # into a ~100x slowdown whose numbers still look plausible, because the portable
            # splat implements the same truncation contract.
            on_failure="raise",
            second_order=True,
        ),
        Backend(
            name="portable",
            kernel=(_PORTABLE, "add_isotropic_plain_var", "add_anisotropic_plain_var"),
            # The base case: no device or dtype restriction, so it always matches. That is
            # what makes ``select`` unable to fail.
            expect_available="always",
            on_failure="raise",
            second_order=True,
        ),
    ),
)
