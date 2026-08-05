"""Fixtures for the structure-factor oracle package.

Package-scoped fixtures capture the real redundancy -- three test modules otherwise
recomputing the same oracle -- while being recomputed on every run, so there is no
invalidation logic and no staleness risk at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchref
from torchref.config import device as device_cfg, dtypes

from tests.conftest import _accelerator

from . import helpers as H


# ---------------------------------------------------------------------------
# Device axis
# ---------------------------------------------------------------------------
# Built at **import time**, copying ``tests/conftest.py:295``. That is load-bearing: the
# backend mark has to be attached during *collection*, because
# ``pytest_collection_modifyitems`` is what gates on it and cannot see a mark added later
# from inside a fixture. On a CPU-only host the accelerator param does not exist at all,
# so there is no skip noise -- and on this host the ``cuda`` leg likewise never appears.
#
# The accelerator carries its *specific* backend mark (``cuda`` or ``mps``), not the
# generic ``gpu`` one, so a CUDA-less host skips the cuda leg with an accurate reason.
_DEVICES = [pytest.param(torch.device("cpu"), id="cpu")]
_ACCELERATOR = _accelerator()
if _ACCELERATOR is not None:
    _DEVICES.append(
        pytest.param(
            _ACCELERATOR,
            id=_ACCELERATOR.type,
            marks=getattr(pytest.mark, _ACCELERATOR.type),
        )
    )

#: float32 first: the production dtype. MPS cannot hold float64 at all, which
#: ``helpers.device_supports_dtype`` filters -- so ``(mps, float64)`` yields no kernels
#: and therefore no test, rather than a test that skips or silently passes.
_DTYPES = [torch.float32, torch.float64]


def device_dtype_kernels():
    """Every ``(device, dtype, kernel_name)`` that names a real production path.

    Enumerated from the kernel registry rather than written out, so a kernel added to
    ``helpers._KERNEL_SPECS`` is covered automatically and an unsupported combination
    produces no test instead of a vacuous one.
    """
    out = []
    for dev_param in _DEVICES:
        device = dev_param.values[0]
        for dtype in _DTYPES:
            for name in H.kernels_for(device, dtype):
                out.append(
                    pytest.param(
                        device,
                        dtype,
                        name,
                        id=f"{device.type}-{str(dtype).replace('torch.float', 'f')}-{name}",
                        marks=dev_param.marks,
                    )
                )
    return out


def ds_device_dtype_kernels():
    """Every ``(device, dtype, ds_kernel_name)`` naming a real direct-summation path.

    Separate from the splat list because the two families have different device/dtype
    envelopes -- notably there is no Metal DS kernel, so the MPS leg here is
    ``_checkpointed_*`` running on-device rather than a native shader.
    """
    out = []
    for dev_param in _DEVICES:
        device = dev_param.values[0]
        for dtype in _DTYPES:
            for name in H.ds_kernels_for(device, dtype):
                out.append(
                    pytest.param(
                        device, dtype, name,
                        id=f"{device.type}-{str(dtype).replace('torch.float', 'f')}-{name}",
                        marks=dev_param.marks,
                    )
                )
    return out


DEVICE_DTYPE_KERNELS = device_dtype_kernels()
DS_DEVICE_DTYPE_KERNELS = ds_device_dtype_kernels()


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
@pytest.fixture(scope="package", autouse=True)
def _float64_cpu():
    """float64/complex128 on CPU for this package; restore afterwards.

    Required, not cosmetic: ``iso_structure_factor_torched`` casts ``hkl`` to the
    *global* ``dtypes.float`` (``torchref/base/direct_summation/isotropic.py:121``), so
    under the default float32 config a float64 leaf produces a dtype-mismatched matmul.
    That is why the pre-existing tests wrapped every eager-SF call in a ``double_cpu``
    fixture.

    ``sigma_cutoff_ed`` is restored here too -- the three copies of ``double_cpu`` this
    replaces did not, so a test that changed the cutoff leaked it into everything that
    ran after it.
    """
    f0, c0, d0 = dtypes.float, dtypes.complex, device_cfg.current
    s0 = torchref.sigma_cutoff_ed.value
    dtypes.float = torch.float64
    dtypes.complex = torch.complex128
    device_cfg.current = torch.device("cpu")
    try:
        yield
    finally:
        dtypes.float = f0
        dtypes.complex = c0
        device_cfg.current = d0
        torchref.sigma_cutoff_ed.value = s0


@pytest.fixture
def sigma_cutoff():
    """``sigma_cutoff(value)`` sets ``torchref.sigma_cutoff_ed`` and restores after."""
    original = torchref.sigma_cutoff_ed.value

    def _set(value: float) -> None:
        torchref.sigma_cutoff_ed.value = value

    try:
        yield _set
    finally:
        torchref.sigma_cutoff_ed.value = original


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
@pytest.fixture(scope="package")
def scene_small() -> H.Scene:
    """5 atoms, few reflections -- sized for ``gradcheck``'s O(n_params) cost."""
    return H.synthetic_scene(n_atoms=5, a=14.0, d_min=3.0, max_refl=24, seed=3)


@pytest.fixture(scope="package")
def scene_fine() -> H.Scene:
    """Synthetic P1 scene for parametrized coverage sweeps.

    60 atoms rather than 10. Aliasing contamination in the *derivatives* cancels across
    atoms roughly as 1/sqrt(N), so a 10-atom scene overstates it badly -- the deviatoric-U
    gradnorm ratio is 1.43-1.70 at 10 atoms against 0.997 at 300 and 0.9995 on 7L84. See
    the note on the gate constants in ``__init__.py``.

    Even at 60 atoms this remains pessimistic relative to a real structure, so it backs the
    *comparative* tests -- backend parity, dtype coverage, cell angle -- while the absolute
    accuracy gates are calibrated on ``gemmi_aniso_p1``.
    """
    return H.synthetic_scene(n_atoms=60, a=24.0, d_min=1.6, max_refl=600, seed=0)


@pytest.fixture(scope="package")
def scene_monoclinic() -> H.Scene:
    """beta = 115 deg. The non-orthogonal metric is where a node-centred or diagonal
    truncation diverges most from the true Cartesian sphere."""
    return H.synthetic_scene(
        n_atoms=60, a=24.0, beta=115.0, d_min=1.6, max_refl=600, seed=1
    )


@pytest.fixture(scope="package")
def scene_coarse() -> H.Scene:
    """Deliberately under-sampled: pins the sampling-dominated regime so the fine-grid
    gate cannot pass by accident."""
    return H.synthetic_scene(n_atoms=60, a=24.0, d_min=3.5, max_refl=200, seed=0)


# ---------------------------------------------------------------------------
# gemmi scenes, from real deposited structures
# ---------------------------------------------------------------------------
_PDB_DIR = Path(__file__).resolve().parents[2] / "files" / "pdb"


@pytest.fixture(scope="package")
def gemmi_iso_p1():
    """3GR5 forced to P1. 1329 atoms after hydrogen removal."""
    pytest.importorskip("gemmi")
    return H.gemmi_scene(_PDB_DIR / "3GR5.pdb", p1=True, d_min=3.0, max_refl=200)


@pytest.fixture(scope="package")
def gemmi_aniso_p1():
    """7L84 forced to P1 -- every one of its 1209 atoms carries ANISOU."""
    pytest.importorskip("gemmi")
    return H.gemmi_scene(_PDB_DIR / "7L84.pdb", p1=True, d_min=3.0, max_refl=200)


@pytest.fixture(scope="package")
def gemmi_aniso_grad():
    """7L84 P1 at 1.5 A with enough reflections to calibrate derivative gates.

    Separate from ``gemmi_aniso_p1`` (which is sized for the cheap forward gemmi
    comparison) because the absolute gradient and HVP gates must be set on a scene that is
    representative, and representativeness here is driven by atom count and resolution.
    """
    pytest.importorskip("gemmi")
    return H.gemmi_scene(_PDB_DIR / "7L84.pdb", p1=True, d_min=1.5, max_refl=500)


@pytest.fixture(scope="package")
def oracle_aniso_grad(gemmi_aniso_grad):
    """DS oracle for the real-structure scene: F, obs, gradients and an HVP."""
    scene, _ = gemmi_aniso_grad
    return _oracle_bundle(scene)


@pytest.fixture(scope="package")
def gemmi_iso_symmetry():
    """3GR5 in its deposited P 65 2 2.

    Hexagonal on purpose. ``h' = h.R`` and ``h' = R.h`` coincide whenever the rotation
    matrix is symmetric, so orthorhombic and tetragonal groups cannot distinguish them;
    trigonal/hexagonal groups can. This is the same reasoning that drives the group
    choice in ``tests/unit/symmetry/test_hkl_symmetry_gemmi.py``.
    """
    pytest.importorskip("gemmi")
    return H.gemmi_scene(_PDB_DIR / "3GR5.pdb", p1=False, d_min=3.0, max_refl=200)


# ---------------------------------------------------------------------------
# Memoized oracle results
# ---------------------------------------------------------------------------
@pytest.fixture(scope="package")
def oracle_fine(scene_fine):
    """Forward ``F``, first-order grads and an HVP from the DS oracle, computed once."""
    return _oracle_bundle(scene_fine)


@pytest.fixture(scope="package")
def oracle_monoclinic(scene_monoclinic):
    return _oracle_bundle(scene_monoclinic)


def _oracle_bundle(scene: H.Scene) -> dict:
    """Forward ``F``, pseudo-observations, gradients and an HVP, all from the oracle.

    ``obs`` is derived here and handed to the tests, so the candidate and the oracle are
    scored against *identical* pseudo-observations. Regenerating it per test would make
    the two sides differentiate slightly different targets and the comparison would
    measure that instead.
    """
    out: dict = {}
    v = _direction(scene)
    for aniso in (False, True):
        key = "aniso" if aniso else "iso"
        fn = H.ds_aniso_oracle if aniso else H.ds_iso_oracle

        with torch.no_grad():
            F = fn(scene)
        obs = H.synthetic_obs(F)
        out[f"{key}_F"] = F
        out[f"{key}_obs"] = obs
        out[f"{key}_v"] = v

        xyz, occ, third = scene.leaves(aniso=aniso)
        grads = torch.autograd.grad(
            H.ls_target(fn(scene, xyz, occ, third), obs), (xyz, occ, third)
        )
        out[f"{key}_grads"] = tuple(g.detach() for g in grads)

        xyz, occ, third = scene.leaves(aniso=aniso)
        (g1,) = torch.autograd.grad(
            H.ls_target(fn(scene, xyz, occ, third), obs), xyz, create_graph=True
        )
        (hv,) = torch.autograd.grad((g1 * v).sum(), xyz)
        out[f"{key}_hvp"] = hv.detach()
    return out


def _direction(scene: H.Scene) -> torch.Tensor:
    """Fixed unit direction for HVPs. Seeded, so the oracle and the candidate agree."""
    g = torch.Generator().manual_seed(17)
    v = torch.randn(scene.xyz.shape, generator=g, dtype=scene.xyz.dtype)
    return v / v.norm()
