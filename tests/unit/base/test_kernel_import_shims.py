"""Guards the kernel-refactor backward-compat surface.

The density splat implementations were moved out of
``torchref.base.electron_density.main`` into one-file-per-backend modules under
``torchref.base.electron_density.kernels``. These tests assert that every public name
and every historically-imported private name still resolves at its original import
path, so the move cannot silently break a downstream import.
"""

import importlib

import pytest

from torchref.utils.backends import triton_available

_needs_triton = pytest.mark.skipif(
    not triton_available(), reason="CUDA/Triton backend requires the triton package"
)


def test_electron_density_public_api_resolves():
    ed = importlib.import_module("torchref.base.electron_density")
    expected = [
        "build_electron_density",
        "find_relevant_voxels",
        "vectorized_add_to_map",
        "vectorized_add_to_map_aniso",
        "scatter_add_nd",
        "excise_angstrom_radius_around_coord",
    ]
    for name in expected:
        assert hasattr(ed, name), f"missing torchref.base.electron_density.{name}"


def test_math_torch_legacy_reexports_resolve():
    mt = importlib.import_module("torchref.base.math_torch")
    for name in (
        "vectorized_add_to_map",
        "vectorized_add_to_map_aniso",
        "find_relevant_voxels",
    ):
        assert hasattr(mt, name), f"missing torchref.base.math_torch.{name}"


def test_main_namespace_preserves_moved_symbols():
    """``main`` keeps its two dispatchers, which tests reach for by name."""
    main = importlib.import_module("torchref.base.electron_density.main")
    moved = [
        "_add_isotropic",
        "_add_anisotropic",
    ]
    for name in moved:
        assert hasattr(main, name), f"missing torchref.base.electron_density.main.{name}"


@pytest.mark.parametrize(
    "modname",
    [
        "torchref.base.electron_density.kernels",
        "torchref.base.electron_density.kernels.cpu.jit_reference",
        "torchref.base.electron_density.kernels.cpu.variable_radius",
        # CUDA/Triton backend: importing pulls in `triton`, absent on non-CUDA
        # hosts (e.g. macOS), so gate these on triton availability.
        pytest.param(
            "torchref.base.electron_density.kernels.cuda.fused", marks=_needs_triton
        ),
        pytest.param(
            "torchref.base.electron_density.kernels.cuda.variable_radius",
            marks=_needs_triton,
        ),
        # MPS Metal kernels: importing must not trigger shader compilation, so
        # these resolve cleanly on every platform (compile is deferred to first use).
        "torchref.base.electron_density.kernels.mps",
        "torchref.base.electron_density.kernels.mps.compile",
        "torchref.base.electron_density.kernels.mps.variable_radius",
        "torchref.base.electron_density.kernels.mps._shaders",
    ],
)
def test_new_kernel_modules_import(modname):
    importlib.import_module(modname)
