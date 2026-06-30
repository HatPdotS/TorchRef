"""Guards the kernel-refactor backward-compat surface.

The density splat implementations were moved out of
``torchref.base.electron_density.main`` into one-file-per-backend modules under
``torchref.base.kernels``. These tests assert that every public name and every
historically-imported private name still resolves at its original import path,
so the move cannot silently break a downstream import.
"""

import importlib

import pytest


def test_kernels_public_api_resolves():
    kernels = importlib.import_module("torchref.base.kernels")
    expected = [
        "vectorized_add_to_map",
        "build_electron_density",
        "compute_metric_tensor",
        "precompute_fractional_coords",
        "warmup",
        "get_cache_dir",
        "clear_cache",
    ]
    for name in expected:
        assert hasattr(kernels, name), f"missing torchref.base.kernels.{name}"


def test_electron_density_public_api_resolves():
    ed = importlib.import_module("torchref.base.electron_density")
    expected = [
        "build_electron_density",
        "find_relevant_voxels",
        "vectorized_add_to_map",
        "vectorized_add_to_map_aniso",
        "scatter_add_nd",
        "scatter_add_nd_super_slow",
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
    """``main`` re-imports the moved splat impls so its namespace is unchanged.

    ``torchref.scaling.solvent`` and ``tests/unit/base/test_aniso_map_building``
    import these private symbols from ``...electron_density.main`` directly.
    """
    main = importlib.import_module("torchref.base.electron_density.main")
    moved = [
        "_get_box_radius",
        "_get_radius_offsets",
        "_do_structured_scatter",
        "_get_cpp_scatter",
        "_separable_density",
        "_get_compiled_separable_density",
        "_splat_chunk",
        "_CHUNK_SIZES",
        "_add_isotropic_cpu_separable",
        "_add_isotropic_cpu_separable_compiled",
        "_add_isotropic_cpu_fused",
        "_aniso_density_cube",
        "_add_anisotropic_cpu",
        "_add_isotropic_mps_single",
        "_add_isotropic_original",
        "_add_anisotropic_original",
        # dispatchers stay defined here
        "_add_isotropic",
        "_add_anisotropic",
    ]
    for name in moved:
        assert hasattr(main, name), f"missing torchref.base.electron_density.main.{name}"


def test_solvent_radius_offsets_import_path():
    """solvent.py does ``from ...electron_density.main import _get_radius_offsets``."""
    from torchref.base.electron_density.main import _get_radius_offsets  # noqa: F401


@pytest.mark.parametrize(
    "modname",
    [
        "torchref.base.electron_density.kernels",
        "torchref.base.electron_density.kernels.offsets",
        "torchref.base.electron_density.kernels.cpu.separable",
        "torchref.base.electron_density.kernels.cpu.fused",
        "torchref.base.electron_density.kernels.cpu.aniso",
        "torchref.base.electron_density.kernels.cpu.scatter",
        "torchref.base.electron_density.kernels.cpu.scatter_dispatch",
        "torchref.base.electron_density.kernels.cpu.jit_reference",
        "torchref.base.electron_density.kernels.cpu.eager_reference",
        "torchref.base.electron_density.kernels.cuda.fused",
        "torchref.base.electron_density.kernels.cuda.separable",
        "torchref.base.electron_density.kernels.mps.separable",
    ],
)
def test_new_kernel_modules_import(modname):
    importlib.import_module(modname)


def test_legacy_kernels_compat_shim():
    """``torchref.base.kernels`` stays a compat shim re-exporting the public API."""
    shim = importlib.import_module("torchref.base.kernels")
    for name in (
        "vectorized_add_to_map",
        "build_electron_density",
        "compute_metric_tensor",
        "precompute_fractional_coords",
        "warmup",
        "get_cache_dir",
        "clear_cache",
    ):
        assert hasattr(shim, name), f"compat shim missing {name}"
