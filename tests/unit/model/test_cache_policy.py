"""Caches are never copied or serialized; they are recomputed lazily on access.

Pins the policy for every cache the model stack holds: the derived Model
buffers (``_Z``, ``_A``/``_B``, ``vdw_radii``, ``_heavy_atom_mask``), the SF
index cache, the FFT grid tensors, ``Symmetry``/``Cell`` derived-quantity
caches under pickle and deepcopy, ``CachedForwardMixin`` forward caches, and
the ``AtomGraph`` bond adjacency.
"""

import copy
import pickle

import pytest
import torch

CACHE_BUFFER_NAMES = ("_Z", "_A", "_B", "vdw_radii", "_heavy_atom_mask")


@pytest.mark.unit
def test_state_dict_carries_no_cache_buffers(pdb_dir):
    from torchref.model import ModelFT

    model = ModelFT(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    # Populate every cache, then serialize.
    model.Z
    model.get_scattering_params_iso()
    model.get_vdw_radii()
    model.exclude_H_from_sf = True
    model.setup_grid()
    sd = model.state_dict()

    for name in CACHE_BUFFER_NAMES:
        assert name not in sd, f"cache buffer {name} leaked into state_dict"
    grid_keys = [k for k in sd if k.startswith("_fft.")]
    assert grid_keys == [], f"FFT grid caches leaked into state_dict: {grid_keys}"


@pytest.mark.unit
def test_restore_rebuilds_caches_lazily(pdb_dir):
    from torchref.model import ModelFT

    cpu = torch.device("cpu")
    model = ModelFT(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    model.to(cpu)
    model.get_scattering_params_iso()
    model.get_vdw_radii()

    restored = ModelFT.create_from_state_dict(
        model.state_dict(), device=cpu, verbose=0
    )
    A_new, B_new = restored.get_scattering_params_iso()
    A_old, B_old = model.get_scattering_params_iso()
    assert torch.allclose(A_new, A_old) and torch.allclose(B_new, B_old)
    assert torch.allclose(restored.get_vdw_radii(), model.get_vdw_radii())
    assert torch.allclose(restored.Z, model.Z)


@pytest.mark.unit
def test_explicit_gridsize_is_a_setting_and_round_trips(pdb_dir):
    """The derived grid is a cache, but a user-requested gridsize is config."""
    from torchref.model import ModelFT

    cpu = torch.device("cpu")
    model = ModelFT(verbose=0, gridsize=(30, 30, 30))
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    model.to(cpu)
    model.setup_grid()
    assert tuple(model.gridsize.tolist()) == (30, 30, 30)

    restored = ModelFT.create_from_state_dict(
        model.state_dict(), device=cpu, verbose=0
    )
    assert restored._explicit_gridsize == (30, 30, 30)
    restored.setup_grid()
    assert tuple(restored.gridsize.tolist()) == (30, 30, 30)


@pytest.mark.unit
@pytest.mark.parametrize("cls_name", ["Model", "ModelFT"])
def test_copy_skips_cache_buffers_and_recomputes(pdb_dir, cls_name):
    import torchref.model as M

    cls = getattr(M, cls_name)
    model = cls(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    model.Z
    model.get_scattering_params_iso()
    model.get_vdw_radii()

    duplicate = model.copy()
    for name in CACHE_BUFFER_NAMES:
        assert name not in duplicate._buffers, f"copy carried cache buffer {name}"

    # The copy recomputes on access, to the same values but its own storage.
    assert torch.allclose(duplicate.get_vdw_radii(), model.get_vdw_radii())
    assert duplicate.vdw_radii.data_ptr() != model.vdw_radii.data_ptr()
    # The SF index cache also rebuilds lazily on the copy.
    out = duplicate.get_iso()
    assert out[0].shape[0] == len(duplicate.pdb)


@pytest.mark.unit
def test_symmetry_and_cell_caches_survive_neither_pickle_nor_deepcopy():
    from torchref.symmetry import Cell, SpaceGroup

    sg = SpaceGroup("P 21 21 21")
    sg.reciprocal  # populate the cache
    assert sg._cache
    for clone in (pickle.loads(pickle.dumps(sg)), copy.deepcopy(sg)):
        assert clone._cache == {}
        # Derived quantities recompute on access.
        assert torch.allclose(clone.reciprocal.matrices, sg.reciprocal.matrices)

    cell = Cell([40.0, 50.0, 60.0, 90.0, 90.0, 90.0])
    cell.fractional_matrix  # populate the cache
    assert cell._cache
    for clone in (pickle.loads(pickle.dumps(cell)), copy.deepcopy(cell)):
        assert clone._cache == {}
        assert torch.allclose(clone.fractional_matrix, cell.fractional_matrix)


@pytest.mark.unit
def test_forward_cache_survives_neither_pickle_nor_deepcopy(pdb_dir):
    from torchref.model.parameter_wrappers import MixedTensor

    wrapper = MixedTensor(
        initial_values=torch.arange(12.0).reshape(4, 3),
        refinable_mask=torch.tensor([True, True, False, False]),
    )
    wrapper()  # populate the forward cache
    assert wrapper._fwd_cached_output is not None

    clone = copy.deepcopy(wrapper)
    assert getattr(clone, "_fwd_cached_output", None) is None
    assert torch.allclose(clone(), wrapper())


@pytest.mark.unit
def test_atom_graph_adjacency_is_lazy_and_not_carried(pdb_dir):
    from torchref.model import Model
    from torchref.topology import build_topology

    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    topology = build_topology(model.pdb, model.restraints.cif_dict)
    atoms = topology.atoms

    assert atoms._adj_indptr is None, "adjacency built eagerly at construction"
    reference = atoms.neighbors(0)  # first access builds it
    assert atoms._adj_indptr is not None

    duplicate = atoms.copy()
    assert duplicate._adj_indptr is None, "copy carried the adjacency"
    assert torch.equal(duplicate.neighbors(0), reference)

    clone = copy.deepcopy(atoms)
    assert clone._adj_indptr is None, "deepcopy carried the adjacency"
    assert torch.equal(clone.neighbors(0), reference)
