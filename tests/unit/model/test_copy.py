"""``Model.copy`` / ``ModelFT.copy`` must carry the derived per-atom state.

Two things in ``copy()`` are neither buffers nor parameter wrappers, so the
buffer and module loops do not carry them:

* the **iso/aniso partition** (``_iso_indices``, ``_aniso_indices`` and the two
  fast-path flags) is rebuilt from ``aniso_flag`` and the heavy-atom mask by
  ``_rebuild_sf_indices``, which otherwise runs only in ``load()``. Without it a
  copy raises ``AttributeError`` from ``get_iso()``/``get_aniso()``.
* the **space group**. ``Model.spacegroup`` is a property, but ``SpaceGroup`` is
  an ``nn.Module``, so ``model.spacegroup = sg_object`` is intercepted by
  ``nn.Module.__setattr__``, stored in ``_modules`` under the property's own
  name, and the setter never runs. The copy must own its space group, not
  register the original's under a second key.

``4BX9`` is used because it carries ``ANISOU`` records, so the partition is
genuinely mixed (220 isotropic, 9973 anisotropic) rather than all-isotropic,
where an empty partition would still pass the fast path.
"""

import pytest
import torch

_MIXED_ADP_PDB = "4BX9.pdb"


@pytest.fixture(scope="module")
def mixed_adp_path(pdb_dir):
    p = pdb_dir / _MIXED_ADP_PDB
    if not p.exists():
        pytest.skip(f"{_MIXED_ADP_PDB} not available")
    return str(p)


def _load(cls, path):
    return cls(verbose=0).load_pdb(path)


@pytest.mark.unit
@pytest.mark.parametrize("cls_name", ["Model", "ModelFT"])
def test_copy_has_a_usable_iso_aniso_partition(cls_name, mixed_adp_path):
    """``get_iso``/``get_aniso`` must work on the copy and agree with the source."""
    import torchref.model as tm

    cls = getattr(tm, cls_name)
    m = _load(cls, mixed_adp_path)
    c = m.copy()

    for attr in ("_iso_indices", "_aniso_indices", "_iso_covers_all",
                 "_aniso_is_empty"):
        assert hasattr(c, attr), f"{cls_name}.copy() dropped {attr}"

    assert torch.equal(c._iso_indices, m._iso_indices)
    assert torch.equal(c._aniso_indices, m._aniso_indices)
    assert c._iso_covers_all == m._iso_covers_all
    assert c._aniso_is_empty == m._aniso_is_empty

    # ``ModelFT`` appends the per-atom form-factor tables to the same tuple, so
    # unpack positionally rather than by a fixed arity.
    iso = c.get_iso()
    aniso = c.get_aniso()
    xyz_i, occ_i = iso[0], iso[2]
    xyz_a, u_a, occ_a = aniso[0], aniso[1], aniso[2]
    assert xyz_i.shape[0] == m._iso_indices.numel()
    assert xyz_a.shape[0] == m._aniso_indices.numel()
    assert u_a.shape[-1] == 6
    assert occ_i.shape[0] == xyz_i.shape[0]
    assert occ_a.shape[0] == xyz_a.shape[0]


@pytest.mark.unit
@pytest.mark.parametrize("cls_name", ["Model", "ModelFT"])
def test_the_partition_is_mixed_so_the_test_has_teeth(cls_name, mixed_adp_path):
    """Guard the premise: an all-isotropic model would not exercise the split."""
    import torchref.model as tm

    m = _load(getattr(tm, cls_name), mixed_adp_path)
    assert m._iso_indices.numel() > 0
    assert m._aniso_indices.numel() > 0
    assert not m._iso_covers_all
    assert not m._aniso_is_empty


@pytest.mark.unit
@pytest.mark.parametrize("cls_name", ["Model", "ModelFT"])
def test_copy_owns_its_spacegroup(cls_name, mixed_adp_path):
    """The copy carries the space group without aliasing or double-registering."""
    import torchref.model as tm

    m = _load(getattr(tm, cls_name), mixed_adp_path)
    c = m.copy()

    assert c.spacegroup is not None
    assert str(c.spacegroup) == str(m.spacegroup)
    # Own object: `.to(device)` on the copy must not move the original's matrices.
    assert c.spacegroup is not m.spacegroup
    # Exactly one registration, under the private name the property reads.
    assert "spacegroup" not in c._modules
    assert "_spacegroup" in c._modules
    stray = [k for k in c.state_dict() if k.startswith("spacegroup.")]
    assert stray == [], f"copy registered a second space group: {stray}"


@pytest.mark.unit
def test_copy_can_skip_the_grid_build(mixed_adp_path):
    """``build_grid=False`` skips the grid, and setting cell+spacegroup restores it.

    Building the grid also builds the map-symmetry operator, which precomputes
    one sampling grid per symmetry operation over the whole map. A caller that
    is about to replace the cell, the spacegroup or ``max_res`` would have that
    work thrown away, because each of those setters rebuilds the FFT submodule.
    """
    from torchref.model import ModelFT

    m = _load(ModelFT, mixed_adp_path)
    assert m._fft is not None and m._fft.real_space_grid is not None

    lean = m.copy(build_grid=False)
    assert lean._fft is not None
    assert lean._fft.real_space_grid is None
    assert lean._fft.map_symmetry is None

    full = m.copy()
    assert full._fft.real_space_grid is not None

    # The skipped grid is recoverable: this is what the cell setter triggers.
    lean.setup_grid(max_res=m.max_res)
    assert lean._fft.real_space_grid is not None
    assert torch.equal(lean._fft.gridsize, full._fft.gridsize)


@pytest.mark.unit
def test_skipping_the_grid_build_does_not_change_structure_factors(mixed_adp_path):
    """The two copies must give identical ``F_calc`` once each has a grid.

    ``build_grid`` is a pure waste-removal switch: the grid it skips is rebuilt
    by the cell/spacegroup setters before any structure factor is computed, so
    no amplitude may depend on it.
    """
    from torchref.model import ModelFT

    m = _load(ModelFT, mixed_adp_path)
    hkl = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 3], [5, -2, 1], [7, 7, 7]],
        dtype=torch.long,
    )

    with torch.no_grad():
        f_full = m.copy().get_structure_factor(hkl, recalc=True)
        lean = m.copy(build_grid=False)
        lean.setup_grid(max_res=m.max_res)
        f_lean = lean.get_structure_factor(hkl, recalc=True)

    assert torch.equal(f_lean, f_full)
