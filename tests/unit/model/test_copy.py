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
