"""A copy must be usable and independent, however the state gets there.

Two things in ``copy()`` are neither buffers nor parameter wrappers, so the
buffer and module loops do not carry them, and each is handled by the model
rather than by ``copy()`` itself:

* the **iso/aniso partition** (``_iso_indices``, ``_aniso_indices`` and the two
  fast-path flags) is derived on access and keyed on ``aniso_flag``'s identity.
  It used to be rebuilt eagerly, which is what made this fragile: a copy is
  constructed, *then* has its context replaced and its buffers cloned, so
  eagerly-built indices described the wrong ``aniso_flag`` -- and silently, since
  a stale partition gathers the wrong atoms rather than raising.
* the **space group**, which lives on ``ModelContext``. That is deliberately a
  dataclass and not an ``nn.Module``, so assigning one cannot be intercepted by
  ``nn.Module.__setattr__`` and land in ``_modules`` under the property's own
  name. The copy must own its space group, not alias the original's.

These assert the *outcome* -- a copy whose partition is right and whose space
group is its own -- so they keep their teeth regardless of which mechanism
delivers it.

A third pair of tests here covered ``ModelFT.copy(build_grid=)``, which skipped
building a real-space grid that the cell/spacegroup setters immediately replace.
That option is gone: ``real_space_grid`` is legacy -- the density splat
reconstructs voxel positions from ``frac_matrix`` and never reads it -- so the
waste is being removed where it is produced rather than worked around here.

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

    # Mutating the original's flags after the copy must not reach the copy, and
    # must be picked up by the original -- the property that eager rebuilding
    # could not give us.
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
    # The space group is context state, not a submodule: nothing may register it
    # under the property's own name, which is how the original bug manifested.
    assert "spacegroup" not in c._modules
    stray = [k for k in c.state_dict() if k.startswith("spacegroup.")]
    assert stray == [], f"copy registered a second space group: {stray}"
