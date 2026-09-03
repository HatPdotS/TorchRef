"""Guards for two aliasing traps in the molecular-replacement pipeline.

`Model.rotate` and `Model.translate` mutate in place and return ``self``. Every
call site that treats them as returning a fresh model therefore has to
``.copy()`` first. `MolecularReplacementPipeline._make_rotated` is called once
per rotation candidate off the same ``self.model``, so without the copy
candidate *k+1* is evaluated at an orientation composed on top of candidate *k*
and ``self.model`` is destroyed along the way.

`Model.spacegroup` is a property, but ``SpaceGroup`` is an ``nn.Module``:
assigning a SpaceGroup *object* is intercepted by ``nn.Module.__setattr__``,
stored in ``_modules`` under the property's own name, and the setter never runs.
The pipeline builds P1 copies for its dense-transform stages, so a silent no-op
there means the "P1 search model" still carries the crystal symmetry.

Both are pinned here rather than only in the slow integration tests, which
``--run-slow`` gates off by default.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def small_model(pdb_dir):
    from torchref.model import ModelFT

    p = pdb_dir / "1DAW.pdb"
    if not p.exists():
        pytest.skip("1DAW.pdb not available")
    return ModelFT(verbose=0).load_pdb(str(p))


def _peak(alpha, beta, gamma):
    from torchref.experimental.alignment.frf.types import RotationPeak

    return RotationPeak(alpha=alpha, beta=beta, gamma=gamma, score=1.0, sigma=1.0)


def test_rotate_mutates_in_place_and_returns_self(small_model):
    """The premise. If this ever changes, the copies below stop being necessary."""
    m = small_model.copy()
    before = m.xyz().clone()
    R = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=m.dtype_float,
    )
    out = m.rotate(R)
    assert out is m, "rotate no longer returns self"
    assert not torch.allclose(m.xyz(), before), "rotate no longer mutates in place"


def test_make_rotated_leaves_the_search_model_untouched(small_model):
    """The pipeline's own model must survive candidate generation."""
    from torchref.experimental.alignment.pipeline import MolecularReplacementPipeline

    pipe = object.__new__(MolecularReplacementPipeline)
    pipe.model = small_model
    reference = small_model.xyz().clone()

    rotated, _ = pipe._make_rotated(_peak(0.3, 0.7, 1.1))

    assert rotated is not pipe.model
    assert torch.allclose(pipe.model.xyz(), reference), (
        "_make_rotated mutated the pipeline's search model"
    )


def test_successive_candidates_do_not_compound(small_model):
    """Candidate k+1 must not be rotated on top of candidate k."""
    from torchref.experimental.alignment.pipeline import MolecularReplacementPipeline

    pipe = object.__new__(MolecularReplacementPipeline)
    pipe.model = small_model

    p1 = _peak(0.3, 0.7, 1.1)
    p2 = _peak(2.0, 1.3, 0.4)

    first, _ = pipe._make_rotated(p1)
    second, _ = pipe._make_rotated(p2)

    # A fresh pipeline that only ever sees p2 is the ground truth for p2.
    solo = object.__new__(MolecularReplacementPipeline)
    solo.model = small_model.copy()
    expected, _ = solo._make_rotated(p2)

    assert torch.allclose(second.xyz(), expected.xyz(), atol=1e-5), (
        "the second candidate depends on the first -- rotations are compounding"
    )
    assert not torch.allclose(first.xyz(), second.xyz()), (
        "the two candidates are identical; the peaks chosen do not discriminate"
    )


def test_spacegroup_name_assignment_works(small_model):
    """The supported form: pass the space-group NAME."""
    m = small_model.copy()
    m.spacegroup = "P 1"
    assert m.spacegroup.number == 1
    assert int(m.spacegroup.matrices.shape[0]) == 1


def test_spacegroup_object_assignment_now_takes_effect(small_model):
    """The trap this used to pin is gone, fixed at the root rather than avoided.

    ``Model.spacegroup`` is a property, but ``SpaceGroup`` is an ``nn.Module``, so
    ``model.spacegroup = sg_object`` used to be intercepted by
    ``nn.Module.__setattr__``, filed under ``_modules["spacegroup"]`` with the
    setter never running -- the assignment silently did nothing, and the name was
    then a registered child module, so the *correct* string assignment afterwards
    raised ``TypeError``. Call sites worked around it by passing a name string.

    The space group now lives on ``ModelContext``, which is deliberately a
    dataclass and not an ``nn.Module``, so there is nothing to intercept. Both
    forms work and neither registers a submodule.
    """
    from torchref.symmetry import SpaceGroup

    m = small_model.copy()
    assert m.spacegroup.number != 1, "1DAW should not already be P1"

    m.spacegroup = SpaceGroup("P 1")
    assert m.spacegroup.number == 1, "object assignment did not take effect"
    assert "spacegroup" not in m._modules, (
        "the space group was registered as a child module -- the interception "
        "this test exists for has come back"
    )

    # The string form must still work afterwards, which is what used to raise.
    m.spacegroup = "P 21 21 21"
    assert m.spacegroup.number == 19
