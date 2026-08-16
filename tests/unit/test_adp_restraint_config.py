"""ADP restraint configuration must survive target rebuilds.

`Refinement._init_targets` rebuilds `adp_target` from scratch -- once per
resolution cutoff inside `refine_rigid_body`, and again on the ensemble and
`create_from_state_dict` paths. Configuration passed to the constructor has to
be reapplied on every one of those rebuilds; anything assigned to the target
object afterwards is discarded, silently.

These tests exercise `TotalADPTarget` directly so they need only a Model, no
reflection data or refinement run.
"""

import pytest

from torchref.refinement.targets.combined import TotalADPTarget


DEFAULT_SIMU_SIGMA = 2.0
DEFAULT_SIMU_SIGMA_ANISO = 1.0


def _build(model, **kwargs):
    return TotalADPTarget(model, verbose=0, **kwargs)


def test_defaults_unchanged_without_config(loaded_model):
    """No config means exactly the previous behaviour."""
    adp = _build(loaded_model)
    assert adp["simu"].simu_sigma == pytest.approx(DEFAULT_SIMU_SIGMA)
    assert adp["simu"].simu_sigma_aniso == pytest.approx(DEFAULT_SIMU_SIGMA_ANISO)
    assert adp.component_config == {}


def test_config_reaches_the_component(loaded_model):
    adp = _build(loaded_model, component_config={"simu": {"simu_sigma": 0.4}})
    assert adp["simu"].simu_sigma == pytest.approx(0.4)
    # Untouched kwargs keep their defaults.
    assert adp["simu"].simu_sigma_aniso == pytest.approx(DEFAULT_SIMU_SIGMA_ANISO)


def test_config_survives_a_rebuild(loaded_model):
    """The regression: a rebuilt target must come back configured.

    This is what `refine_rigid_body` does per resolution cutoff, via
    `_rebind_for_data` -> `_init_targets`.
    """
    config = {"simu": {"simu_sigma": 0.4, "simu_sigma_aniso": 0.2}}
    first = _build(loaded_model, component_config=config)
    rebuilt = _build(loaded_model, component_config=first.component_config)

    assert rebuilt["simu"].simu_sigma == pytest.approx(0.4)
    assert rebuilt["simu"].simu_sigma_aniso == pytest.approx(0.2)


def test_post_construction_assignment_does_not_survive_a_rebuild(loaded_model):
    """Pin the behaviour that motivates the constructor argument.

    Assigning to the target is still legal and still takes effect immediately --
    it just cannot outlive the object. Documenting that here so the next reader
    does not "fix" the setter instead of using `adp_restraints`.
    """
    first = _build(loaded_model)
    first["simu"].simu_sigma = 0.4
    assert first["simu"].simu_sigma == pytest.approx(0.4)

    rebuilt = _build(loaded_model, component_config=first.component_config)
    assert rebuilt["simu"].simu_sigma == pytest.approx(DEFAULT_SIMU_SIGMA)


def test_config_is_copied_not_aliased(loaded_model):
    """Mutating the caller's dict afterwards must not change the target."""
    config = {"simu": {"simu_sigma": 0.4}}
    adp = _build(loaded_model, component_config=config)
    config["simu"]["simu_sigma"] = 99.0
    assert adp["simu"].simu_sigma == pytest.approx(0.4)
    assert adp.component_config["simu"]["simu_sigma"] == pytest.approx(0.4)


def test_unknown_component_raises(loaded_model):
    """A name that reaches no component is a silent no-op -- the exact failure
    mode this machinery exists to prevent. It must raise instead."""
    with pytest.raises(ValueError, match="no such component"):
        _build(loaded_model, component_config={"simuu": {"simu_sigma": 0.4}})


def test_unknown_kwarg_raises(loaded_model):
    """A misspelled kwarg must not be swallowed either."""
    with pytest.raises(TypeError):
        _build(loaded_model, component_config={"simu": {"simu_sgima": 0.4}})
