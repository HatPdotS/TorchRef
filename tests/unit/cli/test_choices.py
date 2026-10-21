"""The torch-free choice lists in ``torchref.cli._choices`` match their sources."""

import pytest

from torchref.cli import _choices

pytestmark = pytest.mark.unit


def test_ded_weight_schemes_match_the_registry():
    from torchref.maps.ded_weights import DEFAULT_SCHEME, SCHEMES

    assert _choices.DED_WEIGHT_SCHEMES == tuple(SCHEMES)
    assert _choices.DEFAULT_DED_WEIGHT_SCHEME == DEFAULT_SCHEME


def test_scale_targets_match_the_scaler():
    from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET, SCALE_TARGETS

    assert _choices.SCALE_TARGETS == tuple(SCALE_TARGETS)
    assert _choices.DEFAULT_SCALE_TARGET == DEFAULT_SCALE_TARGET


def test_xray_target_docs_match_the_table():
    from torchref.refinement.targets.xray._specs import XRAY_TARGETS

    assert _choices.XRAY_TARGET_DOCS == tuple(
        (spec.name, spec.doc) for spec in XRAY_TARGETS.specs
    )
