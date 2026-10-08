"""Shared argument helpers of the torchref command-line tools."""

import argparse
import json

import pytest

from torchref.cli._common import add_dual_model_args, parse_weights

pytestmark = pytest.mark.unit

_DUAL_INPUTS = ["-dm", "d.pdb", "-lm", "l.pdb", "-dsf", "d.mtz", "-lsf", "l.mtz"]


def test_parse_weights_long_inline_json(tmp_path):
    """An inline dict longer than a file name parses; a JSON file path still does."""
    # No "/": the whole string is then a single path component, too long for one.
    user = {f"similarity_{i:03d}": 1.0 for i in range(20)}
    inline = json.dumps(user)
    assert len(inline) > 255
    assert parse_weights(inline) == (user, None)

    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"xray": 2.0}))
    assert parse_weights(str(path), defaults={"geometry": 1.0}) == (
        {"geometry": 1.0, "xray": 2.0},
        None,
    )


def test_parse_weights_reports_unreadable_input(tmp_path):
    weights, err = parse_weights(str(tmp_path / "absent.json"), defaults={"xray": 1.0})
    assert weights == {"xray": 1.0}
    assert "--weights" in err


@pytest.mark.parametrize("value", ["1.5", "0", "-0.2"])
def test_fraction_range_refuses(value):
    parser = argparse.ArgumentParser()
    add_dual_model_args(parser, fraction_required=False)
    with pytest.raises(SystemExit):
        parser.parse_args([*_DUAL_INPUTS, "--fraction", value])


@pytest.mark.parametrize("value", [0.37, 1.0])
def test_fraction_range_accepts(value):
    parser = argparse.ArgumentParser()
    add_dual_model_args(parser, fraction_required=False)
    args = parser.parse_args([*_DUAL_INPUTS, "--fraction", str(value)])
    assert args.fraction == value
