"""Shared argument helpers of the torchref command-line tools."""

import json

import pytest

from torchref.cli._common import parse_weights

pytestmark = pytest.mark.unit


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

