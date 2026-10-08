"""Argument checks that ``torchref.difference-refine`` makes before reading any file."""

import sys

import pytest

from torchref.cli import collection_difference_refine

pytestmark = pytest.mark.unit


def _run(monkeypatch, *extra):
    argv = [
        "torchref.difference-refine",
        "-dm", "missing_dark.pdb", "-lm", "missing_light.pdb",
        "-dsf", "missing_dark.mtz", "-lsf", "missing_light.mtz",
        "--fraction", "0.3",
        "-o", "missing_out",
        *extra,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    return collection_difference_refine.main()


@pytest.mark.parametrize(
    "extra",
    [
        ["--weights", '{"xray/difference": 2}'],
        ["--difference-target", "difference_sd"]
        + ["--weights", '{"xray/difference_sd": 2}'],
    ],
)
def test_weights_refuses_the_scheduled_row(monkeypatch, capsys, extra):
    """--weight-schedule alone sets the row it drives."""
    assert _run(monkeypatch, *extra) == 1
    err = capsys.readouterr().err
    assert "--weight-schedule" in err
    assert "not found" not in err


def test_weights_accepts_the_unscheduled_row(monkeypatch, capsys):
    assert _run(monkeypatch, "--weights", '{"xray/difference_sd": 0.5}') == 1
    err = capsys.readouterr().err
    assert "--weight-schedule" not in err
    assert "file not found" in err


def test_help_shows_no_difference_weight(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "--help")
    assert '"xray/difference' not in capsys.readouterr().out
