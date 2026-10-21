"""Argument checks that ``torchref.difference-map`` makes before reading any file."""

import sys

import pytest

from torchref.cli import difference_map

pytestmark = pytest.mark.unit


def _run(monkeypatch, *extra, output="out.mtz"):
    argv = [
        "torchref.difference-map",
        "-dm", "missing_dark.pdb",
        "-dsf", "missing_dark.mtz", "-lsf", "missing_light.mtz",
        "-o", output,
        *extra,
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    return difference_map.main()


def _refused(monkeypatch, capsys, *extra, **kwargs):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, *extra, **kwargs)
    assert exc.value.code == 2
    return capsys.readouterr().err


def test_valid_arguments_reach_the_file_check(monkeypatch, capsys):
    assert _run(monkeypatch) == 1
    assert "file not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--fraction", "0.3"], "--fraction needs -lm"),
        (["-lm", "light.pdb"], "--fraction is required"),
        (["--all-columns"], "--all-columns needs -lm"),
        (["--ded-weight", "inverse_variance"], "--ded-weight only weights"),
        (
            ["-lm", "light.pdb", "--fraction", "0.3", "--ded-weight", "none"],
            "--ded-weight only weights",
        ),
    ],
)
def test_ignored_arguments_are_refused(monkeypatch, capsys, extra, message):
    assert message in _refused(monkeypatch, capsys, *extra)


def test_ded_weight_is_accepted_where_it_weights_columns(monkeypatch, capsys):
    extra = ["-lm", "light.pdb", "--fraction", "0.3", "--all-columns"]
    assert _run(monkeypatch, *extra, "--ded-weight", "none") == 1
    assert "file not found" in capsys.readouterr().err


@pytest.mark.parametrize("output", ["out/", "out.map", "results"])
def test_output_must_name_an_mtz_file(monkeypatch, capsys, output):
    assert "-o/--output" in _refused(monkeypatch, capsys, output=output)


def test_output_refuses_an_existing_directory(monkeypatch, capsys, tmp_path):
    (tmp_path / "out.mtz").mkdir()
    err = _refused(monkeypatch, capsys, output=str(tmp_path / "out.mtz"))
    assert "not a directory" in err
