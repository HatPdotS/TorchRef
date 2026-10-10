"""Column-selection flags shared by the command-line tools."""

import argparse
import sys

import pytest

from torchref.cli._common import add_dual_model_args, build_dual_column_names

pytestmark = pytest.mark.integration


def _parse_dual(*extra, dark="dark.mtz", light="light.mtz"):
    parser = argparse.ArgumentParser()
    add_dual_model_args(parser)
    inputs = ["-dm", "d.pdb", "-lm", "l.pdb", "-dsf", dark, "-lsf", light]
    return parser.parse_args(inputs + ["--fraction", "0.3", *extra])


@pytest.mark.parametrize("flag", ["-cphi-dark", "-cphi-light"])
def test_dual_tools_have_no_phase_column_flag(flag):
    """No reader takes an observed phase column, so naming one is an argparse error."""
    _parse_dual()

    with pytest.raises(SystemExit):
        _parse_dual(flag, "PHIB")


def test_dual_tools_refuse_columns_for_sf_mmcif():
    """A column named for an SF-mmCIF file stops the run instead of being ignored."""
    args = _parse_dual("-csf-dark", "FP", "-csf-light", "FP", dark="dark.cif")
    with pytest.raises(SystemExit, match="-csf-dark/-csig-dark"):
        build_dual_column_names(args)

    args = _parse_dual("-csf-light", "FP", "-csig-light", "SIGFP", dark="dark.cif")
    assert build_dual_column_names(args) == (None, {"F": "FP", "SIGF": "SIGFP"})


def test_refine_refuses_columns_for_sf_mmcif(test_files_dir, tmp_path, monkeypatch):
    """``torchref.refine -csf`` on SF-mmCIF exits naming the flag, before any load."""
    from torchref.cli import refine

    argv = [
        "torchref.refine",
        "-m",
        str(test_files_dir / "pdb" / "1DAW.pdb"),
        "-sf",
        str(test_files_dir / "cif_sf" / "1DAW-sf.cif"),
        "-csf",
        "FP",
        "-o",
        str(tmp_path / "refined"),
        "-v",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="-csf/-csig"):
        refine.main()
