"""Column-selection flags shared by the command-line tools."""

import argparse

import pytest

from torchref.cli._common import add_dual_model_args

pytestmark = pytest.mark.integration

DUAL_INPUTS = ["-dm", "d.pdb", "-lm", "l.pdb", "-dsf", "d.mtz", "-lsf", "l.mtz"]


@pytest.mark.parametrize("flag", ["-cphi-dark", "-cphi-light"])
def test_dual_tools_have_no_phase_column_flag(flag):
    """No reader takes an observed phase column, so naming one is an argparse error."""
    parser = argparse.ArgumentParser()
    add_dual_model_args(parser)
    parser.parse_args(DUAL_INPUTS + ["--fraction", "0.3"])

    with pytest.raises(SystemExit):
        parser.parse_args(DUAL_INPUTS + ["--fraction", "0.3", flag, "PHIB"])
