"""``torchref.add-metadata`` annotates a deposited file without re-refining it."""

import sys

import gemmi
import pytest

from torchref.cli import add_metadata

pytestmark = pytest.mark.integration


def test_cif_annotation_keeps_the_input_refinement(cif_dir, tmp_path, monkeypatch):
    """An mmCIF input keeps its ``_refine`` record and authors under a new title."""
    output = tmp_path / "annotated.cif"
    argv = ["torchref.add-metadata", "-i", str(cif_dir / "1DAW.cif"), "-o", str(output)]
    monkeypatch.setattr(sys, "argv", argv + ["--title", "Annotated", "-v", "0"])

    assert add_metadata.main() == 0

    block = gemmi.cif.read(str(output)).sole_block()
    assert block.find_value("_refine.ls_R_factor_R_work") == "0.2120000"
    authors = [gemmi.cif.as_string(v) for v in block.find_values("_audit_author.name")]
    assert authors[0] == "Niefind, K."
    assert gemmi.cif.as_string(block.find_value("_struct.title")) == "Annotated"
