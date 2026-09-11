"""Exercise hydrogen opt-in from CLI parsing through deposited-model loading."""

import sys
from pathlib import Path

import pytest

from torchref.refinement.base_refinement import Refinement
from torchref.refinement.lbfgs_refinement import LBFGSRefinement


class _ModelLoaded(Exception):
    """Stop after real model loading, before scaling and optimization."""


@pytest.mark.integration
@pytest.mark.parametrize("add_hydrogens", [False, True])
@pytest.mark.parametrize("model_format", ["pdb", "cif"])
def test_cli_hydrogen_generation_is_opt_in(
    test_files_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    add_hydrogens: bool,
    model_format: str,
) -> None:
    """The CLI flag generates missing hydrogens for PDB and mmCIF inputs."""
    from torchref.cli import refine

    loaded = []

    def stop_after_loading(refinement: Refinement) -> None:
        loaded.append(refinement.model)
        raise _ModelLoaded

    monkeypatch.setattr(Refinement, "_sync_model_cell_to_data", stop_after_loading)
    argv = [
        "torchref.refine",
        "-m",
        str(test_files_dir / model_format / f"1DAW.{model_format}"),
        "-sf",
        str(test_files_dir / "mtz" / "1DAW.mtz"),
        "-o",
        str(tmp_path / "refined"),
        "-v",
        "0",
    ]
    if add_hydrogens:
        argv.append("--add-hydrogens")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(_ModelLoaded):
        refine.main()

    (model,) = loaded
    assert model.ctx.add_hydrogens is add_hydrogens
    assert len(model.pdb) > 0
    n_hydrogens = int(model.pdb["element"].str.strip().eq("H").sum())
    assert (n_hydrogens > 0) is add_hydrogens


@pytest.mark.integration
def test_cli_generates_from_the_user_cif(
    test_files_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--cif`` reaches the model before it loads, so generation reads it."""
    from torchref.cli import refine

    loaded = []

    def stop_after_loading(refinement: Refinement) -> None:
        loaded.append(refinement.model)
        raise _ModelLoaded

    monkeypatch.setattr(Refinement, "_sync_model_cell_to_data", stop_after_loading)
    cif = str(test_files_dir / "restraints" / "GLU_renamed.cif")
    argv = [
        "torchref.refine",
        "-m",
        str(test_files_dir / "pdb" / "1DAW.pdb"),
        "-sf",
        str(test_files_dir / "mtz" / "1DAW.mtz"),
        "-o",
        str(tmp_path / "refined"),
        "-v",
        "0",
        "--add-hydrogens",
        "--cif",
        cif,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_ModelLoaded):
        refine.main()

    (model,) = loaded
    registered = model.ctx.cif_path
    assert (registered if isinstance(registered, list) else [registered]) == [cif]
    pdb = model.pdb
    glu_h = (pdb["resname"].str.strip() == "GLU") & (pdb["element"].str.strip() == "H")
    assert {"HAX", "HBX", "HBY", "HGX", "HGY"} <= set(pdb.loc[glu_h, "name"].str.strip())


@pytest.mark.unit
@pytest.mark.parametrize("add_hydrogens", [False, True])
def test_empty_refinement_preserves_hydrogen_setting(add_hydrogens: bool) -> None:
    """An empty refinement shell forwards the setting to its model too."""
    refinement = LBFGSRefinement(verbose=0, add_hydrogens=add_hydrogens)
    assert refinement.model.ctx.add_hydrogens is add_hydrogens
