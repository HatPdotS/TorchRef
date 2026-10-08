"""Exercise the hydrogen flags from CLI parsing through deposited-model loading."""

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
    """``--hydrogens add`` generates missing hydrogens for PDB and mmCIF inputs."""
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
        argv += ["--hydrogens", "add"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(_ModelLoaded):
        refine.main()

    (model,) = loaded
    assert model.ctx.hydrogens == ("add" if add_hydrogens else "keep")
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
        "--hydrogens",
        "add",
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
@pytest.mark.parametrize("hydrogens", ["keep", "add", "strip"])
def test_empty_refinement_preserves_hydrogen_setting(hydrogens: str) -> None:
    """An empty refinement shell forwards the policy to its model too."""
    refinement = LBFGSRefinement(verbose=0, hydrogens=hydrogens, hydrogen_mode="atoms")
    assert refinement.model.ctx.hydrogens == hydrogens


@pytest.mark.integration
def test_cli_refuses_riding_on_stripped_hydrogens(
    test_files_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--hydrogens strip --hydrogen-mode riding`` fails before anything loads."""
    from torchref.cli import refine

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
        "--hydrogens",
        "strip",
        "--hydrogen-mode",
        "riding",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="requires hydrogens=.keep. or .add."):
        refine.main()


@pytest.mark.integration
def test_difference_refine_keeps_input_hydrogens(
    test_files_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """difference-refine writes the hydrogens its models were refined with."""
    import gemmi

    from torchref.cli import collection_difference_refine

    pdb = test_files_dir / "pdb" / "1AK5_with_H.pdb"
    mtz = test_files_dir / "mtz" / "1AK5.mtz"
    outdir = tmp_path / "diff"
    argv = [
        "torchref.difference-refine",
        "-dm", str(pdb), "-lm", str(pdb),
        "-dsf", str(mtz), "-lsf", str(mtz),
        "--fraction", "0.3",
        "--n-cycles", "0",
        "--output-format", "pdb",
        "--no-header",
        "--device", "cpu",
        "-o", str(outdir),
        "-v", "0",
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    assert collection_difference_refine.main() == 0

    def n_hydrogens(path: Path) -> int:
        structure = gemmi.read_structure(str(path))
        return sum(cra.atom.is_hydrogen() for cra in structure[0].all())

    assert n_hydrogens(outdir / "fractions_70_30_light.pdb") == n_hydrogens(pdb) > 0
