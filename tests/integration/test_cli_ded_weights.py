"""The registered difference weights through the CLIs.

Pinned: ``torchref.difference-map`` writes ``DF`` with one mean-one weight column per
scheme and ``KSCALE``; ``torchref.mtz2map`` builds the weighted map from those columns
and the electrons map is the volume-normalised synthesis on the absolute scale;
``torchref.validate-ded`` reports every scheme side by side and records a fallback;
``torchref.difference-refine`` runs with the ``difference_sd`` row and reports the fit.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

DIFF_COLUMNS = {
    "Fo_dark": "SFAmplitude",
    "SIGFo_dark": "Stddev",
    "Fo_light": "SFAmplitude",
    "SIGFo_light": "Stddev",
    "DF": "SFAmplitude",
    "SIGDF": "Stddev",
    "PHDELWT": "Phase",
    "W_IVW": "Weight",
    "W_SD": "Weight",
    "KSCALE": "MTZReal",
    "Fc_dark": "SFAmplitude",
    "FreeR_flag_dark": "MTZInt",
    "FreeR_flag_light": "MTZInt",
}


@pytest.fixture(scope="module")
def pair(mtz_dir, pdb_dir, tmp_path_factory):
    """A dark/light pair from 1DAW with a perturbed light state.

    The light amplitudes carry an added difference proportional to ``F`` with a
    resolution-dependent power, so the sigma_D fit has signal to find; the dark set
    keeps the deposited values. The light model is the dark one shifted by 0.2 A.
    """
    import torch

    from torchref import ReflectionData
    from torchref.config import get_int_dtype

    mtz = mtz_dir / "1DAW.mtz"
    pdb = pdb_dir / "1DAW.pdb"
    assert mtz.is_file() and pdb.is_file()
    out = tmp_path_factory.mktemp("ded_weights_cli")

    data = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    n = len(data)
    idx = torch.arange(n, dtype=get_int_dtype())
    dark = data.__select__(idx < int(n * 0.97))
    dark.write_mtz(str(out / "dark.mtz"))

    sel = data.__select__(idx >= int(n * 0.03))
    g = torch.Generator().manual_seed(11)
    f = sel.F
    dss = 1.0 / sel.resolution**2
    change = 0.08 * f * torch.exp(-2.0 * dss) * torch.randn(len(sel), generator=g)
    light = ReflectionData.from_tensors(
        hkl=sel.hkl,
        F=(f + change).clamp(min=0.0),
        F_sigma=sel.F_sigma,
        cell=sel.cell,
        spacegroup=sel.spacegroup,
        rfree_flags=sel.rfree_flags,
        device="cpu",
        verbose=0,
    )
    light.write_mtz(str(out / "light.mtz"))

    lines = []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) + 0.2
            line = line[:30] + f"{x:8.3f}" + line[38:]
        lines.append(line)
    (out / "light.pdb").write_text("\n".join(lines) + "\n")
    return {"dir": out, "pdb": pdb, "light_pdb": out / "light.pdb"}


def _run(project_root, module, *argv, timeout=1800):
    script = project_root / "torchref" / "cli" / module
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(script), *map(str, argv)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    assert proc.returncode == 0, (
        f"{module} failed ({proc.returncode})\nstdout tail:\n{proc.stdout[-2000:]}"
        f"\nstderr tail:\n{proc.stderr[-3000:]}"
    )
    return proc


@pytest.fixture(scope="module")
def diff_mtz(project_root, pair):
    out = pair["dir"] / "diff.mtz"
    _run(
        project_root,
        "difference_map.py",
        "-dm",
        pair["pdb"],
        "-dsf",
        pair["dir"] / "dark.mtz",
        "-lsf",
        pair["dir"] / "light.mtz",
        "--dmin",
        "2.2",
        "--device",
        "cpu",
        "--ded-weight",
        "sigma_d",
        "-v",
        "1",
        "-o",
        out,
    )
    return out


def _read(path):
    import reciprocalspaceship as rs

    return rs.read_mtz(str(path))


def test_difference_map_writes_df_weights_and_scale(diff_mtz):
    df = _read(diff_mtz)
    assert {c: str(df.dtypes[c]) for c in df.columns} == DIFF_COLUMNS
    for col in ("W_IVW", "W_SD"):
        w = df[col].to_numpy().astype(float)
        assert np.isfinite(w).all() and (w >= 0).all()
        assert abs(w.mean() - 1.0) < 1e-4
    assert (df["KSCALE"].to_numpy().astype(float) > 0).all()
    # The sigma_D weights favour the strong reflections, inverse variance does not.
    f = df["Fo_dark"].to_numpy().astype(float)
    w_sd = df["W_SD"].to_numpy().astype(float)
    strong = f > np.median(f)
    assert w_sd[strong].mean() > w_sd[~strong].mean()


def test_mtz2map_builds_the_weighted_and_electron_maps(project_root, pair, diff_mtz):
    import gemmi

    out = pair["dir"]
    _run(
        project_root,
        "mtz2map.py",
        "-sf",
        diff_mtz,
        "-csf",
        "DF",
        "-cw",
        "W_SD",
        "-cphi",
        "PHDELWT",
        "--device",
        "cpu",
        "-o",
        out / "sd_sigma.ccp4",
    )
    _run(
        project_root,
        "mtz2map.py",
        "-sf",
        diff_mtz,
        "-csf",
        "DF",
        "-cw",
        "W_SD",
        "-cphi",
        "PHDELWT",
        "--units",
        "raw",
        "--device",
        "cpu",
        "-o",
        out / "sd_raw.ccp4",
    )
    _run(
        project_root,
        "mtz2map.py",
        "-sf",
        diff_mtz,
        "-csf",
        "DF",
        "-cw",
        "W_SD",
        "-cphi",
        "PHDELWT",
        "--units",
        "electrons",
        "--device",
        "cpu",
        "-o",
        out / "sd_e.ccp4",
    )
    sigma = np.array(gemmi.read_ccp4_map(str(out / "sd_sigma.ccp4")).grid, copy=False)
    raw = np.array(gemmi.read_ccp4_map(str(out / "sd_raw.ccp4")).grid, copy=False)
    electrons = np.array(gemmi.read_ccp4_map(str(out / "sd_e.ccp4")).grid, copy=False)
    assert abs(sigma.std() - 1.0) < 1e-3 and abs(sigma.mean()) < 1e-3
    # Same map up to normalisation: the correlation is one.
    assert np.corrcoef(sigma.ravel(), raw.ravel())[0, 1] > 0.9999
    # Dividing by the per-reflection KSCALE reshapes the map slightly, so electrons is
    # highly but not perfectly correlated with the sigma map.
    assert 0.9 < np.corrcoef(sigma.ravel(), electrons.ravel())[0, 1] < 0.9999
    ratio = electrons.std() / raw.std()
    assert np.isfinite(ratio) and ratio > 0


def test_validate_ded_reports_every_scheme(project_root, pair):
    out = pair["dir"] / "val"
    proc = _run(
        project_root,
        "validate_ded.py",
        "-dsf",
        pair["dir"] / "dark.mtz",
        "-lsf",
        pair["dir"] / "light.mtz",
        "-dm",
        pair["pdb"],
        "-lm",
        pair["light_pdb"],
        "--fraction",
        "0.3",
        "--dmin",
        "2.2",
        "--device",
        "cpu",
        "--ded-weight",
        "sigma_d",
        "-v",
        "1",
        "-o",
        out,
    )
    results = json.loads((out / "validate_ded_results.json").read_text())
    assert results["weights"]["requested"] == "sigma_d"
    assert results["weights"]["applied"] in ("sigma_d", "inverse_variance")
    assert set(results["by_weight"]) == {"none", "inverse_variance", "sigma_d"}
    for entry in results["by_weight"].values():
        assert np.isfinite(entry["reciprocal_cc_overall"])
        assert "full_cell" in entry["realspace_correlation"]
    headline = results["by_weight"][results["weights"]["applied"]]
    assert results["reciprocal_cc_overall"] == pytest.approx(
        headline["reciprocal_cc_overall"], abs=1e-3
    )
    assert "weights " in proc.stdout and "sigma_d" in proc.stdout


def test_difference_refine_runs_the_sigma_d_row(project_root, pair):
    out = pair["dir"] / "refine"
    _run(
        project_root,
        "collection_difference_refine.py",
        "-dm",
        pair["pdb"],
        "-lm",
        pair["light_pdb"],
        "-dsf",
        pair["dir"] / "dark.mtz",
        "-lsf",
        pair["dir"] / "light.mtz",
        "--fraction",
        "0.25",
        "--difference-target",
        "difference_sd",
        "--n-cycles",
        "1",
        "--n-steps",
        "1",
        "--max-iter",
        "3",
        "--dmin",
        "2.2",
        "--device",
        "cpu",
        "--verbose",
        "0",
        "-o",
        out,
    )
    summaries = list(out.glob("*_summary.json"))
    assert len(summaries) == 1
    results = json.loads(summaries[0].read_text())["results"]
    assert results["ded_weights"]["scheme"] == "inverse_variance"
    assert results["ded_weights"]["applied"] == "inverse_variance"
    assert "gamma" in results["ded_weights"]["sigma_d"]
