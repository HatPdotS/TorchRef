"""The difference-refinement MTZ layout, pinned.

``write_results_mtz`` assigns MTZ column types from hard-coded name lists and never calls
``infer_mtz_dtypes()``, so a column added to the output dict but missed in the type lists
is written with whatever dtype numpy produced -- silently, and into a file that gets
deposited. Nothing else in the suite asserts on these names.

Two things are checked: the baseline column set is unchanged by the two-moment work, and
under ``--two-moment`` the thirteen extra columns appear with the right types and are
internally consistent.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


BASELINE_COLUMNS = {
    "Fo_dark", "SIGFo_dark", "Fo_light", "SIGFo_light",
    "DF", "SIGDF", "WDF",
    "Fc_dark", "Fc_light", "DFc", "DFc_complex",
    "2mDFop-DFc", "mDFop-DFc",
    "PHIC_dark", "PHIC_mixed", "PHIC_diff", "PHIC_light",
    "Fextp", "2Fextp-Fc", "Fextp-Fc",
    "Fextc", "SIGFextc", "2Fextc-Fc", "Fextc-Fc",
    "Fextb", "SIGFextb", "2Fextb-Fc", "Fextb-Fc",
    "FreeR_flag_dark", "FreeR_flag_light",
}

TWO_MOMENT_COLUMNS = {
    "Io_light": "Intensity",
    "SIGIo_light": "Stddev",
    "Ic_light_coh": "Intensity",
    "Ic_light_2mom": "Intensity",
    "IVAR_ALPHA": "Intensity",
    "Fo_light_corr": "SFAmplitude",
    "SIGFo_light_corr": "Stddev",
    "DF_corr": "SFAmplitude",
    "SIGDF_corr": "Stddev",
    "2mDFop-DFc_corr": "SFAmplitude",
    "mDFop-DFc_corr": "SFAmplitude",
    "DDF": "SFAmplitude",
    "W_2MOM": "Weight",
}

FRACTION = 0.25
LAMBDA_TWIN = 0.2


@pytest.fixture(scope="module")
def cli_script(project_root):
    script = project_root / "torchref" / "cli" / "collection_difference_refine.py"
    if not script.exists():
        pytest.skip("difference-refine CLI not found")
    return script


@pytest.fixture(scope="module")
def intensity_pair(mtz_dir, pdb_dir, tmp_path_factory):
    """A dark/light pair carrying I/SIGI, from the only fixture that has them.

    1DAW is the sole reflection file under ``tests/files`` with intensity columns; 3GR5,
    which the other difference-refine CLI test uses, has none and so cannot exercise an
    intensity-space path at all.
    """
    import torch

    from torchref import ReflectionData

    mtz = mtz_dir / "1DAW.mtz"
    pdb = pdb_dir / "1DAW.pdb"
    if not (mtz.exists() and pdb.exists()):
        pytest.skip("1DAW fixture not present")

    data = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    if data.I is None:
        pytest.skip("1DAW loaded without intensities")

    out = tmp_path_factory.mktemp("two_moment_cli")
    n = len(data)
    idx = torch.arange(n)
    # Slightly different reflection sets, as a real dark/light pair would be.
    data.__select__(idx < int(n * 0.97)).write_mtz(str(out / "dark.mtz"))
    data.__select__(idx >= int(n * 0.03)).write_mtz(str(out / "light.mtz"))
    return {"dir": out, "pdb": pdb}


def _run(cli_script, pair, outdir, *extra):
    env = dict(os.environ)
    # The installed torchref may point at a different checkout; make the subprocess
    # import the tree under test.
    root = str(cli_script.parents[2])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, str(cli_script),
        "-dm", str(pair["pdb"]), "-lm", str(pair["pdb"]),
        "-dsf", str(pair["dir"] / "dark.mtz"),
        "-lsf", str(pair["dir"] / "light.mtz"),
        "--fraction", str(FRACTION),
        "--n-cycles", "1", "--n-steps", "1", "--max-iter", "3",
        "--dmin", "2.2", "-o", str(outdir),
        "--device", "cpu", "--verbose", "0",
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    assert proc.returncode == 0, (
        f"CLI failed ({proc.returncode})\nstderr tail:\n{proc.stderr[-3000:]}"
    )
    prefix = f"fractions_{round((1 - FRACTION) * 100)}_{round(FRACTION * 100)}_"
    return outdir / f"{prefix}difference_data.mtz", outdir / f"{prefix}summary.json"


@pytest.fixture(scope="module")
def baseline_mtz(cli_script, intensity_pair, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("baseline")
    return _run(cli_script, intensity_pair, outdir)


@pytest.fixture(scope="module")
def two_moment_mtz(cli_script, intensity_pair, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("two_moment")
    return _run(
        cli_script, intensity_pair, outdir,
        "--two-moment", "--lambda-twin", str(LAMBDA_TWIN),
    )


def _read(path):
    import reciprocalspaceship as rs

    return rs.read_mtz(str(path))


class TestBaselineLayoutIsUnchanged:
    def test_baseline_columns_are_exactly_the_expected_set(self, baseline_mtz):
        mtz, _ = baseline_mtz
        assert set(_read(mtz).columns) == BASELINE_COLUMNS

    def test_no_two_moment_columns_without_the_flag(self, baseline_mtz):
        mtz, _ = baseline_mtz
        present = set(_read(mtz).columns) & set(TWO_MOMENT_COLUMNS)
        assert present == set(), f"unexpected two-moment columns: {sorted(present)}"


class TestTwoMomentLayout:
    def test_baseline_columns_all_survive(self, two_moment_mtz):
        mtz, _ = two_moment_mtz
        assert BASELINE_COLUMNS.issubset(set(_read(mtz).columns))

    def test_every_new_column_is_present_with_the_right_mtz_type(self, two_moment_mtz):
        mtz, _ = two_moment_mtz
        df = _read(mtz)
        for name, expected in TWO_MOMENT_COLUMNS.items():
            assert name in df.columns, f"missing column {name}"
            actual = df.dtypes[name].name
            assert actual == expected, (
                f"{name} written as {actual}, expected {expected} -- this writer has no "
                f"infer_mtz_dtypes() safety net"
            )

    def test_the_column_set_is_exactly_baseline_plus_the_new_ones(self, two_moment_mtz):
        mtz, _ = two_moment_mtz
        assert set(_read(mtz).columns) == BASELINE_COLUMNS | set(TWO_MOMENT_COLUMNS)


class TestTwoMomentValuesAreConsistent:
    def test_ivar_alpha_is_sigma_sq_times_the_squared_difference(self, two_moment_mtz):
        """The variance column must be the quantity it claims, not a rescaling of it."""
        import numpy as np

        mtz, summary = two_moment_mtz
        df = _read(mtz)
        results = json.loads(summary.read_text())["results"]

        sigma_sq = results["sigma_alpha_sq"]
        dfc = df["DFc_complex"].to_numpy().astype(float)
        ivar = df["IVAR_ALPHA"].to_numpy().astype(float)

        expected = sigma_sq * dfc**2
        scale = max(float(np.abs(expected).max()), 1e-30)
        assert np.abs(ivar - expected).max() / scale < 1e-5

    def test_the_two_moment_intensity_exceeds_the_coherent_one_by_the_variance(
        self, two_moment_mtz
    ):
        import numpy as np

        mtz, _ = two_moment_mtz
        df = _read(mtz)
        coh = df["Ic_light_coh"].to_numpy().astype(float)
        two = df["Ic_light_2mom"].to_numpy().astype(float)
        ivar = df["IVAR_ALPHA"].to_numpy().astype(float)

        scale = max(float(np.abs(ivar).max()), 1e-30)
        assert np.abs((two - coh) - ivar).max() / scale < 1e-3
        # The variance term has no sign: it can only add.
        assert (two >= coh - 1e-6).all()

    def test_the_weight_lies_in_zero_to_one_and_bites(self, two_moment_mtz):
        w = _read(two_moment_mtz[0])["W_2MOM"].to_numpy().astype(float)
        assert (w > 0).all() and (w <= 1.0 + 1e-6).all()
        assert w.min() < 0.99, (
            "W_2MOM is 1 everywhere, so the correction is doing nothing here and this "
            "fixture cannot detect a change in it"
        )

    def test_the_correction_moves_the_difference_amplitudes(self, two_moment_mtz):
        """DDF is the diagnostic; if it were identically zero the whole column set
        would be decorative."""
        import numpy as np

        df = _read(two_moment_mtz[0])
        ddf = df["DDF"].to_numpy().astype(float)
        assert np.count_nonzero(ddf) > 0.5 * len(ddf)
        # Subtracting a positive contamination lowers the light amplitude on average.
        assert ddf.mean() < 0.0

    def test_summary_reports_the_activation_moments(self, two_moment_mtz):
        _, summary = two_moment_mtz
        results = json.loads(summary.read_text())["results"]
        for key in ("alpha_mean", "lambda_twin", "sigma_alpha_sq"):
            assert key in results, f"missing summary key: {key}"
        assert results["lambda_twin"] == pytest.approx(LAMBDA_TWIN, abs=1e-5)
        assert results["alpha_mean"] == pytest.approx(FRACTION, abs=1e-5)
        assert results["sigma_alpha_sq"] == pytest.approx(
            FRACTION * (1 - FRACTION) * LAMBDA_TWIN, rel=1e-4
        )
