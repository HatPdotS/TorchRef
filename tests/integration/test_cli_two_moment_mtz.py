"""The difference-refinement MTZ layout, pinned.

Nothing else in the suite asserts on these column names, and they go into files that get
deposited, so the layout is pinned here deliberately: this file is expected to move in
lockstep with a change to the writer, and to fail loudly if one happens by accident.

Three layouts are checked. The default is the map a reader wants and can identify --
``DELFWT``/``PHDELWT``, the weighted difference on dark phases, plus the extrapolated
map. ``--two-moment`` adds the activation-heterogeneity correction. ``--all-columns``
adds the alternative constructions of both, which are informative once you know which is
which and misleading before then.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# The default set, with a light model supplied (which the refinement CLI always does).
# H/K/L are the index, so they are not among ``.columns``.
DEFAULT_COLUMNS = {
    "Fo_dark": "SFAmplitude", "SIGFo_dark": "Stddev",
    "Fo_light": "SFAmplitude", "SIGFo_light": "Stddev",
    "DF": "SFAmplitude", "SIGDF": "Stddev",
    # The difference map. CCP4/Coot open these by name.
    "DELFWT": "SFAmplitude", "PHDELWT": "Phase",
    "Fc_dark": "SFAmplitude",
    # The mixed model, and the extrapolated map to refine against.
    "FC": "SFAmplitude", "PHIC": "Phase",
    "FEXT": "SFAmplitude", "SIGFEXT": "Stddev",
    "FWT": "SFAmplitude", "PHWT": "Phase",
}
FLAG_COLUMNS = {"FreeR_flag_dark", "FreeR_flag_light"}

TWO_MOMENT_COLUMNS = {
    "DELFWT_corr": "SFAmplitude",
    "Fo_light_corr": "SFAmplitude", "SIGFo_light_corr": "Stddev",
    "DF_corr": "SFAmplitude", "SIGDF_corr": "Stddev",
    "DDF": "SFAmplitude",
}

# What ``--all-columns`` adds on top, given a light model.
ALL_COLUMNS_EXTRA = {
    "2mDFop-DFc": "SFAmplitude", "mDFop-DFc": "SFAmplitude",
    "PHIC_diff": "Phase",
    "DFc": "SFAmplitude", "DFc_phased": "SFAmplitude",
    "FEXT_PHASED": "SFAmplitude", "SIGFEXT_PHASED": "Stddev",
    "2FEXT_PHASED-Fc": "SFAmplitude", "FEXT_PHASED-Fc": "SFAmplitude",
    "PHFEXT_PHASED": "Phase",
    "FEXT_SCALAR": "SFAmplitude", "SIGFEXT_SCALAR": "Stddev",
    "2FEXT_SCALAR-Fc": "SFAmplitude", "FEXT_SCALAR-Fc": "SFAmplitude",
    "PHFEXT_SCALAR": "Phase",
}

# And what it adds again once the two-moment model is on.
ALL_COLUMNS_TWO_MOMENT_EXTRA = {
    "Io_light": "Intensity", "SIGIo_light": "Stddev",
    "Ic_light_coh": "Intensity", "Ic_light_2mom": "Intensity",
    "IVAR_ALPHA": "Intensity", "W_2MOM": "Weight",
    "2mDFop-DFc_corr": "SFAmplitude", "mDFop-DFc_corr": "SFAmplitude",
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


@pytest.fixture(scope="module")
def two_moment_all_mtz(cli_script, intensity_pair, tmp_path_factory):
    """Everything on. The value-consistency tests below need the diagnostic columns,
    which is exactly what ``--all-columns`` is for."""
    outdir = tmp_path_factory.mktemp("two_moment_all")
    return _run(
        cli_script, intensity_pair, outdir,
        "--two-moment", "--lambda-twin", str(LAMBDA_TWIN), "--all-columns",
    )


def _read(path):
    import reciprocalspaceship as rs

    return rs.read_mtz(str(path))


class TestDefaultLayout:
    def test_default_columns_are_exactly_the_expected_set(self, baseline_mtz):
        mtz, _ = baseline_mtz
        assert set(_read(mtz).columns) == set(DEFAULT_COLUMNS) | FLAG_COLUMNS

    def test_every_default_column_carries_the_right_mtz_type(self, baseline_mtz):
        mtz, _ = baseline_mtz
        df = _read(mtz)
        for name, expected in DEFAULT_COLUMNS.items():
            assert name in df.columns, f"missing column {name}"
            assert df.dtypes[name].name == expected, (
                f"{name} written as {df.dtypes[name].name}, expected {expected}"
            )

    def test_no_two_moment_columns_without_the_flag(self, baseline_mtz):
        mtz, _ = baseline_mtz
        present = set(_read(mtz).columns) & set(TWO_MOMENT_COLUMNS)
        assert present == set(), f"unexpected two-moment columns: {sorted(present)}"

    def test_no_gated_columns_without_all_columns(self, baseline_mtz):
        mtz, _ = baseline_mtz
        present = set(_read(mtz).columns) & set(ALL_COLUMNS_EXTRA)
        assert present == set(), f"unexpected gated columns: {sorted(present)}"

    def test_the_difference_map_is_the_weighted_difference_on_dark_phases(
        self, baseline_mtz
    ):
        """``DELFWT`` must be ``(Fo_light - Fo_dark) * w`` with ``w`` the mean-normalised
        inverse variance -- the construction ``torchref.validate-ded`` correlates
        against. If these two ever diverge, the map in the file stops being the map the
        validation reports on, which is how the output drifted from the science before.
        """
        import numpy as np

        df = _read(baseline_mtz[0])
        dfo = (df["Fo_light"].to_numpy().astype(float)
               - df["Fo_dark"].to_numpy().astype(float))
        sig = np.sqrt(df["SIGFo_dark"].to_numpy().astype(float) ** 2
                      + df["SIGFo_light"].to_numpy().astype(float) ** 2)
        w = 1 / sig**2
        w = w / w.mean()

        expected = dfo * w
        got = df["DELFWT"].to_numpy().astype(float)
        scale = max(float(np.abs(expected).max()), 1e-30)
        assert np.abs(got - expected).max() / scale < 1e-5

        # And the phase is the dark model's, not the mixed model's.
        assert not np.allclose(
            df["PHDELWT"].to_numpy().astype(float),
            df["PHIC"].to_numpy().astype(float),
        )


class TestTwoMomentLayout:
    def test_default_columns_all_survive(self, two_moment_mtz):
        mtz, _ = two_moment_mtz
        assert set(DEFAULT_COLUMNS).issubset(set(_read(mtz).columns))

    def test_every_new_column_is_present_with_the_right_mtz_type(self, two_moment_mtz):
        mtz, _ = two_moment_mtz
        df = _read(mtz)
        for name, expected in TWO_MOMENT_COLUMNS.items():
            assert name in df.columns, f"missing column {name}"
            actual = df.dtypes[name].name
            assert actual == expected, (
                f"{name} written as {actual}, expected {expected}"
            )

    def test_the_column_set_is_exactly_default_plus_the_new_ones(self, two_moment_mtz):
        mtz, _ = two_moment_mtz
        assert set(_read(mtz).columns) == (
            set(DEFAULT_COLUMNS) | FLAG_COLUMNS | set(TWO_MOMENT_COLUMNS)
        )

    def test_the_corrected_difference_map_pairs_with_the_same_phases(
        self, two_moment_mtz
    ):
        """``DELFWT_corr`` is the corrected difference on the *same* dark phases, so it
        is opened against ``PHDELWT`` and must be built the same way as ``DELFWT``."""
        import numpy as np

        df = _read(two_moment_mtz[0])
        sig = np.sqrt(df["SIGFo_dark"].to_numpy().astype(float) ** 2
                      + df["SIGFo_light"].to_numpy().astype(float) ** 2)
        w = 1 / sig**2
        w = w / w.mean()

        expected = df["DF_corr"].to_numpy().astype(float) * w
        got = df["DELFWT_corr"].to_numpy().astype(float)
        scale = max(float(np.abs(expected).max()), 1e-30)
        assert np.abs(got - expected).max() / scale < 1e-5


class TestAllColumns:
    def test_all_columns_is_a_strict_superset(self, two_moment_mtz, two_moment_all_mtz):
        default = set(_read(two_moment_mtz[0]).columns)
        full = set(_read(two_moment_all_mtz[0]).columns)
        assert default < full, "--all-columns must add columns, never remove any"

    def test_the_gated_columns_are_exactly_the_expected_ones(self, two_moment_all_mtz):
        df = _read(two_moment_all_mtz[0])
        assert set(df.columns) == (
            set(DEFAULT_COLUMNS) | FLAG_COLUMNS | set(TWO_MOMENT_COLUMNS)
            | set(ALL_COLUMNS_EXTRA) | set(ALL_COLUMNS_TWO_MOMENT_EXTRA)
        )

    def test_every_gated_column_carries_the_right_mtz_type(self, two_moment_all_mtz):
        df = _read(two_moment_all_mtz[0])
        expected_types = {**ALL_COLUMNS_EXTRA, **ALL_COLUMNS_TWO_MOMENT_EXTRA}
        for name, expected in expected_types.items():
            assert name in df.columns, f"missing column {name}"
            assert df.dtypes[name].name == expected, (
                f"{name} written as {df.dtypes[name].name}, expected {expected}"
            )

    def test_no_column_escapes_with_a_plain_numpy_dtype(self, two_moment_all_mtz):
        """Every layer declares its columns' MTZ types beside the values, and the writer
        refuses a column with none. This is the end-to-end version of that check: a
        column reaching the file as a bare numpy dtype is the failure the old parallel
        name lists invited.
        """
        df = _read(two_moment_all_mtz[0])
        bare = [c for c in df.columns if not hasattr(df.dtypes[c], "mtztype")]
        assert bare == [], f"columns written without an MTZ dtype: {bare}"


class TestTwoMomentValuesAreConsistent:
    def test_ivar_alpha_is_sigma_sq_times_the_squared_difference(
        self, two_moment_all_mtz
    ):
        """The variance column must be the quantity it claims, not a rescaling of it."""
        import numpy as np

        mtz, summary = two_moment_all_mtz
        df = _read(mtz)
        results = json.loads(summary.read_text())["results"]

        sigma_sq = results["sigma_alpha_sq"]
        dfc = df["DFc_phased"].to_numpy().astype(float)
        ivar = df["IVAR_ALPHA"].to_numpy().astype(float)

        expected = sigma_sq * dfc**2
        scale = max(float(np.abs(expected).max()), 1e-30)
        assert np.abs(ivar - expected).max() / scale < 1e-5

    def test_the_two_moment_intensity_exceeds_the_coherent_one_by_the_variance(
        self, two_moment_all_mtz
    ):
        """``Ic_2mom - Ic_coh`` must equal ``IVAR_ALPHA``, to whatever precision float32
        leaves after the cancellation.

        This is a catastrophic-cancellation case, and the tolerance is computed rather
        than guessed. The variance term is ~2.6e-6 of the intensity on this fixture,
        while float32 resolves ~1.2e-7 of it -- so only about one significant digit of
        the difference survives, and any fixed tolerance would either pass vacuously or
        fail for reasons that have nothing to do with the code.

        The target itself never forms this difference (it computes
        ``|F|**2 + sigma**2 |dF|**2`` directly), so the loss is unaffected; it is
        recovering the variance term from the two published columns that is lossy.
        """
        import numpy as np

        df = _read(two_moment_all_mtz[0])
        coh = df["Ic_light_coh"].to_numpy().astype(float)
        two = df["Ic_light_2mom"].to_numpy().astype(float)
        ivar = df["IVAR_ALPHA"].to_numpy().astype(float)

        # Absolute error float32 can leave in the difference of two intensities.
        eps32 = float(np.finfo(np.float32).eps)
        floor = eps32 * np.maximum(np.abs(coh), np.abs(two))
        residual = np.abs((two - coh) - ivar)

        assert (residual <= 4.0 * floor + 1e-12).all(), (
            f"recovered variance term differs from IVAR_ALPHA by more than float32 "
            f"cancellation allows: worst {np.max(residual / (floor + 1e-30)):.1f} ulp"
        )
        # The variance term has no sign: it can only add.
        assert (two >= coh - 4.0 * floor).all()

    def test_the_weight_is_the_contamination_ratio(self, two_moment_all_mtz):
        """``W_2MOM`` must be ``sigma_I**2 / (sigma_I**2 + IVAR_ALPHA)``.

        Asserted as the formula rather than as a magnitude. On this fixture the weight
        never falls below ~0.9998, because the contamination is ~1e-3 of a single
        reflection's sigma -- which is the real behaviour of this correction, not a
        defect: it is a systematic that adds coherently over the whole dataset while
        being invisible on any one reflection. A test demanding visible down-weighting
        would be asserting the physics is different from what it is.
        """
        import numpy as np

        df = _read(two_moment_all_mtz[0])
        w = df["W_2MOM"].to_numpy().astype(float)
        sig = df["SIGIo_light"].to_numpy().astype(float)
        ivar = df["IVAR_ALPHA"].to_numpy().astype(float)

        assert (w > 0).all() and (w <= 1.0 + 1e-6).all()
        expected = sig**2 / np.maximum(sig**2 + ivar, 1e-12)
        assert np.allclose(w, expected, rtol=1e-5, atol=1e-7)
        # Anti-vacuity for the formula: the contamination must not be identically zero,
        # or the ratio above is trivially 1 and proves nothing.
        assert (ivar > 0).any()

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
            assert key in results, f"summary is missing {key}"
        assert results["lambda_twin"] == pytest.approx(LAMBDA_TWIN)

    def test_summary_reports_the_shrinkage_diagnostics(self, two_moment_mtz):
        """``tau_sq`` and mean ``w(h)`` say whether the default extrapolated map is
        over-shrunk, so they belong in the summary rather than only in a print."""
        _, summary = two_moment_mtz
        results = json.loads(summary.read_text())["results"]
        assert "tau_sq" in results and "w_shrinkage_mean" in results
        assert 0.0 < results["w_shrinkage_mean"] <= 1.0
