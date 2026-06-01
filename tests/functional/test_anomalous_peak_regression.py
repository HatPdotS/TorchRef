"""Regression test for anomalous (Bijvoet) difference-map peak heights.

Uses 7L84 (sulfur-SAD thermolysin-like lysozyme; P4_3 2_1 2, 1.70 A, data
collected at lambda = 1.892 A). The anomalous-difference Fourier (ANOM/PANOM,
built from the deposited model's calculated phases and the observed Bijvoet
differences |F(+)| - |F(-)|) is FFT'd to a normalized map and sampled at every
sulfur position. The strongest sulfur peak is the regression metric: a drop
signals a regression in the anomalous structure-factor or map-coefficient path.

Input handling: ``rs.read_cif`` preserves the deposited I(+)/I(-), and
``stack_anomalous`` expands the Bijvoet mates onto separate rows so TorchRef can
derive ``friedel_flags`` on load. Loading the SF-CIF *directly* averages the
mates away -- exercised (and expected-failing) by the xfail test below.
"""

import numpy as np
import pytest

# Baseline recorded on the anomalous-refinement branch (CPU, deterministic
# across repeated runs): the ten sulfur sites span ~13.8 - 19.9 sigma in the
# normalized ANOM map, with the maximum at A/CYS115. For reference, phenix.refine
# on the same data yields 20.4 sigma at the same site; torchref reproduces that
# to ~2% once single-mate reflections are excluded from the Bijvoet difference
# (see _build_anomalous_dataframe). We guard the *maximum* peak against
# regression with a one-sided floor; a stronger peak is fine.
BASELINE_MAX_SIGMA = 19.93
REGRESSION_FLOOR = BASELINE_MAX_SIGMA - 1.0  # ~5% tolerance for numerical drift
PER_SITE_FLOOR = 10.0  # every sulfur should carry a clear anomalous peak
WAVELENGTH = 1.892
N_SULFUR = 10


def _measure_sulfur_peaks(out_mtz, pdb_path):
    """Return {site_label: peak_sigma} for every sulfur in the model."""
    import gemmi

    mtz = gemmi.read_mtz_file(str(out_mtz))
    grid = mtz.transform_f_phi_to_map("ANOM", "PANOM", sample_rate=3.0)
    grid.normalize()  # express the map in sigma units

    def peak(pos, box=1.0, n=11):
        offsets = np.linspace(-box, box, n)
        best = -np.inf
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    v = grid.interpolate_value(
                        gemmi.Position(pos.x + dx, pos.y + dy, pos.z + dz)
                    )
                    if np.isfinite(v) and v > best:
                        best = v
        return best

    st = gemmi.read_structure(str(pdb_path))
    peaks = {}
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element.name == "S":
                    label = f"{chain.name}/{res.name}{res.seqid.num}"
                    peaks[label] = peak(atom.pos)
    return peaks


def _build_anomalous_mtz(data_file, pdb_path, out_mtz):
    """Scale the deposited model against the data and write a phenix-style
    anomalous MTZ (ANOM/PANOM). No refinement -- the map is deterministic."""
    from torchref import LBFGSRefinement

    ref = LBFGSRefinement(
        data_file=str(data_file),
        pdb=str(pdb_path),
        wavelength=WAVELENGTH,
        verbose=0,
    )
    ref.get_scales()
    ref.write_out_mtz(str(out_mtz), anomalous=True)
    return out_mtz


@pytest.fixture(scope="module")
def stacked_7l84_mtz(cif_sf_dir, tmp_path_factory):
    """7L84 SF-CIF -> stack_anomalous -> MTZ (Bijvoet mates on separate rows)."""
    import reciprocalspaceship as rs

    out = tmp_path_factory.mktemp("anom7l84") / "7l84_stacked.mtz"
    ds = rs.read_cif(str(cif_sf_dir / "7L84-sf.cif")).stack_anomalous()
    ds.write_mtz(str(out))
    return out


@pytest.mark.integration
def test_7l84_sulfur_anomalous_peak_height(stacked_7l84_mtz, pdb_dir, tmp_path):
    pdb_path = pdb_dir / "7L84.pdb"
    out_mtz = _build_anomalous_mtz(
        stacked_7l84_mtz, pdb_path, tmp_path / "7l84_anom.mtz"
    )
    peaks = _measure_sulfur_peaks(out_mtz, pdb_path)

    assert len(peaks) == N_SULFUR, f"expected {N_SULFUR} sulfur sites, got {peaks}"
    # Each sulfur should carry a clear positive anomalous peak.
    assert all(v > PER_SITE_FLOOR for v in peaks.values()), peaks

    max_peak = max(peaks.values())
    assert max_peak >= REGRESSION_FLOOR, (
        f"anomalous peak height regressed: max={max_peak:.2f} sigma "
        f"(baseline {BASELINE_MAX_SIGMA:.2f}, floor {REGRESSION_FLOOR:.2f}); "
        f"per-site sigma: {peaks}"
    )


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "ReflectionCIFReader averages Bijvoet pairs (F+/F-, I+/I- -> mean) on "
        "load, so friedel_flags come back all-False and the anomalous signal is "
        "lost. Loading an SF-CIF directly should preserve the (+)/(-) members "
        "the way rs.read_cif + stack_anomalous does. Remove the xfail once the "
        "reader gains an anomalous (non-merging) mode."
    ),
    strict=True,
)
def test_7l84_load_cif_direct_preserves_anomalous(cif_sf_dir, pdb_dir, tmp_path):
    """Loading the SF-CIF directly should reproduce the sulfur anomalous peaks."""
    pdb_path = pdb_dir / "7L84.pdb"
    out_mtz = _build_anomalous_mtz(
        cif_sf_dir / "7L84-sf.cif", pdb_path, tmp_path / "7l84_cifdirect_anom.mtz"
    )
    peaks = _measure_sulfur_peaks(out_mtz, pdb_path)
    assert peaks, "no sulfur sites measured"
    assert max(peaks.values()) >= REGRESSION_FLOOR
