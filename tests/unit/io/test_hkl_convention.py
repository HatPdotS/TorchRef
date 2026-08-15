"""Structure factors and written phases live in the canonical CCP4 convention.

``ReflectionData`` evaluates the model at a signed index so Bijvoet mates get
distinct ``|F_calc|``, but expresses the result on the canonical ASU index that
is written as H,K,L. Pairing a signed-convention phase with a canonical index
negates it, which is invisible on data already inside the CCP4 ASU -- every MTZ
under ``tests/files/`` is -- and wrong for roughly half the reflections on data
that is not.

The invariance test below manufactures the second case by negating the Miller
indices, and asserts that precondition so it cannot quietly degenerate into the
first.
"""

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs
import torch

from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT

CODE = "3GR5"  # P 65 2 2 -- the smallest real fixture, and non-centrosymmetric


def _rebuild(hkl, src):
    """``src`` re-expressed on ``hkl``, declared merged so the legacy layout is used."""
    return ReflectionData.from_tensors(
        hkl,
        src.F,
        src.F_sigma,
        src.cell,
        src.spacegroup,
        rfree_flags=src.rfree_flags,
        verbose=0,
        friedel_merged=True,
    )


@pytest.fixture(scope="module")
def canonical_and_negated(mtz_dir):
    """The same reflections indexed two ways: inside the CCP4 ASU, and negated."""
    src = ReflectionData(verbose=0)
    src.load_mtz(str(mtz_dir / f"{CODE}.mtz"))
    return _rebuild(src.hkl, src), _rebuild(-src.hkl, src)


@pytest.fixture
def anomalous_data(mtz_dir, tmp_path):
    """1DAW stacked into explicit (+/-) Bijvoet rows, so mates really are paired."""
    out = tmp_path / "anom_in.mtz"
    rs.read_mtz(str(mtz_dir / "1DAW.mtz")).stack_anomalous().write_mtz(str(out))
    data = ReflectionData(verbose=0)
    data.load_mtz(str(out))
    return data


def _model(pdb_dir, data):
    m = ModelFT(verbose=0, max_res=2.0)
    m.load_pdb(str(pdb_dir / f"{CODE}.pdb"))
    m.cell, m.spacegroup = data.cell, data.spacegroup
    return m


def _write(data, pdb_dir, path):
    with torch.no_grad():
        fcalc = data.structure_factors(_model(pdb_dir, data), cached=False)
    data.write_mtz(str(path), fcalc=fcalc, anomalous=False)
    return rs.read_mtz(str(path)).reset_index()


def _circular_diff(a, b):
    """Absolute difference of two phase arrays in degrees, wrapped to [0, 180]."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


class TestConjugateFriedel:
    def test_is_an_involution(self, canonical_and_negated):
        _, neg = canonical_and_negated
        z = torch.randn(len(neg.hkl), dtype=torch.complex64, device=neg.hkl.device)
        assert torch.equal(neg.conjugate_friedel(neg.conjugate_friedel(z)), z)

    def test_touches_exactly_the_flagged_rows(self, canonical_and_negated):
        _, neg = canonical_and_negated
        flag = neg.friedel_flags
        z = torch.randn(len(neg.hkl), dtype=torch.complex64, device=neg.hkl.device)
        out = neg.conjugate_friedel(z)
        assert torch.equal(out[flag], z[flag].conj())
        assert torch.equal(out[~flag], z[~flag])

    def test_is_identity_without_flagged_rows(self, canonical_and_negated):
        ref, _ = canonical_and_negated
        z = torch.randn(len(ref.hkl), dtype=torch.complex64, device=ref.hkl.device)
        assert torch.equal(ref.conjugate_friedel(z), z)


class TestWrittenPhaseConvention:
    def test_preconditions(self, canonical_and_negated):
        """Without these the invariance test below proves nothing."""
        ref, neg = canonical_and_negated
        assert not bool(ref.friedel_flags.any()), "reference must be inside the ASU"
        assert float(neg.friedel_flags.float().mean()) > 0.4, (
            "negated indices must force canonicalisation to conjugate"
        )
        # Both must land on the same canonical reflections, in the same order,
        # or the column-by-column comparison is meaningless.
        assert torch.equal(ref.hkl, neg.hkl)
        assert torch.equal(ref.F, neg.F)

    @pytest.mark.parametrize("column", ["PH-model", "PHWT", "PHDELWT"])
    def test_phases_do_not_depend_on_input_indexing(
        self, canonical_and_negated, pdb_dir, tmp_path, column
    ):
        ref, neg = canonical_and_negated
        a = _write(ref, pdb_dir, tmp_path / "ref.mtz")
        b = _write(neg, pdb_dir, tmp_path / "neg.mtz")

        assert np.array_equal(a[["H", "K", "L"]].to_numpy(), b[["H", "K", "L"]].to_numpy())

        # Weak reflections have numerically unstable phases and the FFT is not
        # bit-reproducible run to run, so compare where there is real signal.
        strong = a["F-model"].to_numpy() > np.median(a["F-model"].to_numpy())
        diff = _circular_diff(a[column].to_numpy()[strong], b[column].to_numpy()[strong])
        # A sign flip on ~half the rows shows up as ~2*|phase|, i.e. tens of degrees.
        assert diff.max() < 5.0, f"{column}: max {diff.max():.2f} deg"

    def test_written_phase_matches_gemmi(self, canonical_and_negated, pdb_dir, tmp_path):
        """Absolute check against an independent structure-factor calculation."""
        _, neg = canonical_and_negated
        out = _write(neg, pdb_dir, tmp_path / "abs.mtz")

        st = gemmi.read_structure(str(pdb_dir / f"{CODE}.pdb"))
        st.setup_entities()
        st.remove_hydrogens()
        calc = gemmi.StructureFactorCalculatorX(st.cell)

        rng = np.random.default_rng(0)
        rows = out.iloc[rng.choice(len(out), 40, replace=False)]
        checked = 0
        for _, row in rows.iterrows():
            ref_sf = calc.calculate_sf_from_model(
                st[0], (int(row["H"]), int(row["K"]), int(row["L"]))
            )
            if abs(ref_sf) < 1.0:  # phase is meaningless at zero amplitude
                continue
            expected = np.degrees(np.angle(complex(ref_sf.real, ref_sf.imag)))
            assert _circular_diff(row["PH-model"], expected) < 8.0
            checked += 1
        assert checked > 20, "too few usable reflections to be a real check"


class TestFriedelMergedInference:
    """Needing conjugation to reach the ASU is not the same as carrying mates."""

    def test_already_canonical_is_merged(self, mtz_dir):
        d = ReflectionData(verbose=0)
        d.load_mtz(str(mtz_dir / f"{CODE}.mtz"))
        assert not bool(d.friedel_flags.any())
        assert d.friedel_merged is True

    def test_reindexed_without_mates_stays_merged(self, mtz_dir):
        """A merged dataset indexed in another convention flags rows but has no
        mates, so f'' must stay off and the legacy layout must stay selected."""
        src = ReflectionData(verbose=0)
        src.load_mtz(str(mtz_dir / f"{CODE}.mtz"))
        # No explicit friedel_merged: the inference is what is under test.
        neg = ReflectionData.from_tensors(
            -src.hkl, src.F, src.F_sigma, src.cell, src.spacegroup,
            rfree_flags=src.rfree_flags, verbose=0,
        )
        assert float(neg.friedel_flags.float().mean()) > 0.4
        assert neg.friedel_merged is True

    def test_real_bijvoet_pairs_are_unmerged(self, anomalous_data):
        assert bool(anomalous_data.friedel_flags.any())
        assert anomalous_data.friedel_merged is False


class TestAnomalousLayoutUnaffected:
    """Canonicalising early must not flatten the Bijvoet difference."""

    def test_mates_keep_distinct_amplitudes(self, anomalous_data, pdb_dir):
        model = ModelFT(verbose=0, max_res=2.0, wavelength=1.54, apply_bijvoet=True)
        model.load_pdb(str(pdb_dir / "1DAW.pdb"))
        with torch.no_grad():
            fcalc = anomalous_data.structure_factors(model)

        flag = anomalous_data.friedel_flags
        assert bool(flag.any()), "fixture must carry Friedel-flagged rows"
        inverse, n_groups = anomalous_data.asu_group_indices()
        paired = anomalous_data._group_any(
            flag, inverse, n_groups
        ) & anomalous_data._group_any(~flag, inverse, n_groups)
        assert bool(paired.any()), "fixture must contain real Bijvoet pairs"

        # |conj(z)| == |z|, so the anomalous amplitude difference must survive.
        amp = torch.abs(fcalc)
        spread = torch.zeros(n_groups, device=amp.device)
        spread.index_add_(0, inverse[flag], amp[flag])
        spread.index_add_(0, inverse[~flag], -amp[~flag])
        assert float(spread[paired].abs().max()) > 1e-3
