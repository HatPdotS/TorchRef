"""Tests for the ML alpha/beta (sigma_A) estimator and its Scaler integration.

- estimate_alpha_beta recovers a known per-shell sigma_A / beta (Phenix port)
- epsilon_from_hkl is sane
- the base Scaler exposes get_alpha_beta / reset_alpha_beta_cache and caches lazily
"""
import pytest
import torch

from torchref.base.targets.xray_ml_sigmaa import (
    estimate_alpha_beta,
    epsilon_from_hkl,
)


@pytest.mark.unit
class TestEstimateAlphaBeta:
    def _synthetic(self, shells=8, per=2000, seed=0):
        g = torch.Generator().manual_seed(seed)
        dt = torch.float64
        sA = torch.linspace(0.97, 0.55, shells, dtype=dt)
        SigN = torch.linspace(2000.0, 150.0, shells, dtype=dt)
        FO, FC, DSS, SA, BT = [], [], [], [], []
        for i in range(shells):
            fcc = torch.sqrt(SigN[i] / 2) * (
                torch.randn(per, generator=g, dtype=dt)
                + 1j * torch.randn(per, generator=g, dtype=dt)
            )
            var = (1 - sA[i] ** 2) * SigN[i]
            eoc = sA[i] * fcc + torch.sqrt(var / 2) * (
                torch.randn(per, generator=g, dtype=dt)
                + 1j * torch.randn(per, generator=g, dtype=dt)
            )
            FO.append(eoc.abs()); FC.append(fcc.abs())
            DSS.append(torch.full((per,), 0.02 + 0.30 * i / shells, dtype=dt))
            SA.append(sA[i]); BT.append(var)
        return (torch.cat(FO), torch.cat(FC), torch.cat(DSS),
                torch.stack(SA), torch.stack(BT))

    def test_recovers_known_sigma_a(self):
        FO, FC, DSS, sA_true, beta_true = self._synthetic()
        cen = torch.zeros_like(FO, dtype=torch.bool)
        eps = torch.ones_like(FO)
        free = torch.ones_like(FO, dtype=torch.bool)
        alpha, beta, abin, bbin, bdss = estimate_alpha_beta(
            FO, FC, cen, eps, DSS, free, per_bin=1000
        )
        # per-reflection alpha tracks the per-shell truth (monotone decreasing,
        # right magnitude). Tolerance allows mild smoothing/finite-sample bias.
        assert alpha.shape == FO.shape
        assert (alpha > 0).all() and (alpha < 1.05).all()
        # correlation between estimated and true sigma_A across reflections
        sA_per_refl = sA_true.repeat_interleave(FO.numel() // sA_true.numel())
        cc = torch.corrcoef(torch.stack([alpha, sA_per_refl]))[0, 1]
        assert cc > 0.95
        # mid-range accuracy
        mid = (sA_per_refl > 0.6) & (sA_per_refl < 0.95)
        assert (alpha[mid] - sA_per_refl[mid]).abs().mean() < 0.06

    def test_degenerate_free_set(self):
        n = 100
        FO = torch.rand(n, dtype=torch.float64) * 50 + 1
        FC = torch.rand(n, dtype=torch.float64) * 50 + 1
        cen = torch.zeros(n, dtype=torch.bool)
        eps = torch.ones(n, dtype=torch.float64)
        dss = torch.rand(n, dtype=torch.float64) * 0.3 + 0.02
        free = torch.zeros(n, dtype=torch.bool)  # no free reflections
        alpha, beta, *_ = estimate_alpha_beta(FO, FC, cen, eps, dss, free)
        assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
        assert (beta > 0).all()


@pytest.mark.unit
class TestEpsilonFromHkl:
    def test_none_spacegroup_returns_ones(self):
        hkl = torch.randint(-5, 6, (50, 3))
        eps = epsilon_from_hkl(hkl, None)
        assert torch.allclose(eps, torch.ones(50))


@pytest.mark.integration
class TestScalerAlphaBeta:
    @pytest.fixture(scope="class")
    def scaler(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "1DAW.pdb"
        mtz = mtz_dir / "1DAW.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("1DAW fixture not present")
        from torchref import LBFGSRefinement

        ref = LBFGSRefinement(
            data_file=str(mtz), pdb=str(pdb), target_mode="ml_sigmaa", verbose=0
        )
        ref.scaler.initialize()
        ref.scaler.refine_lbfgs()
        return ref.scaler

    def test_lazy_cache_and_reset(self, scaler):
        scaler.reset_alpha_beta_cache()
        assert scaler._alpha_beta_cache is None
        a1, b1, e1 = scaler.get_alpha_beta()
        assert scaler._alpha_beta_cache is not None
        a2, b2, e2 = scaler.get_alpha_beta()           # cached: identical objects
        assert a1 is a2 and b1 is b2
        assert not a1.requires_grad and not b1.requires_grad

    def test_alpha_in_range(self, scaler):
        alpha, beta, eps = scaler.get_alpha_beta()
        v = scaler._data.masks().to(torch.bool)
        assert (alpha[v] > 0).all() and (alpha[v] <= 1.0 + 1e-6).all()
        assert (beta[v] > 0).all()
