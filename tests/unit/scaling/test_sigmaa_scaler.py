"""Tests for the ML model-error variance ``beta`` estimator and its Scaler integration.

- estimate_beta recovers a known per-shell beta (Phenix port; alpha mean-shift dropped)
- epsilon_from_hkl is sane
- the base Scaler exposes get_beta / reset_beta_cache and caches lazily
"""

import pytest
import torch

from torchref.base.targets.xray_ml_sigmaa import (
    epsilon_from_hkl,
    estimate_beta,
)


@pytest.mark.unit
class TestEstimateBeta:
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
            FO.append(eoc.abs())
            FC.append(fcc.abs())
            DSS.append(torch.full((per,), 0.02 + 0.30 * i / shells, dtype=dt))
            SA.append(sA[i])
            BT.append(var)
        return (
            torch.cat(FO),
            torch.cat(FC),
            torch.cat(DSS),
            torch.stack(SA),
            torch.stack(BT),
        )

    def test_recovers_known_beta(self):
        FO, FC, DSS, sA_true, beta_true = self._synthetic()
        cen = torch.zeros_like(FO, dtype=torch.bool)
        eps = torch.ones_like(FO)
        free = torch.ones_like(FO, dtype=torch.bool)
        beta, bbin, bdss = estimate_beta(FO, FC, cen, eps, DSS, free, per_bin=1000)
        assert beta.shape == FO.shape
        assert (beta > 0).all()
        # per-reflection beta tracks the per-shell truth (rises with resolution as
        # sigma_A falls). Correlation + mid-range magnitude.
        beta_per_refl = beta_true.repeat_interleave(FO.numel() // beta_true.numel())
        cc = torch.corrcoef(torch.stack([beta, beta_per_refl]))[0, 1]
        assert cc > 0.95
        rel = (beta - beta_per_refl).abs() / beta_per_refl
        assert rel.mean() < 0.20

    def test_degenerate_free_set(self):
        n = 100
        FO = torch.rand(n, dtype=torch.float64) * 50 + 1
        FC = torch.rand(n, dtype=torch.float64) * 50 + 1
        cen = torch.zeros(n, dtype=torch.bool)
        eps = torch.ones(n, dtype=torch.float64)
        dss = torch.rand(n, dtype=torch.float64) * 0.3 + 0.02
        free = torch.zeros(n, dtype=torch.bool)  # no free reflections
        beta, *_ = estimate_beta(FO, FC, cen, eps, dss, free)
        assert torch.isfinite(beta).all()
        assert (beta > 0).all()


@pytest.mark.unit
class TestEpsilonFromHkl:
    def test_none_spacegroup_returns_ones(self):
        hkl = torch.randint(-5, 6, (50, 3))
        eps = epsilon_from_hkl(hkl, None)
        assert torch.allclose(eps, torch.ones(50))


@pytest.mark.integration
class TestScalerBeta:
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
        scaler.reset_beta_cache()
        assert scaler._beta_cache is None
        b1, e1 = scaler.get_beta()
        assert scaler._beta_cache is not None
        b2, e2 = scaler.get_beta()  # cached: identical objects
        assert b1 is b2 and e1 is e2
        assert not b1.requires_grad

    def test_beta_positive(self, scaler):
        beta, eps = scaler.get_beta()
        v = scaler._data.masks().to(torch.bool)
        assert (beta[v] > 0).all()
        assert torch.isfinite(beta[v]).all()
