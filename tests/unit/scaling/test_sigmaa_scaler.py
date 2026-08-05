"""Tests for the ML model-error variance ``beta`` estimator.

- estimate_beta recovers a known per-shell beta (Phenix port; alpha mean-shift dropped)
- epsilon_from_hkl is sane
- the (target-owned) SigmaAEstimator caches lazily and resets
"""

import pytest
import torch

from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl, estimate_beta


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

    def test_recovers_known_sigma_a(self):
        """sigma_A is the one estimated parameter, so recovery is tested on IT.

        The tolerance is the analytic sampling sd ``(1-sA^2)/(sA*sqrt(2n))``, not an
        arbitrary percentage: with ``per`` reflections per shell there is a floor on how
        well sigma_A can be known, and a tighter assertion would be testing luck.
        ``shrink=False`` so this measures the solve alone -- the stability shrinkage
        deliberately trades per-shell fidelity for stability and is tested separately.
        """
        FO, FC, DSS, sA_true, beta_true = self._synthetic()
        cen = torch.zeros_like(FO, dtype=torch.bool)
        eps = torch.ones_like(FO)
        free = torch.ones_like(FO, dtype=torch.bool)
        # per_bin == per so the fitted shells line up 1:1 with the truth shells
        sh = estimate_beta(
            FO, FC, cen, eps, DSS, free, per_bin=2000, shrink=False
        )
        assert sh.sigma_a.numel() == sA_true.numel()
        sd = (1 - sA_true**2) / (sA_true * torch.sqrt(2 * sh.counts))
        err = (sh.sigma_a - sA_true).abs()
        assert (err < 4 * sd).all(), (
            f"sigma_A off by {err.tolist()} vs 4*sd {(4 * sd).tolist()}"
        )
        # beta follows from sigma_A, so it recovers too
        rel = (sh.beta - beta_true).abs() / beta_true
        assert float(rel.mean()) < 0.15, rel.tolist()
        # and it falls with resolution, because Sigma_N does
        assert sh.beta[0] > sh.beta[-1]

    def test_per_reflection_interpolation_preserves_the_identity(self):
        """The per-reflection derivation must keep ``alpha**2 Sigma_P + beta_model + S2``
        consistent, which is why sigma_A / log Sigma_N / log Sigma_P are interpolated and
        beta is not: interpolating beta directly can produce a value consistent with no
        ``sigma_A <= 1`` at all."""
        from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator

        FO, FC, DSS, sA_true, _bt = self._synthetic()
        cen = torch.zeros_like(FO, dtype=torch.bool)
        eps = torch.ones_like(FO)
        free = torch.ones_like(FO, dtype=torch.bool)
        sig = torch.full_like(FO, 3.0)
        est = SigmaAEstimator().get(FO, FC, cen, eps, DSS, free, sigma_obs=sig)
        assert est.beta.shape == FO.shape
        assert (est.beta > 0).all() and (est.beta_model > 0).all()
        assert (est.beta >= est.beta_model).all()
        assert (est.sigma_a >= 0).all() and (est.sigma_a <= 1).all()
        assert (est.alpha > 0).all() and torch.isfinite(est.alpha).all()

    def test_degenerate_free_set(self):
        n = 100
        FO = torch.rand(n, dtype=torch.float64) * 50 + 1
        FC = torch.rand(n, dtype=torch.float64) * 50 + 1
        cen = torch.zeros(n, dtype=torch.bool)
        eps = torch.ones(n, dtype=torch.float64)
        dss = torch.rand(n, dtype=torch.float64) * 0.3 + 0.02
        free = torch.zeros(n, dtype=torch.bool)  # no free reflections
        beta = estimate_beta(FO, FC, cen, eps, dss, free).beta
        assert torch.isfinite(beta).all()
        assert (beta > 0).all()


@pytest.mark.unit
class TestEpsilonFromHkl:
    def test_none_spacegroup_returns_ones(self):
        hkl = torch.randint(-5, 6, (50, 3))
        eps = epsilon_from_hkl(hkl, None)
        assert torch.allclose(eps, torch.ones(50))


@pytest.mark.unit
class TestSigmaAEstimator:
    """The stateful, target-owned beta estimator: lazy cache + reset contract."""

    def _inputs(self, n=4000, seed=1):
        from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator  # noqa: F401

        g = torch.Generator().manual_seed(seed)
        dt = torch.float64
        F_obs = torch.rand(n, generator=g, dtype=dt) * 100 + 1
        F_calc = torch.rand(n, generator=g, dtype=dt) * 80 + 1
        centric = torch.zeros(n, dtype=torch.bool)
        eps = torch.ones(n, dtype=dt)
        dss = torch.rand(n, generator=g, dtype=dt) * 0.3 + 0.02
        free = torch.rand(n, generator=g) < 0.5
        return F_obs, F_calc, centric, eps, dss, free

    def test_lazy_cache_and_reset(self):
        from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator

        est = SigmaAEstimator()
        assert est._cache is None
        args = self._inputs()
        r1 = est.get(*args)
        b1, e1 = r1.beta, r1.epsilon
        assert est._cache is not None
        r2 = est.get(*args)  # cached: identical object (args ignored)
        b2, e2 = r2.beta, r2.epsilon
        assert b1 is b2 and e1 is e2
        assert not b1.requires_grad
        est.reset()
        assert est._cache is None

    def test_beta_positive(self):
        from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator

        est = SigmaAEstimator()
        _r = est.get(*self._inputs())
        beta, eps = _r.beta, _r.epsilon
        assert (beta > 0).all()
        assert torch.isfinite(beta).all()
