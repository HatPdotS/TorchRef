"""Rebuilding the solvent mask must drop the cached ``F_sol``.

``ScalerBase.forward`` recomputes ``_f_sol_raw`` only when it is ``None``. A caller that
rebuilds the mask without clearing it leaves ``F_calc`` on the old mask -- which is what
``LBFGSRefinement.refine`` did for every macrocycle, so bulk solvent never followed the
coordinates. The two halves are one operation now; these tests hold them together.
"""

import pytest
import torch


class _StubSolvent:
    """Minimal stand-in for :class:`SolventModel`.

    ``get_rec_solvent`` returns a value derived from ``generation``, so "the cache is
    stale" becomes an observable difference rather than an assertion about internals.
    """

    def __init__(self):
        self.generation = 0
        self.n_rebuilds = 0

    def update_solvent(self):
        self.generation += 1
        self.n_rebuilds += 1

    def get_rec_solvent(self, hkl):
        return torch.full(
            (hkl.shape[0],), float(self.generation), dtype=torch.complex64
        )


@pytest.fixture
def scaler_with_stub():
    from torchref.scaling.scaler import Scaler

    scaler = Scaler()
    stub = _StubSolvent()
    scaler.solvent = stub
    return scaler, stub


class TestUpdateSolventInvalidates:
    @pytest.mark.unit
    def test_rebuilds_mask_and_drops_cache(self, scaler_with_stub):
        """One call must do both halves."""
        scaler, stub = scaler_with_stub
        scaler._f_sol_raw = torch.ones(4, dtype=torch.complex64)

        scaler.update_solvent()

        assert stub.n_rebuilds == 1, "mask was not rebuilt"
        assert scaler._f_sol_raw is None, "stale F_sol survived the mask rebuild"

    @pytest.mark.unit
    def test_next_read_reflects_the_new_mask(self, scaler_with_stub):
        """The point of invalidating: what a caller reads back actually changes.

        Without the invalidation this asserts equality with the *old* generation and fails.
        """
        scaler, stub = scaler_with_stub
        hkl = torch.zeros((4, 3))

        scaler._f_sol_raw = stub.get_rec_solvent(hkl)  # generation 0
        before = scaler._f_sol_raw.clone()

        scaler.update_solvent()  # -> generation 1
        after = (
            scaler._f_sol_raw
            if scaler._f_sol_raw is not None
            else stub.get_rec_solvent(hkl)
        )

        assert not torch.allclose(after, before), (
            "F_sol did not change after the mask was rebuilt -- the cache is stale"
        )
        assert torch.allclose(after, torch.full((4,), 1.0, dtype=torch.complex64))

    @pytest.mark.unit
    def test_noop_without_a_solvent_model(self):
        """Callers should not have to guard; a scaler with no solvent must not raise."""
        from torchref.scaling.scaler import Scaler

        scaler = Scaler()
        scaler.solvent = None
        scaler._f_sol_raw = None

        scaler.update_solvent()  # must not raise

    @pytest.mark.unit
    def test_repeated_calls_keep_rebuilding(self, scaler_with_stub):
        """Each macrocycle gets a fresh mask, not just the first."""
        scaler, stub = scaler_with_stub
        for _ in range(3):
            scaler._f_sol_raw = torch.ones(4, dtype=torch.complex64)
            scaler.update_solvent()
        assert stub.n_rebuilds == 3


class TestSolventModelClearsItsOwnCache:
    @pytest.mark.unit
    def test_update_solvent_clears_the_hkl_cache(self, monkeypatch):
        """``SolventModel._cache`` holds the FFT of the mask, so a new mask voids all of it."""
        from torchref.scaling.solvent import SolventModel

        solvent = SolventModel()
        solvent._cache[("fake_key",)] = torch.ones(3, dtype=torch.complex64)
        assert len(solvent._cache) == 1  # anti-vacuity: there is something to clear

        # The mask rebuild itself needs a model; only the cache reset is under test.
        monkeypatch.setattr(solvent, "get_solvent_mask", lambda: None)
        solvent.update_solvent()

        assert len(solvent._cache) == 0, "mask-derived cache survived a mask rebuild"


class TestRefinementDriversUseTheScalerLevelCall:
    """The defect was a *caller* reaching past the scaler to ``solvent.update_solvent()``.

    Both drivers must go through the scaler, which is the only thing that owns
    ``_f_sol_raw``.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "module_name",
        [
            "torchref.refinement.lbfgs_refinement",
            "torchref.refinement.rigid_body_refinement",
        ],
    )
    def test_driver_does_not_bypass_the_scaler(self, module_name):
        import importlib
        import inspect

        module = importlib.import_module(module_name)
        source = inspect.getsource(module)

        assert "solvent.update_solvent()" not in source, (
            f"{module_name} calls SolventModel.update_solvent directly; that rebuilds the "
            "mask without dropping the scaler's cached F_sol. Use scaler.update_solvent()."
        )
        assert "scaler.update_solvent()" in source
