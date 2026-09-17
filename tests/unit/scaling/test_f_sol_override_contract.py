"""Solvent overrides preserve output shape and leave the default solvent cache intact."""

from types import SimpleNamespace

import pytest
import torch


class _StubSolvent:
    """Minimal stand-in for :class:`SolventModel` on the k_sol/B_sol path."""

    optimize_phase = False

    def __init__(self, value: float = 1.0):
        self.value = value
        self.n_reads = 0

    def k_solvent(self):
        return torch.tensor(0.35)

    def damping(self, s_half_sq):
        return torch.ones_like(s_half_sq)

    def get_rec_solvent(self, hkl):
        self.n_reads += 1
        return torch.full((hkl.shape[0],), self.value, dtype=torch.complex64)


@pytest.fixture
def scaler_with_stub():
    """A bare ``Scaler`` with only the solvent branch live."""
    from torchref.scaling.scaler import Scaler

    n = 6
    scaler = Scaler()
    dev = scaler.device
    scaler.bins = torch.zeros(n, dtype=torch.long, device=dev)
    scaler._s_half_sq = torch.zeros(n, device=dev)
    # The no-override path reads the solvent model at self.hkl, which is a read-only
    # property over self._data.
    scaler._data = SimpleNamespace(
        hkl=torch.zeros((n, 3), dtype=torch.long, device=dev)
    )
    scaler.solvent = _StubSolvent()
    scaler._f_sol_raw = None
    return scaler, n, dev


class TestOverrideDoesNotMutateTheCache:

    @pytest.mark.unit
    def test_a_later_call_without_an_override_sees_the_model_solvent(
        self, scaler_with_stub
    ):
        """The consequence of the leak, stated in terms a caller can observe."""
        scaler, n, dev = scaler_with_stub
        fcalc = torch.ones(n, dtype=torch.complex64, device=dev)

        baseline = scaler.forward(fcalc).clone()  # stub solvent, value 1.0
        scaler._f_sol_raw = None  # as update_solvent() would leave it

        far_off = torch.full((n,), 99.0, dtype=torch.complex64, device=dev)
        scaler.forward(fcalc, f_sol_override=far_off)
        after = scaler.forward(fcalc)

        assert torch.allclose(after, baseline), (
            "a plain forward() after an override call returned the override's "
            "solvent contribution"
        )


class TestOverridePreservesRank:

    @pytest.mark.unit
    def test_each_batch_row_matches_the_unbatched_call(self, scaler_with_stub):
        """Rank alone is not enough -- the rows must also be the right ones."""
        scaler, n, dev = scaler_with_stub
        t = 3
        fcalc = torch.stack(
            [
                torch.full((n,), float(i + 1), dtype=torch.complex64, device=dev)
                for i in range(t)
            ]
        )
        override = torch.stack(
            [
                torch.full((n,), float(10 * (i + 1)), dtype=torch.complex64, device=dev)
                for i in range(t)
            ]
        )

        batched = scaler.forward(fcalc, f_sol_override=override)
        assert batched.shape == (t, n)
        for i in range(t):
            scaler._f_sol_raw = None
            single = scaler.forward(fcalc[i], f_sol_override=override[i])
            assert torch.allclose(
                batched[i], single
            ), f"batched row {i} does not match the equivalent unbatched call"

    @pytest.mark.unit
    def test_unbatched_override_is_unchanged(self, scaler_with_stub):
        """The ``(N,)`` solvent path must keep working; it is what every single-dataset
        caller uses."""
        scaler, n, dev = scaler_with_stub
        fcalc = torch.ones(n, dtype=torch.complex64, device=dev)
        override = torch.full((n,), 2.0, dtype=torch.complex64, device=dev)

        out = scaler.forward(fcalc, f_sol_override=override)

        assert out.shape == (n,)
        # fcalc + k_sol * f_sol, with damping == 1 and no aniso/Chebyshev/per-bin B.
        expected = 1.0 + 0.35 * 2.0
        assert torch.allclose(out.real, torch.full((n,), expected, device=dev))
