"""Solvent overrides preserve output shape and leave the default solvent cache intact."""

from types import SimpleNamespace

import pytest
import torch

from torchref.config import get_complex_dtype, get_float_dtype, get_int_dtype


class _StubSolvent:
    """Minimal stand-in for :class:`SolventModel` on the k_sol/B_sol path."""

    optimize_phase = False

    def __init__(self, device, value: float = 1.0):
        self.device = device
        self.value = value
        self.n_reads = 0

    def k_solvent(self):
        return torch.tensor(0.35, device=self.device, dtype=get_float_dtype())

    def damping(self, s_half_sq):
        return torch.ones_like(s_half_sq)

    def get_rec_solvent(self, hkl):
        self.n_reads += 1
        return torch.full(
            (hkl.shape[0],), self.value, dtype=get_complex_dtype(), device=self.device
        )


@pytest.fixture
def scaler_with_stub(any_device):
    """A bare ``Scaler`` with only the solvent branch live."""
    from torchref.scaling.scaler import Scaler

    n = 6
    scaler = Scaler(device=any_device)
    dev = scaler.device
    scaler.bins = torch.zeros(n, dtype=get_int_dtype(), device=dev)
    scaler._s_half_sq = torch.zeros(n, device=dev, dtype=get_float_dtype())
    # The no-override path reads the solvent model at self.hkl, which is a read-only
    # property over self._data.
    scaler._data = SimpleNamespace(
        hkl=torch.zeros((n, 3), dtype=get_int_dtype(), device=dev)
    )
    scaler.solvent = _StubSolvent(dev)
    scaler._f_sol_raw = None
    return scaler, n, dev


class TestOverrideDoesNotMutateTheCache:

    @pytest.mark.unit
    def test_a_later_call_without_an_override_sees_the_model_solvent(
        self, scaler_with_stub
    ):
        """A solvent override applies only to the call that supplies it."""
        scaler, n, dev = scaler_with_stub
        fcalc = torch.ones(n, dtype=get_complex_dtype(), device=dev)

        baseline = scaler.forward(fcalc).clone()  # stub solvent, value 1.0
        scaler._f_sol_raw = None  # as update_solvent() would leave it

        far_off = torch.full((n,), 99.0, dtype=get_complex_dtype(), device=dev)
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
                torch.full((n,), float(i + 1), dtype=get_complex_dtype(), device=dev)
                for i in range(t)
            ]
        )
        override = torch.stack(
            [
                torch.full(
                    (n,), float(10 * (i + 1)), dtype=get_complex_dtype(), device=dev
                )
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
        fcalc = torch.ones(n, dtype=get_complex_dtype(), device=dev)
        override = torch.full((n,), 2.0, dtype=get_complex_dtype(), device=dev)

        out = scaler.forward(fcalc, f_sol_override=override)

        assert out.shape == (n,)
        # fcalc + k_sol * f_sol, with damping == 1 and no aniso/Chebyshev/per-bin B.
        expected = 1.0 + 0.35 * 2.0
        assert torch.allclose(
            out.real, torch.full((n,), expected, device=dev, dtype=get_float_dtype())
        )
