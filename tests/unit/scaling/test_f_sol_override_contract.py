"""``f_sol_override`` must be a pure argument, and must not change the output rank.

Two separate contracts on :meth:`ScalerBase.forward`, both load-bearing for any caller
that scales several models against one shared scaler:

1. Passing ``f_sol_override`` must not write ``_f_sol_raw``. A caller that scales two
   different fraction mixtures in a row otherwise leaves the *second* mixture's solvent
   cached, and every later call that does not pass an override silently reads it.
2. A batched ``(T, N)`` override paired with a batched ``(T, N)`` ``fcalc`` must return
   ``(T, N)``. The solvent term is broadcast with ``unsqueeze(0)``, which is right for a
   per-reflection ``(N,)`` solvent and wrong for one that already carries the batch axis.

Both are exercised through ``forward`` rather than asserted on internals, so the stub only
has to stand in for the solvent model.
"""

from types import SimpleNamespace

import pytest
import torch


class _StubSolvent:
    """Minimal stand-in for :class:`SolventModel` on the k_sol/B_sol path.

    ``get_rec_solvent`` returns a recognisable constant so a cached value can be told
    apart from a freshly-passed override by value as well as by identity.
    """

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
        return torch.full(
            (hkl.shape[0],), self.value, dtype=torch.complex64
        )


@pytest.fixture
def scaler_with_stub():
    """A bare ``Scaler`` with only the solvent branch live.

    ``bins`` drives the full-size check in ``forward``; the anisotropy, Chebyshev and
    per-bin-B branches are all absent, so ``forward`` reduces to
    ``fcalc + k_sol * f_sol`` and any shape or caching defect is unobscured.
    """
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
    def test_override_leaves_the_cache_untouched(self, scaler_with_stub):
        """The override is an argument, not an assignment."""
        scaler, n, dev = scaler_with_stub
        override = torch.full((n,), 2.0, dtype=torch.complex64, device=dev)

        scaler.forward(torch.ones(n, dtype=torch.complex64, device=dev),
                       f_sol_override=override)

        assert scaler._f_sol_raw is not override, (
            "forward stored the override in the solvent cache"
        )
        assert scaler._f_sol_raw is None, (
            "forward populated the solvent cache from an override; a later call "
            "without one will read this instead of the model's own solvent"
        )

    @pytest.mark.unit
    def test_a_later_call_without_an_override_sees_the_model_solvent(
        self, scaler_with_stub
    ):
        """The consequence of the leak, stated in terms a caller can observe.

        Two mixtures scaled in a row, then a plain call: the plain call must use the
        solvent model, not whichever mixture happened to be scaled last.
        """
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
    def test_batched_override_with_batched_fcalc_keeps_the_batch_rank(
        self, scaler_with_stub
    ):
        """``(T, N)`` in, ``(T, N)`` out -- the batched multi-dataset contract."""
        scaler, n, dev = scaler_with_stub
        t = 3
        fcalc = torch.ones((t, n), dtype=torch.complex64, device=dev)
        override = torch.full((t, n), 2.0, dtype=torch.complex64, device=dev)

        out = scaler.forward(fcalc, f_sol_override=override)

        assert out.shape == (t, n), (
            f"batched override changed the output rank: got {tuple(out.shape)}, "
            f"expected {(t, n)}"
        )

    @pytest.mark.unit
    def test_each_batch_row_matches_the_unbatched_call(self, scaler_with_stub):
        """Rank alone is not enough -- the rows must also be the right ones.

        A wrong broadcast can restore the shape and still pair row *i* of ``fcalc``
        with the wrong row of the solvent.
        """
        scaler, n, dev = scaler_with_stub
        t = 3
        fcalc = torch.stack([
            torch.full((n,), float(i + 1), dtype=torch.complex64, device=dev)
            for i in range(t)
        ])
        override = torch.stack([
            torch.full((n,), float(10 * (i + 1)), dtype=torch.complex64, device=dev)
            for i in range(t)
        ])

        batched = scaler.forward(fcalc, f_sol_override=override)
        for i in range(t):
            scaler._f_sol_raw = None
            single = scaler.forward(fcalc[i], f_sol_override=override[i])
            assert torch.allclose(batched[i], single), (
                f"batched row {i} does not match the equivalent unbatched call"
            )

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
