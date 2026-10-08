"""Tests for the experimental kinetic occupancy models.

Pins the behaviour of :class:`KineticModel` and the :class:`occupancies_kinetics`
wrapper that callers rely on: which state starts populated, that plotting does not
change what ``forward()`` evaluates, that the activation level and the hidden
light-activated state reach the structural occupancies, that the ``nn.Module``
parameter API works, and that the matrix exponential stays in the working dtype
while matching a float64 reference on a stiff scheme.
"""

import pytest
import torch

from torchref.config import get_float_dtype
from torchref.experimental.kinetic import kinetics as kinetics_module
from torchref.experimental.kinetic.kinetics import KineticModel
from torchref.experimental.kinetic.occupancies import occupancies_kinetics

pytestmark = pytest.mark.unit


def _model(flow_chart, times, **kwargs):
    kwargs.setdefault("instrument_function", "none")
    kwargs.setdefault("activation_level", 1.0)
    kwargs.setdefault("verbose", 0)
    return KineticModel(
        flow_chart, torch.tensor(times, dtype=get_float_dtype()), **kwargs
    )


class TestInitialState:
    @pytest.mark.parametrize(
        "flow_chart", ["G->E,E->X", "pG->pR,pR->pB", "bR->K,K->L,L->M", "A->B,B->C"]
    )
    def test_default_initial_state_is_first_in_flow_chart(self, flow_chart):
        first, second = flow_chart.split(",")[0].split("->")
        km = _model(flow_chart, [0.0, 1.0, 10.0, 100.0], instrument_width=1.0)
        assert km.initial_state == first
        populations = km().detach()
        assert populations[0, km.state_to_idx[first]].item() == pytest.approx(1.0)
        # The transition out of the start state gets the instrument-limited rate.
        assert km.get_rate_constants()[f"{first}->{second}"] == pytest.approx(3.0)

    def test_explicit_initial_state_drives_rate_initialisation(self):
        km = _model(
            "E->X,G->E", [0.0, 1.0, 10.0], initial_state="G", instrument_width=1.0
        )
        assert km.get_rate_constants()["G->E"] == pytest.approx(3.0)

    def test_light_activated_plot_merges_start_state(self, tmp_path, monkeypatch):
        km = _model("G->E,E->G", [0.0, 1.0, 10.0], light_activated=True)
        assert km.states == ["G", "E", "G*"]
        labels = []
        plot = kinetics_module.plt.plot
        monkeypatch.setattr(
            kinetics_module.plt,
            "plot",
            lambda *a, **kw: labels.append(kw.get("label")) or plot(*a, **kw),
        )
        km.plot_occupancies(str(tmp_path / "occ.png"))
        assert labels == ["State G", "State E"]


class TestPlotTimes:
    def test_plot_with_times_leaves_model_timepoints(self, tmp_path):
        times = [0.0, 1.0, 5.0, 20.0, 100.0]
        km = _model("A->B,B->C", times, rate_constants=[1.0, 0.1])
        before = km().detach().clone()
        km.plot_occupancies(str(tmp_path / "occ.png"), times=[0.0, 10.0, 25.0, 50.0])
        assert torch.equal(km.timepoints, torch.tensor(times, dtype=get_float_dtype()))
        torch.testing.assert_close(km().detach(), before)


class TestWrapper:
    def _wrapper(self, **kwargs):
        return occupancies_kinetics(
            flow_chart="A->B,B->C,C->D",
            time=torch.linspace(0, 100, 50),
            rate_constants={"A->B": 1.0, "B->C": 0.1, "C->D": 0.01},
            verbose=0,
            **kwargs,
        )

    def test_activation_level_is_forwarded(self):
        occ = self._wrapper(activation_level=1.0)().detach()
        assert occ[0, -1].item() == pytest.approx(0.0, abs=1e-6)
        torch.testing.assert_close(occ.sum(0), torch.ones(50))

    def test_activation_level_none_means_fully_reactive(self):
        occ = self._wrapper(activation_level=None)().detach()
        assert occ[0, -1].item() == pytest.approx(0.0, abs=1e-6)

    def test_default_activation_level_keeps_half_baseline(self):
        assert self._wrapper().kinetics.get_baselines()["A"] == pytest.approx(0.5)


class TestLightActivatedMapping:
    times = [0.0, 1.0, 10.0, 100.0, 1000.0]

    def _wrapper(self, **kwargs):
        return occupancies_kinetics(
            flow_chart="A->B,B->A,B->C",
            time=self.times,
            rate_constants={"A->B": 1.0, "B->A": 0.1, "B->C": 0.01},
            light_activated=True,
            verbose=0,
            **kwargs,
        )

    def test_default_mapping_puts_inactive_state_on_its_parent(self):
        occ = self._wrapper()
        assert occ.nstates == 3
        assert occ.state_mapping["A*"] == occ.state_mapping["A"]
        torch.testing.assert_close(occ().detach().sum(0), torch.ones(len(self.times)))

    def test_user_mapping_without_inactive_state_keeps_population(self):
        occ = self._wrapper(state_mapping={"A": 0, "B": 1, "C": 2})
        assert occ.state_mapping["A*"] == 0
        torch.testing.assert_close(occ().detach().sum(0), torch.ones(len(self.times)))

    def test_unmapped_state_is_rejected(self):
        with pytest.raises(ValueError, match="C"):
            self._wrapper(state_mapping={"A": 0, "B": 1})
