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
