#!/usr/bin/env python
"""Which objective should the joint (model-to-data) scale fit use?

``CollectionScaler.refine_lbfgs_joint`` used to hand-roll a Rice likelihood at
``beta = sigma_obs**2`` on an objective with no normaliser at all. It now builds a row of
``XRAY_TARGETS``, defaulting to unit-weight ``ls`` -- the same default the single-dataset
scale fit was moved to after measurement.

This measures the swap, on the figure-4 pair, against the two things that could make a
single-run comparison meaningless:

* **Thread nondeterminism.** TorchRef's CPU F_calc is not bit-reproducible, and R-free
  jitter of order 0.017 has been measured at zero true difference. So every arm is
  repeated and the spread is reported alongside the mean; a difference smaller than the
  spread is not a result.
* **Circularity.** Each objective is scored on ``rfree``, which none of them optimises
  (they all fit the work set), and by the same ``rfactor_work_free`` for every arm.

The legacy fit is reconstructed here rather than kept selectable: the point is to measure
what it did, not to preserve it.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def build(dark_sf, light_sf, dark_pdb, light_pdb, cif, d_min, fraction, device):
    from torchref.cli.collection_difference_refine import (
        setup_dataset_collection,
        setup_model_collection,
    )

    mc = setup_model_collection(
        dark_pdb, light_pdb, [1.0 - fraction, fraction], cif, d_min, device, 0
    )
    dc = setup_dataset_collection(dark_sf, light_sf, d_min, device)
    return dc, mc


def legacy_joint(scaler, dc, mc, nsteps=3, max_iter=200):
    """The pre-change fit: hand-rolled Rice at beta=sigma**2, no normaliser."""
    import torch.nn as nn

    from torchref.base.reciprocal import get_scattering_vectors
    from torchref.base.targets.xray_likelihoods import complex_var_from_beta, rice_math
    from torchref.refinement.loss_state import LossState
    from torchref.refinement.model_error_estimation.sigma_a import (
        SigmaAEstimator,
        epsilon_from_hkl,
    )
    from torchref.scaling.collection_scaler import CollectionScaler

    keys = [k for k in ([mc.dark_key] + mc.timepoint_names) if k in dc]
    cache = {}
    for name in keys:
        data, model = dc[name], mc[name]
        with torch.no_grad():
            fc = model(data.hkl).detach()
            fracs = model.fractions.detach()
            scaled0 = CollectionScaler.forward_mixed(scaler, fc, fracs)
            amp0 = torch.abs(scaled0).reshape(-1)
            fobs, sig = data.get_corrected_data()
            eps0 = epsilon_from_hkl(data.hkl, getattr(data, "spacegroup", None)).to(amp0.dtype)
            s = get_scattering_vectors(data.hkl, data.cell)
            dss0 = (torch.norm(s, dim=1) ** 2).to(amp0.dtype)
            est = SigmaAEstimator().get(
                fobs.to(amp0.dtype).reshape(-1), amp0, data.centric, eps0, dss0,
                data.free.mask, sigma_obs=sig.to(amp0.dtype).reshape(-1),
            )
        cache[name] = (fc, fracs, est.beta, est.epsilon, data.work, data.centric)

    class _T(nn.Module):
        name = "scaler/joint"

        def forward(self):
            total, n = torch.tensor(0.0, device=scaler.device), 0
            for nm in keys:
                fc, fracs, beta, eps, work, cen = cache[nm]
                scaled = CollectionScaler.forward_mixed(scaler, fc, fracs)
                amp = torch.abs(scaled).reshape(-1)
                fo = work.F.to(amp.dtype)
                bw = work.select(beta).to(fo.dtype)
                ew = work.select(eps).to(fo.dtype) if eps is not None else None
                loss = rice_math(
                    fo, work.select(amp), complex_var_from_beta(bw, ew), work.select(cen)
                )
                if torch.isfinite(loss):
                    total, n = total + loss, n + 1
            if n:
                total = total / n
            return total + torch.sum(scaler.U**2)

    state = LossState(device=scaler.device)
    state.register_target("scaler/joint", _T())
    opt = torch.optim.LBFGS(
        scaler.parameters(), lr=1.0, max_iter=max_iter, history_size=10,
        line_search_fn="strong_wolfe",
    )
    state.run(opt, nsteps=nsteps, log=False, context="probe.legacy_joint")


def rfactors(scaler, dc, mc):
    """``{key: (rwork, rfree)}`` for every dataset under the current scale."""
    from torchref.base.metrics.rfactor import rfactor_work_free
    from torchref.scaling.collection_scaler import CollectionScaler

    out = {}
    with torch.no_grad():
        for name in ([mc.dark_key] + mc.timepoint_names):
            if name not in dc:
                continue
            data, model = dc[name], mc[name]
            fc = model(data.hkl).detach()
            scaled = CollectionScaler.forward_mixed(scaler, fc, model.fractions.detach())
            out[name] = rfactor_work_free(data, torch.abs(scaled))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fig4 = Path(__file__).resolve().parent / "figure4_difference_refinement"
    ap.add_argument("--dark-sf", default=str(fig4 / "data/8QL2-sf.cif"))
    ap.add_argument("--light-sf", default=str(fig4 / "data/7YYZ-light.mtz"))
    ap.add_argument("--dark-pdb", default=str(fig4 / "data/8QL2_no_altloc.pdb"))
    ap.add_argument("--light-pdb", default=str(fig4 / "work_no_altloc.pdb"))
    ap.add_argument("--cif", nargs="*", default=[str(fig4 / "data/IBL_grade.cif")])
    ap.add_argument("--dmin", type=float, default=2.2)
    ap.add_argument("--fraction", type=float, default=0.22)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    from torchref.scaling.collection_scaler import CollectionScaler

    dev = torch.device(args.device)
    dc, mc = build(args.dark_sf, args.light_sf, args.dark_pdb, args.light_pdb,
                   args.cif, args.dmin, args.fraction, dev)

    arms = ["legacy_rice_unnormalised", "ls", "nll", "ml_noalpha"]
    results = {a: {"rfree_dark": [], "rfree_light": [], "rwork_dark": []} for a in arms}

    for rep in range(args.repeats):
        for arm in arms:
            # A fresh scaler each time: `initialize()` reseeds every parameter, so no arm
            # inherits another's answer.
            scaler = CollectionScaler(dc, mc, verbose=0).initialize()
            if arm == "legacy_rice_unnormalised":
                legacy_joint(scaler, dc, mc)
            else:
                scaler.refine_lbfgs_joint(verbose=False, scale_target=arm)
            rf = rfactors(scaler, dc, mc)
            dark, light = mc.dark_key, mc.timepoint_names[0]
            results[arm]["rwork_dark"].append(rf[dark][0])
            results[arm]["rfree_dark"].append(rf[dark][1])
            results[arm]["rfree_light"].append(rf[light][1])
        print(f"  repeat {rep + 1}/{args.repeats} done", flush=True)

    def ms(v):
        m = statistics.mean(v)
        s = statistics.stdev(v) if len(v) > 1 else 0.0
        return m, s

    print()
    print(f"Joint scale fit, {args.repeats} repeats -- scored on reflections it did not fit")
    print("=" * 82)
    print(f"{'objective':28s} {'rwork_dark':>18s} {'rfree_dark':>18s} {'rfree_light':>18s}")
    print("-" * 82)
    for a in arms:
        cells = []
        for k in ("rwork_dark", "rfree_dark", "rfree_light"):
            m, s = ms(results[a][k])
            cells.append(f"{m:.5f}+-{s:.5f}")
        print(f"{a:28s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")
    print("-" * 82)
    base = results["legacy_rice_unnormalised"]
    for a in arms[1:]:
        for k in ("rfree_dark", "rfree_light"):
            # Paired by repeat index: the same thread-nondeterminism realisation.
            d = [x - y for x, y in zip(results[a][k], base[k])]
            m, s = ms(d)
            flag = "  <-- exceeds its own spread" if abs(m) > 2 * (s or 1e-9) else ""
            print(f"{a:28s} d({k}) vs legacy = {m:+.5f} +- {s:.5f}{flag}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
