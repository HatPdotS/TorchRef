"""Sixteen parameters, 8.6 seconds. Where does Scaler.refine_lbfgs spend it?

`refine_lbfgs` builds its x-ray target with ``model=None`` and passes a detached
``fcalc`` per closure call, and the comment there states the fit never recomputes
structure factors. This counts them rather than trusting that, and times the
three phases separately -- construction, ``initialize``, and the fit -- because
the solvent contribution is also refined and a mask rebuilt per closure call
would look identical from outside.
"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(True)
from lab import BENCH_PDBS, load_case  # noqa: E402


def main():
    from torchref.scaling import Scaler
    from torchref.base.metrics.rfactor import rfactor_work_free
    import torchref.model.model_ft as mft
    import torchref.scaling.solvent as solv

    for pdb in sys.argv[1:] or ["1DAW"]:
        model, data = load_case(pdb)
        N = data.hkl.shape[0]

        counts = {"model_forward": 0, "solvent_forward": 0}
        orig_fwd = mft.ModelFT.forward
        def counted_fwd(self, *a, **kw):
            counts["model_forward"] += 1
            return orig_fwd(self, *a, **kw)
        mft.ModelFT.forward = counted_fwd
        orig_sol = solv.SolventModel.forward
        def counted_sol(self, *a, **kw):
            counts["solvent_forward"] += 1
            return orig_sol(self, *a, **kw)
        solv.SolventModel.forward = counted_sol

        t0 = time.perf_counter()
        s = Scaler(model=model, data=data, nbins=20, verbose=0)
        t_ctor = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            fc = model(data.hkl).detach()
        t_fcalc = time.perf_counter() - t0
        n_after_fcalc = counts["model_forward"]

        t0 = time.perf_counter()
        s.initialize(fc)
        t_init = time.perf_counter() - t0
        n_after_init = counts["model_forward"]
        sol_after_init = counts["solvent_forward"]

        t0 = time.perf_counter()
        s.refine_lbfgs(fcalc=fc)
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            rw, _ = rfactor_work_free(data, torch.abs(s.forward(fc)))
        t_r = time.perf_counter() - t0

        mft.ModelFT.forward = orig_fwd
        solv.SolventModel.forward = orig_sol
        print(f"ROW pdb={pdb} N={N} ctor={t_ctor:.2f}s fcalc={t_fcalc:.2f}s "
              f"init={t_init:.2f}s fit={t_fit:.2f}s rfac={t_r:.2f}s "
              f"total={t_ctor+t_fcalc+t_init+t_fit+t_r:.2f}s "
              f"| model_fwd_during_fit={counts['model_forward']-n_after_init} "
              f"(1 expected: the explicit fcalc) "
              f"solvent_fwd_during_fit={counts['solvent_forward']-sol_after_init} "
              f"R={float(rw):.4f}", flush=True)


main()
