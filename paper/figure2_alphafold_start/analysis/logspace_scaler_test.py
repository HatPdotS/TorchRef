"""Scaler-target comparison from the analytic initialize() start.

For each structure: initialize() (good per-bin analytic scale), then optimize the
scaler params with LBFGS(max_iter=200) against three x-ray targets and report R:
  - linear   : nll_xray            (current get_scales/refine_lbfgs target; blows up)
  - lognormal: nll_xray_lognormal  (log-space; no zero-attractor)
  - sigma_a  : the ml_sigmaa Read/Rice target (beta/eps from get_beta, free-set)
Tests the prediction: log-space robustly holds; Rice/sigma_a only partially fixes.
"""
import gc, torch
from torchref.refinement.lbfgs_refinement import LBFGSRefinement
from torchref.base.metrics.loss import nll_xray, nll_xray_lognormal
B="paper/figure2_alphafold_start"

def build(code):
    ref=LBFGSRefinement(data_file=f"paper/data/{code}/{code}.mtz",
        pdb=f"{B}/runs/torchref_noscaler/{code}/refined.pdb",
        verbose=0, target_mode="ml_sigmaa", device=torch.device("cpu"))
    ref.scaler.initialize()
    return ref

def get_fo_sigma(ref):
    fo,sig=ref.reflection_data.get_corrected_data()
    if hasattr(fo,"get_mask"):
        m=fo.get_mask(); return fo.get_data()[m], (sig.get_data() if hasattr(sig,"get_mask") else sig)[m], m
    return fo,sig,None

def opt(ref, closure):
    o=torch.optim.LBFGS(list(ref.scaler.parameters()),lr=1.0,max_iter=200,
                        history_size=10,line_search_fn="strong_wolfe")
    o.step(closure)
    with torch.no_grad():
        rw,rf=ref.get_rfactor()
    res=(float(rw),float(rf))
    del o
    return res

def run(code, mode):
    ref=build(code); sc=ref.scaler
    if mode=="sigma_a":
        ref.xray_target_work()  # prime beta cache (free-set), detached
        def closure():
            o=None
            for p in sc.parameters(): pass
            ref.zero_grad() if hasattr(ref,"zero_grad") else None
            for p in sc.parameters():
                if p.grad is not None: p.grad=None
            l=ref.xray_target_work(); l.backward(); return l
        return opt(ref, closure)
    fcalc=sc.compute_fcalc().detach(); fo,sig,m=get_fo_sigma(ref)
    lossfn=nll_xray if mode=="linear" else nll_xray_lognormal
    def closure():
        for p in sc.parameters():
            if p.grad is not None: p.grad=None
        sc_f=sc.forward(fcalc).reshape(-1)
        if m is not None: sc_f=sc_f[m]
        l=lossfn(fo, sc_f, sig)+torch.sum(sc.U**2); l.backward(); return l
    return opt(ref, closure)

CODES=["6COK","1VR4","2R6R","2Z1Z","2F9I","6U75","6SRB","5MLZ"]
print(f"{'code':6} {'init':>12} {'linear':>12} {'lognormal':>12} {'sigma_a':>12}  (Rw/Rf)")
for code in CODES:
    r=build(code)
    with torch.no_grad():
        rw0,rf0=r.get_rfactor()
    del r; gc.collect()
    out={}
    for mode in ["linear","lognormal","sigma_a"]:
        try: rw,rf=run(code,mode); out[mode]=f"{rw:.3f}/{rf:.3f}"
        except Exception as e: out[mode]=f"ERR:{type(e).__name__}"
        gc.collect()
    print(f"{code:6} {float(rw0):.3f}/{float(rf0):.3f} {out['linear']:>12} {out['lognormal']:>12} {out['sigma_a']:>12}", flush=True)
