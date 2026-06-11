"""Full 25-cycle refinement, scaler co-refined in body (baseline) vs held fixed.

Mirrors the CLI flow (refine then a final get_scales) and reports the honest
get_rfactor. Tests whether excluding the scaler from the xyz/adp body steps
(corefine_scaler=False) improves the final R-free, not just cycle 1.
"""
import torch
from torchref.refinement.lbfgs_refinement import LBFGSRefinement

B = "paper/figure2_alphafold_start"
PHENIX = {"3E8V": (0.241, 0.260), "2A9S": (0.231, 0.238)}  # Rw/Rf from logs


def run(code, corefine):
    ref = LBFGSRefinement(
        data_file=f"paper/data/{code}/{code}.mtz",
        pdb=f"{B}/placed/{code}_af.pdb",
        verbose=0, target_mode="ml_sigmaa", device=torch.device("cpu"),
        corefine_scaler=corefine,
    )
    ref.loss_state.set_weight("adp", 0.1)
    ref.refine(macro_cycles=25)
    ref.get_scales()  # final scale, as the CLI does
    rw, rf = ref.get_rfactor()
    return float(rw), float(rf)


print(f"{'code':6} {'config':22} {'Rwork':>7} {'Rfree':>7}   phenix Rf")
for code in ["3E8V", "2A9S"]:
    for cf, lab in [(True, "corefine_scaler=True"), (False, "corefine_scaler=False")]:
        rw, rf = run(code, cf)
        print(f"{code:6} {lab:22} {rw:7.4f} {rf:7.4f}   {PHENIX[code][1]:.3f}", flush=True)
