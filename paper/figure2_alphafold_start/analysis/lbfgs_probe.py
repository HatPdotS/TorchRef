import torch
from torchref.refinement.lbfgs_refinement import LBFGSRefinement
B="paper/figure2_alphafold_start"
def build():
    ref=LBFGSRefinement(data_file="paper/data/1DAW/1DAW.mtz", pdb=f"{B}/placed/1DAW_af.pdb",
                        verbose=0, target_mode="ml_sigmaa", device=torch.device("cpu"))
    ref.get_scales()
    return ref


def one_cycle(ref, max_iter, nsteps):
    state = ref.complete_loss_state()
    state.set_weight("xray", 1.0)
    state.set_weight("xyz", 1.0)
    state.set_weight("adp", 0.1)
    body = ref.model.parameters_of_types(("xyz",))
    params = body
    optimizer_xyz = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")
    adp = ref.model.parameters_of_types(("adp",))
    params_adp = adp
    optimizer_adp = torch.optim.LBFGS(params_adp, max_iter=max_iter, line_search_fn="strong_wolfe")
    state.run(optimizer_xyz, nsteps)
    post_xyz, post_xyz_free = ref.get_rfactor()
    state.run(optimizer_adp, nsteps)
    post_adp, post_adp_free = ref.get_rfactor()
    return post_xyz, post_xyz_free, post_adp, post_adp_free





ref=build(); rw0,rf0=ref.get_rfactor()
print(f"START (after init scale): Rwork={float(rw0):.4f} Rfree={float(rf0):.4f}")
print(f"PHENIX cycle1 target:     Rwork=0.3320 Rfree=0.3411  (default torchref = max_iter20,nsteps1)")
print(f"{'max_iter':>8} {'nsteps':>6} | {'afterXYZ Rw/Rf':>18} | {'afterADP Rw/Rf':>18}")
for mi,ns in [(20,1),(20,3),(50,1),(100,1),(100,3),(200,1),(200,3)]:
    r=build()
    rwx,rfx,rwa,rfa=one_cycle(r,mi,ns)
    print(f"{mi:>8} {ns:>6} | {rwx:7.4f}/{rfx:7.4f}    | {rwa:7.4f}/{rfa:7.4f}", flush=True)
