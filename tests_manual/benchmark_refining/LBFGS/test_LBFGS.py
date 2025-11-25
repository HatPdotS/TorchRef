#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/LBFGS/test_LBFGS_even_shorter_range.out
#SBATCH -p day
#SBATCH -t 1-00:00:00 
import torch
from tqdm import tqdm
from torchref.base_refinement import Refinement
import numpy as np


def refine(refinement, nsteps, weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}):
    # refinement.model.freeze_all()
    # refinement.model.unfreeze(target)
    refinement.get_scales()
    refinement.effective_weights = weights

    optimizer = torch.optim.LBFGS(
    refinement.parameters(),
    lr=1.0,                 # LBFGS ignores this until line search, but 1.0 is a good baseline
    max_iter=20,            # recommended
    history_size=100,       # if memory allows
    line_search_fn="strong_wolfe"   # improves stability
)
    
    def grad_norm(loss, params):
        for p in params:
            p.grad = None
            loss.backward(retain_graph=True)
            vec = torch.cat([p.grad.flatten().detach()
                            for p in params if p.grad is not None])
        return vec.norm().item()

    # compute norms
    loss_xray = refinement.xray_loss()
    loss_geom = refinement.restraints_loss()
    loss_adp = refinement.adp_loss()

    gx = grad_norm(loss_xray, refinement.model.parameters())
    gg = grad_norm(loss_geom, refinement.model.parameters())
    ga = grad_norm(loss_adp, refinement.model.parameters())

    weight_adp = (gx / (ga + 1e-12)) / weights['adp']
    weight_restraints = (gx / (gg + 1e-12)) / weights['restraints']

    refinement.target_weights['restraints'] = weight_restraints
    refinement.target_weights['adp'] = weight_adp

    def closure():
        optimizer.zero_grad()
        loss = loss_function(refinement)
        loss.backward(retain_graph=True)
        return loss
    
    for i in range(nsteps):
        optimizer.step(closure)
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            print(f"Step {i+1}/{nsteps}, X-ray work: {xray_work.item():.4f}, X-ray test: {xray_test.item():.4f}, Rwork: {rwork:.4f}, Rfree: {rfree:.4f}")
            if rfree < best_rfree:
                best_rfree = rfree
                best_model = refinement.model.copy()
    return best_model , best_rfree


def loss_function(refinement):
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss +  refinement.target_weights['xray'] * refinement.xray_loss()
    if 'restraints' in refinement.target_weights:
        loss = loss + refinement.target_weights['restraints'] * refinement.restraints_loss()
    if 'adp' in refinement.target_weights:
        loss = loss + refinement.target_weights['adp'] * refinement.adp_loss()
    return loss

to_refine = ['xyz', 'b']


mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/LBFGS'



weights={'xray':1.0, 'restraints':10.0, 'adp': 0}
refinement = Refinement(mtz, pdb, verbose=0,target_weights=weights)

for i in range(10):
    refine(refinement, nsteps=5, weights=weights)
