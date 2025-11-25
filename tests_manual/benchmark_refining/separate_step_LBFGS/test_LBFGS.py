#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/test_LBFGS_separate_weight_scaling.out
#SBATCH -p day
#SBATCH -t 1-00:00:00 
import torch
from tqdm import tqdm
from torchref.base_refinement import Refinement
import numpy as np


def refine(refinement,  weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}):
    # refinement.model.freeze_all()
    # refinement.model.unfreeze(target)
    refinement.get_scales()
    refinement.effective_weights = weights

    def grad_norm(loss, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
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

    print('Effective weights:', refinement.target_weights)
    for p in refinement.model.parameters():
        print(p.shape)
    for target in ['xyz', 'b']:
        refinement.model.freeze_all()
        refinement.model.unfreeze(target)

        optimizer = torch.optim.LBFGS(
        refinement.parameters(),
        lr=1.0,                 # LBFGS ignores this until line search, but 1.0 is a good baseline
        max_iter=20,            # recommended
        history_size=100,       # if memory allows
        line_search_fn="strong_wolfe"   # improves stability
    )
        
        def closure():
            optimizer.zero_grad()
            loss = loss_function(refinement)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            print(f"Target: {target}, X-ray work: {xray_work.item():.4f}, X-ray test: {xray_test.item():.4f}, Rwork: {rwork:.4f}, Rfree: {rfree:.4f}")
        refinement.model.unfreeze_all()
    return refinement.model.copy() , rfree


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
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS'



weights={'xray': 1.0, 'restraints': 0.5, 'adp': 0.2}

refinement = Refinement(
    pdb=pdb,
    data_file=mtz,
    verbose=2,
)

for i in range(3):
    best_model, best_rfree = refine(refinement, 
                                     weights=weights)
    pdb = f"{outdir}/refined_model_round_{i+1}.pdb"
    best_model.write_pdb(outdir + f"/refined_model_round_{i+1}.pdb")
    print(f"Saved refined model to {pdb} with Rfree: {best_rfree:.4f}")