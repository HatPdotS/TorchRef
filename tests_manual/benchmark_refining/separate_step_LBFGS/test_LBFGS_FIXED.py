#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/test_LBFGS_FIXED.out
#SBATCH -p day
#SBATCH -t 1-00:00:00 
import torch
from tqdm import tqdm
from torchref.base_refinement import Refinement
import numpy as np


def refine(refinement,  weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}):
    refinement.get_scales()
    refinement.effective_weights = weights
    
    refine_xyz(refinement, weights=weights)
    refine_adp(refinement, weights=weights)

    return refinement.model, refinement.get_rfactor()[1]


def loss_function(refinement):
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss + refinement.target_weights['xray'] * refinement.xray_loss()
    if 'restraints' in refinement.target_weights:
        loss = loss + refinement.target_weights['restraints'] * refinement.restraints_loss()
    if 'adp' in refinement.target_weights:
        loss = loss + refinement.target_weights['adp'] * refinement.adp_loss()
    return loss

def refine_adp(refinement,  weights={'xray':1.0, 'restraints':1.0, 'adp':0.3}):
    refinement.model.freeze_all()
    refinement.model.unfreeze('b')
    print("ADP parameters:")


    def grad_norm(loss, params):
        for p in params():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward(retain_graph=True)
        vec = torch.cat([p.grad.flatten().detach()
                            for p in params() if p.grad is not None])
        return vec.norm().clip(0.01).item()

    # compute norms
    loss_xray_ = refinement.xray_loss()
    loss_adp_ = refinement.adp_loss()
    print([p.shape for p in refinement.model.parameters()])

    gx = grad_norm(loss_xray_, refinement.model.parameters)
    ga = grad_norm(loss_adp_, refinement.model.parameters)

    weight_adp = (gx / (ga + 1e-12)) * weights['adp']
    refinement.target_weights['adp'] = weight_adp

    optimizer = torch.optim.LBFGS(
    refinement.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn="strong_wolfe"
)

    def closure():
        optimizer.zero_grad()
        loss = loss_adp(refinement)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        xray_work, xray_test = refinement.nll_xray()
        rwork, rfree = refinement.get_rfactor()
        print(f"Target: adp, X-ray work: {xray_work.item():.4f}, X-ray test: {xray_test.item():.4f}, Rwork: {rwork:.4f}, Rfree: {rfree:.4f}")
    refinement.model.unfreeze_all()

def refine_xyz(refinement,  weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}):
    refinement.model.freeze_all()
    refinement.scaler.freeze()
    refinement.model.unfreeze('xyz')
    print("XYZ parameters:")
    print([p.shape for p in refinement.model.parameters()])

    def grad_norm(loss, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward(retain_graph=True)
        vec = torch.cat([p.grad.flatten().detach()
                            for p in params if p.grad is not None])
        return vec.norm().clip(0.01).item()
    
    # compute norms
    loss_xray_ = refinement.xray_loss()
    loss_geom_ = refinement.restraints_loss()

    gx = grad_norm(loss_xray_, refinement.parameters())
    gg = grad_norm(loss_geom_, refinement.parameters())

    weight_restraints = (gx / (gg + 1e-12)) * weights['restraints']
    refinement.target_weights['restraints'] = weight_restraints

    optimizer = torch.optim.LBFGS(
    refinement.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn="strong_wolfe"
)

    def closure():
        optimizer.zero_grad()
        loss = loss_geom(refinement)
        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        xray_work, xray_test = refinement.nll_xray()
        rwork, rfree = refinement.get_rfactor()
        print(f"Target: xyz, X-ray work: {xray_work.item():.4f}, X-ray test: {xray_test.item():.4f}, Rwork: {rwork:.4f}, Rfree: {rfree:.4f}")
    refinement.model.unfreeze_all()


def loss_geom(refinement):
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss + refinement.target_weights['xray'] * refinement.xray_loss()    
    if 'restraints' in refinement.target_weights:
        loss = loss + refinement.target_weights['restraints'] * refinement.restraints_loss()
    return loss

def loss_adp(refinement):
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss = loss + refinement.target_weights['xray'] * refinement.xray_loss()
    if 'adp' in refinement.target_weights:
        loss = loss + refinement.target_weights['adp'] * refinement.adp_loss()
    return loss

to_refine = ['xyz', 'b']

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS'

weights = {'xray': 1.0, 'restraints': 0.7, 'adp': 0.4}

refinement = Refinement(
    pdb=pdb,
    data_file=mtz,
    verbose=2,
)

for i in range(20):
    best_model, best_rfree = refine(refinement, weights=weights)
    pdb = f"{outdir}/refined_model_round_{i+1}_FIXED.pdb"
    best_model.write_pdb(pdb)
    print(f"Saved refined model to {pdb} with Rfree: {best_rfree:.4f}")
