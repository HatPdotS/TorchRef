#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/test_LBFGS.out

import torch
from tqdm import tqdm
from torchref.base_refinement import Refinement



def refine(refinement, target, nsteps):
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    refinement.get_scales()

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
    
    for i in range(nsteps):
        optimizer.step(closure)
        with torch.no_grad():
            xray_work, xray_test = refinement.nll_xray()
            rwork, rfree = refinement.get_rfactor()
            print(f"Step {i+1}/{nsteps}, Target: {target}, X-ray work: {xray_work.item():.4f}, X-ray test: {xray_test.item():.4f}, Rwork: {rwork:.4f}, Rfree: {rfree:.4f}")


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


refinement = Refinement(mtz, pdb, verbose=0,target_weights={'xray':1.0, 'restraints':1.0, 'adp':0.3})

for i in range(4):
    for target in to_refine:
        refine(refinement, target, nsteps=3)