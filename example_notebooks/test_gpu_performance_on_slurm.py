#!/das/work/p17/p17490/CONDA/torchref/bin/python 

#SBATCH -p gpu-day
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -o /das/work/p17/p17490/Peter/Library/torchref/example_notebooks/test_gpu_performance_on_slurm.out


from time import time
from torchref.refinement import LBFGSRefinement


pdb_file = '/das/work/p17/p17490/Peter/Library/torchref/example_notebooks/1DAW.pdb'
mtz = '/das/work/p17/p17490/Peter/Library/torchref/example_notebooks/1DAW.mtz'

refinement_object = LBFGSRefinement(pdb=pdb_file, data_file=mtz)


#Create the loss state for refinement and add target info, meta info, and weights

loss_state = refinement_object.create_loss_state()

refinement_object.add_target_info_to_state(loss_state) # add target info to the loss state
refinement_object.populate_state_meta(loss_state) # populate the state with meta info

refinement_object.update_weights(loss_state) # update/set weights in the loss state

print(loss_state.weights) # weights
print("Total loss", loss_state.aggregate())


refinement_object.cuda() # move to GPU
loss_state.cuda()

t_start = time()


hkl, F, sigF, rfree = refinement_object.data()

print('Data loaded on GPU:', hkl.device, F.device, sigF.device, rfree.device)

from torch.profiler import profile, record_function, ProfilerActivity


parameters = refinement_object.parameters()


for loss in loss_state.targets.values():
    loss()

refinement_object.model.reset_cache()

for name, loss in loss_state.targets.items():
    t_start_loss = time()
    loss()
    t_end_loss = time()
    print(f"Loss {name} computed in {t_end_loss - t_start_loss} seconds")

# del loss_state.targets['geometry/planarity']

for name, loss in loss_state.targets.items():
    t_start_loss = time()
    loss()
    t_end_loss = time()
    print(f"Loss {name} computed in {t_end_loss - t_start_loss} seconds")


t_start = time()
refinement_object._optimize_lbfgs(loss_state, parameters, max_iter=100,nsteps=1) # run 100 iterations of L-BFGS and step once

print(refinement_object.get_rfactor()) # final rfactor

print('time taken', time() - t_start)


