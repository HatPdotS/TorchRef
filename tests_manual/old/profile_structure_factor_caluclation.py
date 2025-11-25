#!/das/work/units/LBR-FEL/p17490/CONDA/cctbx_peter/bin/python

#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -p gpu-day
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/profiles/profiling_strcuture_factor_calculaiton.txt

import torchref.Model as Model
import torchref.refinement as refinement
import old.restraints_handler as restraints_handler
import torchref.Data as Data
import pandas as pd
import torch
import os
import pickle
import torch
from torch.profiler import profile, record_function, ProfilerActivity
torch.cuda.reset_peak_memory_stats()

input_model = '/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/manual_refinement/input_files/Alvra_BT_01-2025_refine_100-coot-2.pdb'
mtz_path = '/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/manual_refinement/input_files/mtzs_all_separate/dark_0.mtz'

M = Model.model()
M.load_pdb_from_file(input_model)

hkl = Data.read_mtz(mtz_path)
torch.cuda.reset_peak_memory_stats()


ref = refinement.Refinement(hkl,model=M,
                            structure_factors_to_refine=['Pd'],use_parametrization=True,weigth_xray=1,weight_restraints=0.1)


ref.cuda()
M.cache_isotropic_residues()
M.cache_anisotropic_residues()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("refine_fast"):
        ref.refine(n_iter=100)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
events = prof.key_averages()
df = pd.DataFrame(events)
df.to_csv("/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/profiles/fast_no_special_atoms.csv", index=False)
M.clear_caches()
print('-----------------' *5)


for residue in M.residues.values():
    if residue.resname == 'PD':
        residue.set_anharmonic()
        residue.set_core_deformation()
        
M.cache_isotropic_residues()
M.cache_anisotropic_residues()

print("Peak GPU memory usage: {:.2f} MB".format(torch.cuda.max_memory_allocated() / 1024**2))
torch.cuda.reset_peak_memory_stats()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("refine_fast"):
        ref.refine(n_iter=100)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
df = pd.DataFrame(events)

df.to_csv("/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/profiles/fast_special_atoms.csv", index=False)
M.clear_caches()

import torch


print("Peak GPU memory usage: {:.2f} MB".format(torch.cuda.max_memory_allocated() / 1024**2))
torch.cuda.reset_peak_memory_stats()

print('-----------------' *5)

ref.model.use_structure_factor_fast = False

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_refine"):
        ref.refine(n_iter=100)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
df = pd.DataFrame(events)

df.to_csv("/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/profiles/slow.csv", index=False)

print("Peak GPU memory usage: {:.2f} MB".format(torch.cuda.max_memory_allocated() / 1024**2))
torch.cuda.reset_peak_memory_stats()


