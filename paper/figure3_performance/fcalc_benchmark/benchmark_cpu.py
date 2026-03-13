#!/usr/bin/env python


import torch
from torchref import ReflectionData 
from torchref import ModelFT
from time import time
from iotbx import pdb


mtz_file = '/das/work/p17/p17490/Peter/Library/torchref/Figure3_profiling_fcalc/data/1DAW.mtz'
pdb_file = '/das/work/p17/p17490/Peter/Library/torchref/Figure3_profiling_fcalc/data/1DAW.pdb'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = ReflectionData(device=device).load_mtz(mtz_file)
d_min = data.d_min
print(f"d_min: {d_min}")

M = ModelFT(max_res=d_min, device=device,radius_angstrom=3.0).load_pdb(pdb_file)

hkl, _, _, _ = data()
M(hkl, recalc=True)

t_start = time()
with torch.no_grad():
    if device.type == 'cuda':
        torch.cuda.synchronize()
    for _ in range(10):
        fcalc_tr = M(hkl, recalc=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
t_end = time()

print(f"Elapsed time for 10 runs with cache clearing: {t_end - t_start} seconds")

pdb_input = pdb.input(file_name=pdb_file)
xray_structure = pdb_input.xray_structure_simple()
f_calc = xray_structure.structure_factors(d_min=d_min).f_calc()


t_start = time()
for _ in range(10):
    f_calc_cctbx = xray_structure.structure_factors(d_min=d_min).f_calc()
t_end = time()



print(f"Elapsed time for 10 runs of cctbx calculation: {t_end - t_start} seconds")

