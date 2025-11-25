#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/test_FT/quality_testing/compare_cctbx_map_multiplicative_new_cif.log

from torchref.model_ft import ModelFT
from torchref.map_symmetry import MapSymmetry
import torch
import numpy as np
from crystfel_tools.handling.fast_math import calculate_scattering_factor_cctbx,get_resolution
import reciprocalspaceship as rs
import gemmi
from torchref.math_functions.math_torch import ifft, extract_structure_factor_from_grid

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/7CWW/7CWW.cif'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/test_FT/quality_testing'


M = ModelFT()   
M.load_cif(pdb)

def wrap_phases(phases):
    """Wrap phases to [-π, π] accounting for periodicity"""
    return np.arctan2(np.sin(phases), np.cos(phases))

def calculate_map_for_pdb(pdb):
    F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=1)

    cell = [74.530, 92.580, 83.990, 90.00, 96.71, 90.00]
    spacegroup = 'P21'

    dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
    dataset.infer_mtz_dtypes(inplace=True)
    dataset.cell = cell
    dataset.spacegroup = spacegroup
    dataset.write_mtz('_temp.mtz')
    hkl = dataset.reset_index()[['H','K','L']].values.astype(np.int32)

    mtz = gemmi.read_mtz_file('_temp.mtz')

    grid = mtz.transform_f_phi_to_map('F-model', 'PHIF-model', sample_rate=1)
    grid = np.array(grid, dtype=np.float32)
    return grid, hkl, dataset

map_cctbx, hkl, datas = calculate_map_for_pdb(pdb)

print("CCTBX map shape:", map_cctbx.shape)

M.setup_grid(gridsize = map_cctbx.shape)
M.build_complete_map()

M.save_map(f'{outdir}/me.ccp4')

dtached = M.map.detach().cpu().numpy() 

print('Correlation coefficient:' , np.corrcoef(map_cctbx.flatten(), dtached.flatten())[0,1])

hkl = torch.tensor(hkl)

noisy_me = dtached * np.random.normal(1,0.01,dtached.shape)

f_me = np.array(extract_structure_factor_from_grid(ifft(torch.tensor(dtached)), hkl))
f_cctbx = np.array(extract_structure_factor_from_grid(ifft(torch.tensor(map_cctbx)), hkl))
f_noisy = np.array(extract_structure_factor_from_grid(ifft(torch.tensor(noisy_me)), hkl))

Fme = np.abs(f_me)
Fctbx = np.abs(f_cctbx)
Fnoisy = np.abs(f_noisy)
phase_me = wrap_phases(np.angle(f_me))
phase_cctbx = wrap_phases(np.angle(f_cctbx))
phase_noisy = wrap_phases(np.angle(f_noisy))
phase_weighted_me = Fme * phase_me
phase_weighted_cctbx = Fctbx * phase_cctbx
phase_weighted_noisy = Fnoisy * phase_noisy

print('==' * 10)
print('Correlation F-me vs F-cctbx:', np.corrcoef(Fme, Fctbx)[0,1])
print('Correlation phase-me vs phase-cctbx:', np.corrcoef(phase_me, phase_cctbx)[0,1])
print('Correlation phase-weighted-me vs phase-weighted-cctbx:', np.corrcoef(phase_weighted_me, phase_weighted_cctbx)[0,1])
print('--' * 10)
print('Correlation F-noisy vs F-cctbx:', np.corrcoef(Fnoisy, Fctbx)[0,1])
print('Correlation phase-noisy vs phase-me:', np.corrcoef(phase_noisy, phase_me)[0,1])
print('Correlation phase-weighted-noisy vs phase-weighted-me:', np.corrcoef(phase_weighted_noisy, phase_weighted_me)[0,1])
print('--' * 10)
print('Correlation F-noisy vs F-cctbx:', np.corrcoef(Fnoisy, Fctbx)[0,1])
print('Correlation phase-noisy vs phase-cctbx:', np.corrcoef(phase_noisy, phase_cctbx)[0,1])
print('Correlation phase-weighted-noisy vs phase-weighted-cctbx:', np.corrcoef(phase_weighted_noisy, phase_weighted_cctbx)[0,1])
print('--' * 10)

print('==' * 10)
print('With normalization:')




dtached_norm = (dtached - np.mean(dtached)) / np.std(dtached)
map_cctbx_norm = (map_cctbx - np.mean(map_cctbx)) / np.std(map_cctbx)
noisy_norm = (noisy_me - np.mean(noisy_me)) / np.std(noisy_me)

#norm maps

f_me = np.array(extract_structure_factor_from_grid(ifft(torch.tensor(dtached_norm)), hkl))
f_cctbx = np.array(extract_structure_factor_from_grid(ifft(torch.tensor(map_cctbx_norm)), hkl))
f_noisy = np.array(extract_structure_factor_from_grid(ifft(torch.tensor(noisy_norm)), hkl))

Fme = np.abs(f_me)
Fctbx = np.abs(f_cctbx)
Fnoisy = np.abs(f_noisy)
phase_me = wrap_phases(np.angle(f_me))
phase_cctbx = wrap_phases(np.angle(f_cctbx))
phase_noisy = wrap_phases(np.angle(f_noisy))
phase_weighted_me = Fme * phase_me
phase_weighted_cctbx = Fctbx * phase_cctbx
phase_weighted_noisy = Fnoisy * phase_noisy


print('Correlation F-me vs F-cctbx:', np.corrcoef(Fme, Fctbx)[0,1])
print('Correlation phase-me vs phase-cctbx:', np.corrcoef(phase_me, phase_cctbx)[0,1])
print('Correlation phase-weighted-me vs phase-weighted-cctbx:', np.corrcoef(phase_weighted_me, phase_weighted_cctbx)[0,1])
print('--' * 10)
print('Correlation F-noisy vs F-cctbx:', np.corrcoef(Fnoisy, Fctbx)[0,1])
print('Correlation phase-noisy vs phase-me:', np.corrcoef(phase_noisy, phase_me)[0,1])
print('Correlation phase-weighted-noisy vs phase-weighted-me:', np.corrcoef(phase_weighted_noisy, phase_weighted_me)[0,1])
print('--' * 10)
print('Correlation F-noisy vs F-cctbx:', np.corrcoef(Fnoisy, Fctbx)[0,1])
print('Correlation phase-noisy vs phase-cctbx:', np.corrcoef(phase_noisy, phase_cctbx)[0,1])
print('Correlation phase-weighted-noisy vs phase-weighted-cctbx:', np.corrcoef(phase_weighted_noisy, phase_weighted_cctbx)[0,1])
print('--' * 10)



diff = f_me - f_cctbx

amplitude = np.abs(diff)
phase = np.rad2deg(np.angle(diff))

