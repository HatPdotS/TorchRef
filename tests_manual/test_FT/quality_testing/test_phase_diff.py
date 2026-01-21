#!/das/work/p17/p17490/CONDA/torchref/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/torchref/tests_manual/test_FT/quality_testing/compare_cctbx_map_multiplicative_new_io.log

from torchref.model.model_ft import ModelFT
from torchref.symmetry.map_symmetry import MapSymmetry
import torch
import numpy as np
import reciprocalspaceship as rs
import gemmi
from torchref.math_functions.math_torch import ifft, extract_structure_factor_from_grid

pdb = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/test_FT/quality_testing/dark_no_H.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/test_FT/quality_testing'


M = ModelFT()   
M.load_pdb(pdb)

def calculate_scattering_factor_cctbx(pdb_file,hkls = None,d_min=2.0):
    from iotbx import pdb
    pdb_input = pdb.input(file_name=pdb_file)
    xray_structure = pdb_input.xray_structure_simple()
    f_calc = xray_structure.structure_factors(d_min=d_min).f_calc()
    idx = np.array(f_calc.indices())
    f_calc = np.array(f_calc.data())
    if hkls is not None:
        hkls = np.array(hkls,dtype=int)
        hkls = set([tuple(hkl) for hkl in hkls])
        f_calc_new = []
        idx_new = []
        for hkl,val in zip(idx,f_calc):
            if tuple(hkl) in hkls:
                f_calc_new.append(val)
                idx_new.append(hkl)
        f_calc = np.array(f_calc_new)
        idx = np.array(idx_new)
    return f_calc, idx

def reciprocal_basis_matrix(unit_cell):
    # Extract unit cell parameters
    a, b, c, alpha, beta, gamma = unit_cell
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    # Compute real-space basis vectors
    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_gamma = np.sin(gamma)
    volume = np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    a_vec = np.array([a, 0, 0])
    b_vec = np.array([b * cos_gamma, b * sin_gamma, 0])
    c_vec = np.array([
        c * cos_beta,
        c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
        c * volume / sin_gamma
    ])
    # Compute reciprocal basis vectors
    volume_real = np.dot(a_vec, np.cross(b_vec, c_vec))
    a_star = np.cross(b_vec, c_vec) / volume_real
    b_star = np.cross(c_vec, a_vec) / volume_real
    c_star = np.cross(a_vec, b_vec) / volume_real
    # Assemble reciprocal basis matrix
    return np.array([a_star, b_star, c_star])

def get_scattering_vectors(hkl, unit_cell):
    recB = reciprocal_basis_matrix(unit_cell)
    hkl = np.array(hkl)  # Ensure hkl is a numpy array
    s = np.dot(hkl,recB)
    return s


def get_resolution(hkl, unit_cell):
    s = get_scattering_vectors(hkl, unit_cell)
    return 1 / np.sum(s**2, axis=1)**0.5

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

