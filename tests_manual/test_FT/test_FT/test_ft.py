#!/das/work/p17/p17490/CONDA/torchref/bin/python

#SBATCH -c 16 
#SBATCH -o /das/work/p17/p17490/Peter/Library/torchref/tests_manual/test_FT/test_FT/test_ft.out

import torch
import numpy as np
import reciprocalspaceship as rs
import gemmi
from tqdm import tqdm
from torchref.math_functions.math_torch import fft, ifft, place_on_grid, extract_structure_factor_from_grid
from torchref.model.model_ft import ModelFT   

pdb = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/test_FT/quality_testing/dark_no_H.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/test_FT/quality_testing'



M = ModelFT()   
M.load_pdb(pdb)

from matplotlib import pyplot as plt

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

def calculate_map_for_pdb(pdb):
    F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=1.5)

    cell = [74.530, 92.580, 83.990, 90.00, 96.71, 90.00]
    spacegroup = 'P21'

    dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
    dataset.infer_mtz_dtypes(inplace=True)
    dataset.cell = cell
    dataset.spacegroup = spacegroup 
    dataset.write_mtz('_temp.mtz')
    dataset = rs.read_mtz('_temp.mtz')
    dataset = dataset.expand_to_p1()
    mtz = gemmi.read_mtz_file('_temp.mtz')

    # Create real-space map from structure factors
    grid = mtz.transform_f_phi_to_map('F-model', 'PHIF-model', sample_rate=1)
    grid_array = torch.tensor(np.array(grid, dtype=np.float32))
    phases = dataset['PHIF-model'].values.astype(np.float32)
    phases = np.deg2rad(phases)
    HKL = torch.tensor(dataset.reset_index().loc[:,['H','K','L']].values.astype(np.int32))
    structure_factor = torch.tensor(dataset['F-model'].values.astype(np.float32) * np.exp(1j * phases),dtype=torch.complex64)
    reciprocal_grid = place_on_grid(HKL, structure_factor, grid_array.shape, enforce_hermitian=True)

    return grid_array, reciprocal_grid, dataset


map, reciprocal_grid, dataset = calculate_map_for_pdb(pdb)

fft_map = fft(reciprocal_grid)

ifft_map = ifft(fft_map)

print(reciprocal_grid.shape)

print(torch.corrcoef(torch.stack([torch.flatten(fft_map), torch.flatten(map)])))
print(torch.corrcoef(torch.stack([torch.flatten(ifft_map), torch.flatten(reciprocal_grid)])))

HKL = torch.tensor(dataset.reset_index().loc[:,['H','K','L']].values.astype(np.int32))
phases = dataset['PHIF-model'].values.astype(np.float32)
phases = np.deg2rad(phases)
original_sf = torch.tensor(dataset['F-model'].values.astype(np.float32) * np.exp(1j * phases), dtype=torch.complex64)

# Extract structure factors from the reciprocal grid
extracted_sf = extract_structure_factor_from_grid(reciprocal_grid, HKL)

amplitude_correlation = torch.corrcoef(torch.stack([torch.abs(original_sf), torch.abs(extracted_sf)]))[0, 1]



print(torch.mean(fft_map), torch.std(fft_map))
print(torch.mean(map), torch.std(map))

normalized_fft_map = (fft_map - torch.mean(fft_map)) / torch.std(fft_map)

normalized_map = (map - torch.mean(map)) / torch.std(map)


assert torch.allclose(torch.abs(original_sf), torch.abs(extracted_sf), rtol=1e-3, atol=1e-3)
assert torch.allclose(torch.angle(original_sf), torch.angle(extracted_sf), rtol=1e-3, atol=1e-3)
assert torch.allclose(normalized_fft_map, normalized_map, rtol=1e-3, atol=1e-3)

M.setup_grid(gridsize=map.shape)

structure_factor_homemade = M.get_structure_factor(HKL)
map_torch = M.build_complete_map()

rec_grid = ifft(map_torch)
SF = extract_structure_factor_from_grid(rec_grid, HKL)
rec_grid_newly_assembled = place_on_grid(HKL, SF, rec_grid.shape, enforce_hermitian=True)
SF_second_pass = extract_structure_factor_from_grid(rec_grid_newly_assembled, HKL)

assert torch.allclose(SF, SF_second_pass, rtol=1e-3, atol=1e-3), 'extract and place_on_grid consistency check failed'


print('Map correlation', torch.corrcoef(torch.stack([torch.flatten(map_torch), torch.flatten(map)]))[0, 1])
print('Amplitude correlation',torch.corrcoef(torch.stack([torch.abs(original_sf), torch.abs(structure_factor_homemade)]))[0, 1])
print('Phase correlation',torch.corrcoef(torch.stack([torch.angle(original_sf), torch.angle(structure_factor_homemade)]))[0, 1])

# Diagnostic: Check phase correlation for different amplitude ranges
print("\n=== Phase Correlation Analysis ===")
amplitudes = torch.abs(original_sf)
phases_ref = torch.angle(original_sf)
phases_calc = torch.angle(structure_factor_homemade)

# Different amplitude thresholds
amplitude_thresholds = [0, 1, 10, 100, 1000]
for i, threshold in enumerate(amplitude_thresholds):
    if i < len(amplitude_thresholds) - 1:
        mask = (amplitudes >= threshold) & (amplitudes < amplitude_thresholds[i+1])
        range_desc = f"{threshold}-{amplitude_thresholds[i+1]}"
    else:
        mask = amplitudes >= threshold
        range_desc = f">={threshold}"
    
    if mask.sum() > 10:  # Need enough points for correlation
        phase_corr = torch.corrcoef(torch.stack([phases_ref[mask], phases_calc[mask]]))[0, 1]
        phase_diff_deg = torch.rad2deg(torch.abs(phases_ref[mask] - phases_calc[mask]))
        mean_diff = torch.mean(phase_diff_deg)
        max_diff = torch.max(phase_diff_deg)
        print(f"Amplitude range {range_desc:>10}: {mask.sum():>6} reflections, phase_corr={phase_corr:.4f}, mean_diff={mean_diff:.1f}°, max_diff={max_diff:.1f}°")

# Check for systematic phase offset
phase_diff = phases_ref - phases_calc
# Handle phase wrapping
phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
mean_phase_offset = torch.mean(phase_diff)
print(f"\nMean systematic phase offset: {torch.rad2deg(mean_phase_offset):.2f}°")

# Check phase correlation excluding very weak reflections
strong_mask = amplitudes > torch.median(amplitudes)
strong_phase_corr = torch.corrcoef(torch.stack([phases_ref[strong_mask], phases_calc[strong_mask]]))[0, 1]
print(f"Phase correlation (strong reflections only): {strong_phase_corr:.4f}")

# Check R-factor equivalent for phases (in degrees)
phase_diff_abs = torch.rad2deg(torch.abs(phase_diff))
r_phase = torch.mean(phase_diff_abs)
print(f"Phase R-factor (mean absolute difference): {r_phase:.2f}°")
