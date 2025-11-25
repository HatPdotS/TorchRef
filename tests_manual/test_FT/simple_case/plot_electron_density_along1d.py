#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/logs/compare_cctbx_map.log

from torchref.model_ft import ModelFT
from torchref.map_symmetry import MapSymmetry
import torch
import numpy as np
from crystfel_tools.handling.fast_math import calculate_scattering_factor_cctbx,get_resolution
import reciprocalspaceship as rs
import gemmi

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/one_atom.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/simple_case'


M = ModelFT()   
M.load_pdb_from_file(pdb)


from matplotlib import pyplot as plt

def calculate_map_for_pdb(pdb):
    F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=1.5)

    cell = [14.980, 18.870, 18.950, 88.36, 84.87, 67.86]
    spacegroup = 'P1'

    dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
    dataset.infer_mtz_dtypes(inplace=True)
    dataset.cell = cell
    dataset.spacegroup = spacegroup
    dataset.write_mtz('_temp.mtz')

    mtz = gemmi.read_mtz_file('_temp.mtz')

    grid = mtz.transform_f_phi_to_map('F-model', 'PHIF-model', sample_rate=2)
    grid = np.array(grid, dtype=np.float32)
    return grid

map_cctbx = calculate_map_for_pdb(pdb)
print("CCTBX map shape:", map_cctbx.shape)

M.setup_grid(gridsize = map_cctbx.shape)
M.build_density_map(apply_symmetry=True)

normalized_cctbx = map_cctbx 
dtached = M.map.detach().cpu().numpy()
normalized_me = dtached 

print(np.corrcoef(normalized_cctbx.flatten(), normalized_me.flatten()))

plt.imshow(normalized_cctbx[:, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig(outdir + "/xy_cctbx.png")
plt.close()

plt.imshow(normalized_me[:, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig(outdir + "/xy.png")
plt.close()


plt.plot(normalized_cctbx[:, 0, 0], label='CCTBX')
plt.plot(normalized_me[:, 0, 0], label='ME')
plt.legend()
plt.savefig(outdir + "/line_y.png")  
plt.close()

diff = normalized_cctbx - normalized_me

plt.imshow(diff[:, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig(outdir + "/xy_diff.png")  
plt.close()

