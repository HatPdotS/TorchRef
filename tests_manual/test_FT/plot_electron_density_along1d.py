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

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/test_FT'


M = ModelFT()   
M.load_pdb(pdb)

from matplotlib import pyplot as plt

def calculate_map_for_pdb(pdb):
    F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=1)

    cell = [14.980, 18.870, 18.950, 88.36, 84.87, 67.86]
    spacegroup = 'P1'

    dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
    dataset.infer_mtz_dtypes(inplace=True)
    dataset.cell = cell
    dataset.spacegroup = spacegroup
    dataset.write_mtz('{}/_temp.mtz'.format(outdir))

    mtz = gemmi.read_mtz_file('{}/_temp.mtz'.format(outdir))

    grid = mtz.transform_f_phi_to_map('F-model', 'PHIF-model', sample_rate=2)
    grid = np.array(grid, dtype=np.float32)
    return grid

map_cctbx = calculate_map_for_pdb(pdb)
map_cctbx[map_cctbx < 0] = 0
print("CCTBX map shape:", map_cctbx.shape)

M.setup_grid(gridsize = map_cctbx.shape)
M.build_complete_map()

M.save_map('{}/me.ccp4'.format(outdir))

normalized_cctbx = map_cctbx / np.max(map_cctbx)
dtached = M.map.detach().cpu().numpy()
normalized_me = dtached / np.max(dtached)


plt.imshow(normalized_cctbx[:, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig("{}/xy_cctbx.png".format(outdir))
plt.close()

plt.imshow(normalized_me[:, :, 0], cmap='viridis')
plt.colorbar()  

plt.savefig("{}/xy.png".format(outdir))
plt.close()

plt.plot(normalized_cctbx[32, :, 0], label='CCTBX')
plt.plot(normalized_me[32, :, 0], label='ME')
plt.legend()
plt.savefig("{}/line_y.png".format(outdir))  
plt.close()

diff = normalized_cctbx - normalized_me
diff = diff - np.mean(diff)
plt.imshow(diff[:, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig("{}/xy_diff.png".format(outdir))  
plt.close()

M.map = torch.tensor(diff)  

M.save_map('{}/diff.ccp4'.format(outdir))

M.map = torch.tensor(map_cctbx)

M.save_map('{}/cctbx.ccp4'.format(outdir))

print('Correlation coefficient:', np.corrcoef(normalized_cctbx.flatten(), normalized_me.flatten())[0,1])