#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/test_FT/quality_testing/compare_cctbx_map_changed_to_sperical_roi_new_io.log

from torchref.model_ft import ModelFT
from torchref.map_symmetry import MapSymmetry
import torch
import numpy as np
from crystfel_tools.handling.fast_math import calculate_scattering_factor_cctbx,get_resolution
import reciprocalspaceship as rs
import gemmi

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/test_FT/quality_testing/dark_no_H.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/test_FT/quality_testing'


M = ModelFT()   
M.load_pdb(pdb)

from matplotlib import pyplot as plt

def calculate_map_for_pdb(pdb):
    F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=1)

    cell = [74.530, 92.580, 83.990, 90.00, 96.71, 90.00]
    spacegroup = 'P21'

    dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
    dataset.infer_mtz_dtypes(inplace=True)
    dataset.cell = cell
    dataset.spacegroup = spacegroup
    dataset.write_mtz('_temp.mtz')

    mtz = gemmi.read_mtz_file('_temp.mtz')

    grid = mtz.transform_f_phi_to_map('F-model', 'PHIF-model', sample_rate=1)
    grid = np.array(grid, dtype=np.float32)
    return grid

map_cctbx = calculate_map_for_pdb(pdb)

print("CCTBX map shape:", map_cctbx.shape)

M.setup_grid(gridsize = map_cctbx.shape)
M.build_complete_map()

M.save_map(f'{outdir}/me.ccp4')

dtached = M.map.detach().cpu().numpy()


plt.imshow(map_cctbx[:, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig(f"{outdir}/xy_cctbx.png")
plt.close()

plt.imshow(dtached[:, :, 0], cmap='viridis')
plt.colorbar()  

plt.savefig(f"{outdir}/xy.png")
plt.close()

plt.plot(map_cctbx[32, :, 0], label='CCTBX')
plt.plot(dtached[32, :, 0], label='ME')
plt.legend()
plt.savefig(f"{outdir}/line_y.png")  
plt.close()

diff = map_cctbx - dtached
diff = diff - np.mean(diff)
plt.imshow(diff[:, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig(f"{outdir}//xy_diff.png")  
plt.close()

M.map = torch.tensor(diff)  

M.save_map(f"{outdir}/diff.ccp4")

M.map = torch.tensor(map_cctbx)

M.save_map(f"{outdir}/cctbx.ccp4")

print('Correlation coefficient:' , np.corrcoef(map_cctbx.flatten(), dtached.flatten())[0,1])



