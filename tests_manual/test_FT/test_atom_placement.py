from torchref.model_ft import ModelFT
from torchref.map_symmetry import MapSymmetry
import torch



# Load structure and setup
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

print(f"\nStructure info:")
print(f"  Space group: {M.spacegroup}")
print(f"  Cell: {M.cell}")
print(f"  Map shape: {M.map.shape}")

# Build asymmetric unit map
M.build_density_map(apply_symmetry=True)
print(f"Asymmetric unit map sum: {M.map.sum():.2f}")
M.save_map('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/test_tubulin.ccp4')
