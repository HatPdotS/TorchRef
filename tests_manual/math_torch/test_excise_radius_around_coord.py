from torchref.math_functions.math_torch import excise_angstrom_radius_around_coord
from torchref.model_ft import ModelFT



pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'

model = ModelFT().load_pdb_from_file(pdb)


test = excise_angstrom_radius_around_coord(
    real_space_grid=model.real_space_grid,
    start_indices=model.xyz(),
    radius_angstrom=3.0,

)

print(test)
print(test.shape)