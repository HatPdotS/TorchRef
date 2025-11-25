from torchref.model import Model
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.pdb'



model = Model().load_pdb_from_file(pdb)


model.shake_coords(0.1)
model.shake_b_factors(2.0)

model.write_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/shake_coords/shaken_BR_LCLS_refine_8.pdb')


