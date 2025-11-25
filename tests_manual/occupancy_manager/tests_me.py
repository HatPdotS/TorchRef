from torchref import model


pdb_in = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/test_FT/dark.pdb'
m = model.Model(verbose=1)
m.load_pdb_from_file(pdb_in)


m.occupancy()