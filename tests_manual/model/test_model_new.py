from torchref.model import Model


test = Model()

test.load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb')

print(test.b())
