from torchref.model import Model
from torchref.restraints import Restraints

model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1B37/1B37_shaken.pdb'
model.load_pdb(test_pdb)

restraints = Restraints(model, verbose=10)

restraints.cat_dict()

loss = restraints.loss()