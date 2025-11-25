from torchref.model import Model
from torchref.restraints import Restraints


model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
model.load_pdb_from_file(test_pdb)


cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path,verbose=2)


pdb = model.pdb
print(pdb.iloc[restraints.restraints['plane']['10_atoms']['indices'][0].cpu().detach().numpy()])
