import torch
import numpy as np
from torchref.model import Model
from torchref.restraints import Restraints
import matplotlib.pyplot as plt

model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
model.load_pdb_from_file(test_pdb)

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path,verbose=2)


restraints.cat_dict()



indices = restraints.restraints['torsion']['all']['indices']

xyz = model.xyz()


xyz1 = xyz[indices[:,0], :]
xyz2 = xyz[indices[:,1], :]
xyz3 = xyz[indices[:,2], :]
xyz4 = xyz[indices[:,3], :]

pdb = model.pdb 

def compute_dihedral(x1, x2, x3, x4):
    b1 = x2 - x1
    b2 = x3 - x2
    b3 = x4 - x3

    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1 = n1 / torch.linalg.norm(n1, dim=1, keepdim=True)
    n2 = n2 / torch.linalg.norm(n2, dim=1, keepdim=True)
    dihedral = torch.arccos(torch.clamp(torch.sum(n1 * n2, dim=1), -1.0, 1.0))


    return dihedral


dihedrals = compute_dihedral(xyz1, xyz2, xyz3, xyz4) * (180.0 / np.pi)
print(dihedrals[0], pdb.iloc[indices[0].cpu().numpy()])


