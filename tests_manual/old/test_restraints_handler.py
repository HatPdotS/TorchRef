from old import restraints_handler
import torchref.Model as Model
from time import time
import numpy as np
cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'

restraints = restraints_handler.restraints(cif_path)


M = Model.model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb')



key_ca1 = ('A', np.int64(13), None)


Ca1 = M.residues[key_ca1]

restraints.get_sigma_torsion(Ca1.get_xyz(), Ca1.get_names(), Ca1.resname)
