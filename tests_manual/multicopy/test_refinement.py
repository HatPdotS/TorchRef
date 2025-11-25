import torchref.refinement as refinement
import old.restraints_handler as restraints_handler
import torchref.Model as Model
import torchref.Data as Data
import pickle

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'

restraints = restraints_handler.restraints(cif_path)


M = Model.model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.pdb')
hkl = Data.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.mtz')

ref = refinement.Refinement(hkl,model=M,restraints=restraints)


print(ref.get_rfactor())

ref.refine(n_iter=1000,lr=0.0001)

print(ref.get_rfactor())

M.write_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/multicopy/one_copy.pdb')

with open('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/multicopy/one_copy_refined.pickle', 'wb') as f:
    pickle.dump(M, f)