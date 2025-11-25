import torchref.refinement as refinement
import old.restraints_handler as restraints_handler
import torchref.Model as Model
import torchref.Data as Data


cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'

restraints = restraints_handler.restraints(cif_path)


M = Model.model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')
hkl = Data.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.mtz')

ref = refinement.Refinement(hkl,model=M,restraints=restraints)
print(ref.get_rfactor_absorption())

print(ref.optimize_absorption())

print(ref.abs_coeffs)
print(ref.scale)
print(ref.get_rfactor_absorption())