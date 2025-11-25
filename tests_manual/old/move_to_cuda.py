#!/das/work/units/LBR-FEL/p17490/CONDA/cctbx_peter/bin/python
import torchref.refinement as refinement
import old.restraints_handler as restraints_handler
import torchref.Model as Model
import torchref.Data as Data


cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'

restraints = restraints_handler.restraints(cif_path)

M = Model.model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all_perturbed.pdb')
hkl = Data.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_direct_summation.mtz')

ref = refinement.Refinement(hkl,model=M,restraints=restraints)
ref.cuda()

ref.refine(n_iter=100)