import torchref.refinement as refinement
import old.restraints_handler as restraints_handler
import torchref.Model as Model
import torchref.Data as Data


cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'

restraints = restraints_handler.restraints(cif_path)


M = Model.model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all_perturbed.pdb')



from torchref.get_scattering_factor_torch import get_parameterization


print(get_parameterization(M.pdb))