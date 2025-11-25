from torchref.model_ft import ModelFT



pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/dark.pdb'


M = ModelFT(verbose=2,max_res=1.7).load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.pdb')


VDW_radii = M.get_vdw_radii()
print(VDW_radii.shape)
print(VDW_radii.max())
print(VDW_radii.argmax())

print(M.pdb.iloc[int(VDW_radii.argmax())])