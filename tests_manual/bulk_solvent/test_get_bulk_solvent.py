#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u


from torchref.solvent_new import SolventModel as solvent_new
from torchref.solvent import SolventModel as solvent

from torchref.model_ft import ModelFT


pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/dark.pdb'

model = ModelFT(max_res=1.7).load_pdb_from_file(pdb)

solvent_model = solvent_new(model=model)

print(solvent_model.solvent_mask.shape)
print(solvent_model.solvent_mask.sum())

solvent_model_old = solvent(model=model)
print(solvent_model_old.solvent_mask.shape)
print(solvent_model_old.solvent_mask.sum())

