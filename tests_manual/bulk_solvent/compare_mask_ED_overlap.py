#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/mask_overlap.out

from torchref.model_ft import ModelFT
from torchref.solvent import SolventModel




pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/dark.pdb'

model = ModelFT().load_pdb_from_file(pdb)
solvent = SolventModel(model)


ED = model.build_complete_map()



smooth_mask = solvent.smooth_solvent_mask()

inverted = 1.0 - smooth_mask


overlap = (ED * smooth_mask).sum() / ED.sum()
print(f'Fractional overlap of mask with electron density: {overlap}')
neg_overlap = (ED * inverted).sum() / ED.sum()
print(f'Fractional overlap of inverted mask with electron density: {neg_overlap}')


