from torchref import Data
from torchref import difference_refinement



mtz1 = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.mtz'
mtz2 = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.mtz'


mtz1 = Data.read_mtz(mtz1)
mtz2 = Data.read_mtz(mtz2)

import torchref.Model as Model


model = Model.model()

model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')

model2 = Model.model()
model2.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')

ref = difference_refinement.Difference_refinement(mtz1,mtz2,model,model2)

ref.get_scales()