from torchref.alignment.align import PattersonAligner

from torchref.io import ReflectionData

from torchref.model.model_ft import ModelFT as Model
from torchref.alignment.sampling import VectorSampler
import torch


pdb = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/alignment/dark.pdb'
mtz = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/alignment/dark.mtz'


M = Model().load_pdb(pdb)
D = ReflectionData().load_mtz(mtz)



aligner = PattersonAligner(data=D, model=M, verbose=2, n_vectors=int(1e8))
# sampler = VectorSampler(model=M)

# idx1, idx2 = sampler.sample(n_vectors=500)

M = M.select('not resname HOH')  # Exclude water molecules

fractional_coords = M.xyz_fractional()


from torchref.math_functions.math_torch import random_rotation_uniform

def get_rfactor(model, data):
    from torchref.scaling.scaler import Scaler
    S = Scaler(model=model, data=data)
    S.initialize()
    S.fit_all_scales()
    rwork, rfree = S.rfactor()
    return rwork, rfree



R = random_rotation_uniform()
T = torch.zeros(3)

print(get_rfactor(model=M, data=D))

M.xyz[:] = M.xyz().to(torch.float64) @ R.T + T




