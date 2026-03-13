#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 32
#SBATCH -p day
#SBATCH -t 1-00:00:00
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/scripts/shake_cifs.out

from torchref.model import Model
from glob import glob
from tqdm import tqdm
import os

cifs = glob('/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/*/*.cif')
cifs = [cif for cif in cifs if '-sf' not in cif]
def shake_cif(cif_path):
    outname = cif_path.replace('.cif', '_shaken.pdb')
    # if os.path.exists(outname):
    #     return
    try:
        model = Model()
        model.load_cif(cif_path)
        model.shake_coords(0.2)
        model.shake_b_factors(5.0)
        model.write_pdb(outname)
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")

for cif in tqdm(cifs):
    shake_cif(cif)