#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 32
#SBATCH -p day
#SBATCH -t 1-00:00:00
#SBACTH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/scripts/make_standardized_mtzs.out
from glob import glob
import os
from tqdm import tqdm

cif_files = glob('/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/*/*-sf.cif')

from torchref.Data import ReflectionData



def convert_cif_to_mtz(cif_path):
    outname = cif_path.replace('-sf.cif', '.mtz')
    if os.path.exists(outname):
        return
    try:
        refl_data = ReflectionData()
        refl_data.load_cif(cif_path)
        refl_data.regenerate_rfree_flags(force=True)
        refl_data.write_mtz(outname)
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")

for cif in tqdm(cif_files):
    convert_cif_to_mtz(cif)