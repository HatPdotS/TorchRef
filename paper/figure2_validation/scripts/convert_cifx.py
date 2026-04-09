#!/usr/bin/env python -u

#SBATCH -c 32
#SBATCH -p day
#SBATCH -t 1-00:00:00

from pathlib import Path
from torchref.model import Model
from glob import glob
from tqdm import tqdm
import os

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
cifs = glob(str(DATA_DIR / '*/*.cif'))
cifs = [cif for cif in cifs if '-sf' not in cif]
def convert_cif_to_pdb(cif_path):
    outname = cif_path.replace('.cif', '_converted_with_H.pdb')
    if os.path.exists(outname):
        return
    try:
        model = Model(strip_H=False)
        model.load_cif(cif_path)
        model.write_pdb(outname)
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")


from multiprocessing import Pool
with Pool(processes=32) as pool:
    list(tqdm(pool.imap(convert_cif_to_pdb, cifs), total=len(cifs)))