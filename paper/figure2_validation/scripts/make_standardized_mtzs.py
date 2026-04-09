#!/usr/bin/env python -u

#SBATCH -c 32
#SBATCH -p day
#SBATCH -t 1-00:00:00

from pathlib import Path
from glob import glob
import os
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
cif_files = glob(str(DATA_DIR / '*/*-sf.cif'))

from torchref.io import ReflectionData



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