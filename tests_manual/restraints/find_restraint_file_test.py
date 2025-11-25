from torchref.restraints_helper import find_cif_file_in_library
from time import time


test_files = ['ARG','ASP','GLU']

for resname in test_files:
    t_start = time()
    cif_path = find_cif_file_in_library(resname)
    t_end = time()
    print(f"Resname: {resname} -> CIF path: {cif_path} (Time taken: {t_end - t_start:.4f} seconds)")