from torchref.io import ReflectionData 
from torchref.model.simple_model import SimpleModel
from torchref.model.model import Model
from time import time

data = ReflectionData().load_mtz('/das/work/p17/p17490/Peter/Library/torchref/tests_manual/benchmark_simple_modelFT/dark.mtz')

M = Model().load_pdb('/das/work/p17/p17490/Peter/Library/torchref/tests_manual/benchmark_simple_modelFT/dark.pdb')

xyz = M.xyz()
B = M.b()
occ = M.occupancy()
elements = M.pdb.element.tolist()
hkl = data.hkl
cell = data.cell
spacegroup = data.spacegroup


s = SimpleModel()

s(xyz, B, occ, elements, cell, spacegroup, hkl)


t_start = time()
for _ in range(10):
    s(xyz, B, occ, elements, cell, spacegroup, hkl)
t_end = time()

print(f"Elapsed time for 10 runs: {t_end - t_start} seconds")

t_start = time()
for _ in range(10):
    s(xyz, B, occ, elements, cell, spacegroup, hkl)
    s.clear_cache()
t_end = time()

print(f"Elapsed time for 10 runs with cache clearing: {t_end - t_start} seconds")