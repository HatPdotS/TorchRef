from torchref.io import ReflectionData

from time import time
mtz = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/alignment/dark.mtz'



# Profile your startup
tstart = time()
for i in range(10):
    D = ReflectionData().load_mtz(mtz)


tend = time()
print(f"Data load time: {tend - tstart:.2f} seconds")