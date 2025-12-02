from torchref.io.Data import ReflectionData




data = ReflectionData().load_cif('/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/2JBV/2JBV-sf.cif')

hkl, fobs, sigfobs, rfree_flags = data()

data_dict, cell, spacegroup =  data.reader()


print("Reflections loaded from CIF:")
print('Shapes:', hkl.shape, fobs.shape, sigfobs.shape, rfree_flags.shape)
print('dtypes:', hkl.dtype, fobs.dtype, sigfobs.dtype, rfree_flags.dtype)

print(hkl)
print(fobs)
print(sigfobs)
print(rfree_flags)

print('rfree flags sum:' , rfree_flags.sum())