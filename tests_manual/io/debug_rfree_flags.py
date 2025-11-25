from torchref.Data import ReflectionData


data = ReflectionData(verbose=2).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.mtz')


hkl, fobs, _, rfree = data()

print(rfree)
print(rfree.shape)
print(rfree.sum())
print(rfree.dtype)