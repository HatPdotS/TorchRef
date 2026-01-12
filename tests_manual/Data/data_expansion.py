from torchref.io import ReflectionData

mtz_path = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/patterson/dark.mtz'

data = ReflectionData().load_mtz(mtz_path)

sg = data.spacegroup



data_rt = data.expand_to_p1().reduce_to_spacegroup(sg)

print(data)
print(data_rt)