from torchref.model import Model
from torchref.restraints import Restraints

model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1B37/1B37_shaken.pdb'
model.load_pdb(test_pdb)


for param in model.parameters():
    print(param.shape)


print('--'*10, "no xyz")

model.freeze('xyz')


for param in model.parameters():
    print(param.shape)

model.unfreeze_all()

print('--'*10, "no b")

model.freeze('b')
for param in model.parameters():
    print(param.shape)

model.unfreeze_all()

print('--'*10, "no xyz and b")
model.freeze('xyz')
model.freeze('b')

for param in model.parameters():
    print(param.shape)
