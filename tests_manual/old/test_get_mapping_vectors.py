import torchref.Model as Model
import pandas as pd
from pdb_tools import write_file

model = Model.model()

model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')
# model.pdb = model.pdb.loc[~model.pdb['name'].isin(['C16','C18','C7','C13'])]

print(model.pdb)
# selections = [(None,102,None,"A"),(None,102,None,"B"),(None,103,None,'A'),(None,103,None,'B'),(None,102,None,"E"),(None,103,None,'E')]





# model.replace_copies_with_mean(selections)


# selections = [(None,102,None,"C"),(None,102,None,"D"),(None,103,None,'C'),(None,103,None,'D')]


# # model.pdb = model.pdb.loc[~model.pdb['name'].isin(['C16','C18','C7','C13'])]

# model.replace_copies_with_mean(selections)
# print(model.pdb)

# model.write('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_expanded_copies.pdb')

