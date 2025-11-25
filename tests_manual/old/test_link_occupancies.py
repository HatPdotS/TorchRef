import torchref.Model as Model
import pandas as pd
from pdb_tools import write_file

model = Model.model()

model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')

sel1 = [(None,102,"A"),(None,102,"B"),(None,102,"E"),(None,103,'A'),(None,103,'B'),(None,103,'E')]
sel2 = [(None,102,"C"),(None,102,"D"),(None,103,'C'),(None,103,'D')]




model.replace_copies_with_mean(sel1)
model.replace_copies_with_mean(sel2)

model.write_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/projected.pdb')