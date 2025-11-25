from torchref.model import model
import pdb_tools

pdb = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print("DataFrame shape:", pdb.shape)
print("\nDataFrame index type:", type(pdb.index))
print("DataFrame index range:", pdb.index.min(), "to", pdb.index.max())
print("\n'index' column range:", pdb['index'].min(), "to", pdb['index'].max())
print("\nAre they the same?", (pdb.index == pdb['index']).all())

# Check altlocs
altloc_df = pdb[pdb['altloc'] != '']
print("\n\nAltloc entries:")
print(altloc_df[['serial', 'name', 'altloc', 'resname', 'chainid', 'resseq', 'index']].head(10))

# Check if index column matches DataFrame index
print("\n\nFirst altloc entry:")
print("  iloc position:", altloc_df.index[0])
print("  'index' column value:", altloc_df['index'].iloc[0])
print("  Are they the same?", altloc_df.index[0] == altloc_df['index'].iloc[0])
