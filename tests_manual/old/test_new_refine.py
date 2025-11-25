#!/das/work/units/LBR-FEL/p17490/CONDA/cctbx_peter/bin/python -u 

import torchref.Model as Model
import torchref.refinement as refinement
import old.restraints_handler as restraints_handler
import torchref.Data as Data
import pandas as pd
import torch
import os
import pickle

cif_path = '/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/manual_refinement/input_files/Merged_restraints_all.cif'
input_model = '/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/manual_refinement/input_files/Alvra_BT_01-2025_refine_100.pdb'
mtz_path = '/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/manual_refinement/input_files/scaled_separate/dark_0.mtz'

outdir = '/das/work/p17/p17489/Peter/2025-01-31_SwissFEL_SFX/manual_refinement/performance_test'
os.makedirs(outdir,exist_ok=True)
sel1 = [(None,102,"A"),(None,102,"B"),(None,102,"E"),(None,103,'A'),(None,103,'B'),(None,103,'E')]
sel2 = [(None,102,"C"),(None,102,"D"),(None,103,'C'),(None,103,'D')]


restraints = restraints_handler.restraints(cif_path)

M = Model.model()
M.load_pdb_from_file(input_model)
print(M.pdb)

for residue in M.residues.values():
    if residue.resname == 'PD':
        residue.set_anharmonic()
        residue.set_core_deformation()

M.replace_copies_with_mean(sel1)
M.replace_copies_with_mean(sel2)

hkl = Data.read_mtz(mtz_path)



ref = refinement.Refinement(hkl,model=M,restraints=restraints,structure_factors_to_refine=['Pd'],use_parametrization=True,weight_restraints=0.5)



residues = ref.model.residues.values()
for residue in residues:
    a = ref.get_structure_factor_for_residue_compilable(residue)
    b = ref.get_structure_factor_for_residue(residue)
    a = a[0] + 1j*a[1]
    b = b
    print(residue.resname,torch.allclose(a,b))


a = ref.get_structure_factor_no_corrections_compilable()
print(a.shape)
b = ref.get_structure_factor_no_corrections()
print(b.shape)
a = a[0] + 1j*a[1]
print(torch.allclose(a,b))