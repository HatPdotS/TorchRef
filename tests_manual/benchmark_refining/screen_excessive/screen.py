#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 56
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/jsons_weight_screening_only_0p5/weight_screening_0p5_adp.out
#SBATCH -p week
#SBATCH -t 2-00:00:00 


import numpy as np
import torch
from torchref.lbfgs_refinement import LBFGSRefinement
import json
import torch
import os


mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/jsons_weight_screening_only_0p5'
outdir_jsons = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/jsons_weight_screening_only_0p5/jsons_weight_screening_only_0p5'

os.makedirs(outdir_jsons, exist_ok=True)

restraints_weights = [i for i in np.logspace(np.log10(0.1), np.log10(10.0), 20)]
adp_weights = [0.5]

results = {'counter': 0, 'res_to_screen': restraints_weights, 'adp_to_screen': adp_weights,'results':[]}
respath = outdir + '/screening_results.json'
for rw in restraints_weights:
    for aw in adp_weights:
        json_out = outdir_jsons + f'/refine_rw_{rw:.3f}_aw_{aw:.3f}.json'
        weights = {'xray': 1.0, 'restraints': rw, 'adp': aw}
        refine = LBFGSRefinement(mtz, pdb, target_weights=weights,verbose=1)
        res = refine.refine(macro_cycles=20)
        with torch.no_grad():
            rwork, rfree = refine.get_rfactor()
        results['results'].append({'res_weight': rw, 'adp_weight': aw, 'rwork': rwork, 'rfree': rfree})
        results['counter'] += 1
        with open(json_out, 'w') as f:
            json.dump(res, f, indent=4)

        with open(respath, 'w') as f:
            json.dump(results, f, indent=4)
    
        
