"""Stage timing of one AlphaFold search-model placement (verbose pipeline)."""
import sys, time, json
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.experimental.alignment import MolecularReplacementPipeline
code = sys.argv[1]
REVIEW="/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/review/paper/figure2_alphafold_start"
comp = json.load(open(f"{REVIEW}/search_models/{code}/components.json"))[0]
t0=time.time()
data = ReflectionData().load_mtz(f"/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/data/{code}/{code}.mtz")
m = ModelFT().load_pdb(comp["processed_pdb"]); m.spacegroup="P 1"; m.pdb["chainid"]="A"
print(f"load {time.time()-t0:.1f}s n_refl={len(data.hkl)} d_min={float(data.resolution_min) if hasattr(data,'resolution_min') else 'na'}", flush=True)
for rep in range(2):
    t0=time.time()
    pipe = MolecularReplacementPipeline(data, m, d_min=4.0, d_max=15.0, n_shells=20, n_rotation_peaks=200, n_rotation_candidates=10, verbose=2)
    sols = pipe.run(do_translation=True)
    print(f"REP {rep} total {time.time()-t0:.1f}s llg={sols[0].llg_score:.0f}", flush=True)
