"""Is the AlphaFold search model's small P1 box the reason 1A0F fails? Place
(a) the processed search model as-is, (b) the same coordinates in the crystal
cell, (c) Phaser's placed chain A as an oracle."""
import sys, time
sys.path.insert(0, "/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab")
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.experimental.alignment import MolecularReplacementPipeline
code = sys.argv[1]
mtz = f"/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/data/{code}/{code}.mtz"
search = sys.argv[2]
phaser = f"/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper/figure2_alphafold_start/placed/{code}_af.pdb"
data = ReflectionData().load_mtz(mtz)
def run(tag, m):
    m.spacegroup = "P 1"
    t0 = time.time()
    pipe = MolecularReplacementPipeline(data, m, d_min=4.0, d_max=15.0, n_shells=20, n_rotation_peaks=200, n_rotation_candidates=10)
    sols = pipe.run(do_translation=True)
    s = sols[0]
    print(f"ARM {tag} cell={[round(float(x),1) for x in m.cell.data.tolist()]} llg={s.llg_score:.0f} llg1={sols[1].llg_score:.0f} r={s.r_factor:.3f} s={time.time()-t0:.1f}", flush=True)
m = ModelFT().load_pdb(search); run("as_is", m)
m2 = ModelFT().load_pdb(search); m2.cell = data.cell; run("crystal_cell", m2)
ph = ModelFT().load_pdb(phaser); chA = ph.select("chain A"); run("phaser_chainA", chA)
