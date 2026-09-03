#!/bin/bash
# 1BIA: our placement matches Phaser's pose to 0.6 deg yet refines 0.10 worse.
# The remaining difference is that Phaser writes the chain as two chains, which
# gives the rigid-body step two bodies. Merge Phaser's into one and split ours
# at Phaser's break: if the gap follows the chain count, that is the cause.
#SBATCH --job-name=biactl
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%A_%a.err
#SBATCH --partition=hour
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=cpu_epyc9335
#SBATCH --array=0-3
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
DEV=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper
cd "$REPO"; export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
W=$REPO/alignment_lab/runs/af_mr/1BIA/control; mkdir -p $W
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
"$PY" - <<PYEOF
import gemmi
D="$DEV"; W="$W"; R="$REPO"
# Phaser's placement, merged into one chain
ph=gemmi.read_structure(f"{D}/figure2_alphafold_start/placed/1BIA_af.pdb")
new=gemmi.Structure(); new.cell=ph.cell; new.spacegroup_hm=ph.spacegroup_hm
m=gemmi.Model("1"); ch=gemmi.Chain("A")
brk=None
for c in ph[0]:
    if brk is None and c.name!=ph[0][0].name: brk=c[0].seqid.num
    for r in c: ch.add_residue(r)
m.add_chain(ch); new.add_model(m); new.write_pdb(f"{W}/phaser_merged.pdb")
print("break at residue", brk)
# our 3 A placement, split at the same residue
ours=gemmi.read_structure(f"{R}/alignment_lab/runs/af_mr_dmin3.0/1BIA/torchref_placed.pdb")
new2=gemmi.Structure(); new2.cell=ours.cell; new2.spacegroup_hm=ours.spacegroup_hm
m2=gemmi.Model("1"); a=gemmi.Chain("A"); b=gemmi.Chain("B")
for c in ours[0]:
    for r in c: (a if r.seqid.num < brk else b).add_residue(r)
for c in (a,b):
    if len(c): m2.add_chain(c)
new2.add_model(m2); new2.write_pdb(f"{W}/ours_split.pdb")
print("ours split:", [ (c.name, len(c)) for c in new2[0] ])
PYEOF
fi
sleep $((SLURM_ARRAY_TASK_ID * 25))
case $SLURM_ARRAY_TASK_ID in
 0) TAG=phaser_merged; PDB=$W/phaser_merged.pdb;;
 1) TAG=ours_split;    PDB=$W/ours_split.pdb;;
 2) TAG=ours3;         PDB=$REPO/alignment_lab/runs/af_mr_dmin3.0/1BIA/torchref_placed.pdb;;
 3) TAG=phaser;        PDB=$DEV/figure2_alphafold_start/placed/1BIA_af.pdb;;
esac
[ -f "$PDB" ] || { echo "CTL tag=$TAG MISSING"; exit 0; }
OUT=$W/refine_$TAG; mkdir -p $OUT
"$PY" -u torchref/cli/refine.py -m $PDB -sf $DEV/data/1BIA/1BIA.mtz -o $OUT -n 10 --mode separate --xray-mode ml --weights '{"adp": 0.02}' --with-rigid-body > $OUT/refine.log 2>&1
echo "CTL tag=$TAG rc=$? $("$PY" -c "import json;d=json.load(open('$OUT/refinement_history.json'))['final_statistics'];print(f\"rwork={d['R_work']:.4f} rfree={d['R_free']:.4f}\")" 2>/dev/null) chains=$(grep -o 'n_chains=[0-9]*' $OUT/refine.log | head -1)"
