#!/bin/bash
# Place one structure's AlphaFold model by Phaser MR. Self-contained: prepares
# the search model(s) (gemmi, torchref env), converts pLDDT->B
# (process_predicted_model), runs phaser, and copies the top solution to
# placed/{code}_af.pdb. Idempotent: exits early if already placed.
#
# Usage: run_mr_one.sh CODE
# NB: no `set -u` — phenix's env scripts reference unbound vars.
set -o pipefail

REPO=/das/work/p17/p17490/Peter/Library/work_trees_torchref/review
AFDIR="$REPO/paper/figure2_alphafold_start"
PHENIX_ENV=/afs/psi.ch/sys/psi.ra/MX/phenix/phenix-1.20-4459/phenix_env.sh

code="${1:?usage: run_mr_one.sh CODE}"
placed="$AFDIR/placed/$code"_af.pdb
wd="$AFDIR/search_models/$code"

if [ -f "$placed" ]; then
    echo "[$code] already placed -> $placed"
    exit 0
fi

echo "[$code] preparing search model(s)..."
"$REPO/.dev/bin/python" "$AFDIR/prepare_search_model.py" "$code" || {
    echo "[$code] PREP FAILED"; exit 1; }

# Phenix tools (source directly; the cluster modulefile is incompatible with the
# local Environment Modules, so we bypass `module load`).
source "$PHENIX_ENV"

cd "$wd"
echo "[$code] process_predicted_model (pLDDT->B)..."
# maximum_rmsd=100 disables confidence-based residue removal: we already trimmed
# to the construct, so we only want the pLDDT->B conversion, never an emptied
# model (low-confidence small components were being deleted entirely).
for f in *_search.pdb; do
    proc="${f%.pdb}_processed.pdb"
    if ! phenix.process_predicted_model "$f" maximum_rmsd=100 \
            >> "process_${code}.log" 2>&1; then
        echo "[$code] process_predicted_model failed on $f; using trimmed model as-is"
        cp "$f" "$proc"
    fi
done

echo "[$code] phaser MR..."
phenix.phaser < phaser.keywords > "phaser_${code}.log" 2>&1
echo "[$code] phaser exit=$?"

# Top solution from MR_AUTO is ROOT.1.pdb
sol=$(ls -1 "${code}_phaser.1.pdb" 2>/dev/null | head -1)
if [ -n "$sol" ] && [ -f "$sol" ]; then
    mkdir -p "$AFDIR/placed"
    cp "$sol" "$placed"
    echo "[$code] SOLVED -> $placed"
else
    echo "[$code] NO SOLUTION (see $wd/phaser_${code}.log)"
fi
