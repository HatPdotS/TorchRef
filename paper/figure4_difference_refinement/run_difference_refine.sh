#!/bin/bash
# Run difference refinement for Figure 4.

BASEDIR=$(dirname "$0")

DM=$BASEDIR/data/8QL2.pdb
LM=$BASEDIR/data/torchref_0p18.pdb
DSF=$BASEDIR/data/8QL2-sf.cif
LSF=$BASEDIR/data/7YYZ-light.mtz
RES=$BASEDIR/data/IBL_grade.cif
OUTPATH=$BASEDIR/refinement_output

mkdir -p "$OUTPATH"

sbatch -c 16 -o "$OUTPATH/out.log" \
    torchref.difference-refine \
    -dm "$DM" -lm "$LM" \
    -dsf "$DSF" -lsf "$LSF" \
    --fraction 0.18 --cif "$RES" \
    -o "$OUTPATH" --dmin 2.1 \
    --weight-schedule 10,5,2 --n-cycles 3
