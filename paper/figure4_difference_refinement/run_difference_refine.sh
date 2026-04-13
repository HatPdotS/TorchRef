#!/bin/bash
# Run difference refinement for Figure 4.

BASEDIR=figure4_difference_refinement

DM=$BASEDIR/data/8QL2_no_altloc.pdb
LM=$BASEDIR/work_no_altloc.pdb
LSF=$BASEDIR/data/7YYZ-light.mtz
DSF=$BASEDIR/data/8QL2-sf.cif
RES=$BASEDIR/data/IBL_grade.cif
OUTPATH=$BASEDIR/refinement_output

mkdir -p "$OUTPATH"

sbatch -c 16 -p gpu --gres=gpu:1 -o "$OUTPATH/out.log" \
    torchref.difference-refine \
    -dm "$DM" -lm "$LM" \
    -dsf "$DSF" -lsf "$LSF" \
    --fraction 0.22 --cif "$RES" \
    -o "$OUTPATH" --dmin 2.2 \
    --weight-schedule 5,3,1 --n-cycles 10 
