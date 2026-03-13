#!/bin/bash
# Run difference refinement for Figure 4.




module load anaconda
conda activate /das/work/p17/p17490/CONDA/torchref

sbatch -c 16 -o refinement_output/out.log torchref.difference-refine -lm work.pdb -dm data/8QL2.pdb -lsf data/7YYZ-light.mtz -dsf data/dark-phenix.mtz --fraction 0.18 --cif data/IBL_grade.cif -o refinement_output --dmin 1.9  --weight-schedule 10,5,2 --n-cycles 10 