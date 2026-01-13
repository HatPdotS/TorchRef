---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.


To use sbatch for testing please run with the following parameters:

sbatch -c 8 -p day -t 1-00:00:00 

PLease use for small scripts srun to test interactively:

srun -c 8 -p day -t 1-00:00:00 


Please do not use GPU nodes for testing unless absolutely necessary.
Also avoid running direct on login node unless for very small quick tests, everything that takes longer should be run on compute nodes.