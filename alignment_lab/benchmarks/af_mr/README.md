# The AlphaFold molecular-replacement benchmark

Does the pipeline place a predicted search model as well as Phaser does, and
does the placement refine as far?

Each structure's AlphaFold components are placed by
`MolecularReplacementPipeline` (largest first, each copy conditioned on the
chains already placed, the rotation shortlist reused for a sequence-identical
copy). The assembled solution is refined with the Figure-2 recipe plus the
rigid-body step, and Phaser's placement of the same components is refined
identically. Phaser's arm is the control: both arms differ only in where the
atoms started.

## Running it

```
sbatch alignment_lab/benchmarks/af_mr/run.sh                 # 15-3 A
D_MIN=4.0 TAG=dmin4 sbatch alignment_lab/benchmarks/af_mr/run.sh
python alignment_lab/benchmarks/af_mr/table.py --tag dmin3.0
```

One array task per structure, results in `runs/<tag>/<code>/summary.json`.
Re-running a tag overwrites it.

## The sample

`worklist.txt` is 50 structures drawn uniformly, seed 20260904, from the 767
that have a Phaser placement, a components manifest and their own data.
It is committed so the benchmark is the same set every time; `sample_worklist.py`
rebuilds it and can draw a different size or seed.

## Reading the result

Placement and refinement are reported separately because they fail
independently. A component counts as placed within 5 degrees and 2 A of the
Phaser copy it matches, and `placed` is the fraction of residues placed, so a
structure whose large chains are right and whose 13-residue fragment is missed
is not scored as a failure. The R-free comparison is summarised only over the
fully placed structures, since a misplaced model's R-free measures the
misplacement.

Two things must be held fixed for the comparison to mean anything, and both
were once wrong here:

* **The B level.** A predicted model carries a confidence-derived B around
  6 A^2; refinement reads it as a B level and never recovers. `place()` shifts
  the placed model to the data's Wilson B, as Phaser does to its output. Before
  that, identical poses refined 0.01-0.10 apart in R-free.
* **The chain partition.** Phaser writes a chain break as a separate chain, so
  its file hands the rigid-body step two bodies where ours gives one. On 1BIA
  that is worth 0.10 in R-free by itself.

## Known limits

* `pose_vs_phaser` groups Phaser's split chains into copies by C-alpha count
  and can return an undefined pose when a component has no match; that reads as
  a failed component.
* Placement seconds in the array logs are cold-start and kernel-compile
  dominated. `seconds_warm` re-times the first placement with the caches built
  and is the number to quote.
