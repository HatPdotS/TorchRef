Command-Line Tools
==================

TorchRef provides several command-line tools installed as console scripts.
After installation (``pip install torchref``), they are available directly
in your shell.

.. contents:: Commands
   :local:
   :depth: 1

Standard Refinement
-------------------

``torchref.refine``
~~~~~~~~~~~~~~~~~~~

Basic LBFGS crystallographic refinement.

.. code-block:: bash

   torchref.refine -s model.pdb -f data.mtz -o output/

Produces refined coordinates (PDB), structure factors (MTZ), and a
``refinement_history.json`` log.

**Key options:** ``-n`` number of cycles, ``--max-res`` resolution cutoff,
``--device`` (cpu/cuda), ``-w`` JSON weight file, ``-v`` verbose.

:API: :mod:`torchref.cli.refine`

``torchref.refine-static``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Refinement with fixed component weights (default: xray=1.0, geometry=10.0,
adp=5.0).  Useful as a baseline for comparison.

.. code-block:: bash

   torchref.refine-static -s model.pdb -f data.mtz -o output/

**Key options:** ``-w`` JSON file to override weights.

:API: :mod:`torchref.cli.refine_everything_static`

``torchref.refine-hyper``
~~~~~~~~~~~~~~~~~~~~~~~~~

Refinement with user provided hyperparameters.  Uses
``ComponentWeighting`` (XrayScaleWeighting + TargetOffsetWeighting +
OverfittingWeighting) with pre-tuned parameters.

.. code-block:: bash

   torchref.refine-hyper -s model.pdb -f data.mtz -o output/

**Key options:** ``-w`` path to custom hyperparameter JSON, or ``"none"``
to skip.

:API: :mod:`torchref.cli.refine_everything_hyperparameters`

``torchref.refine-policy``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Refinement with a trained neural-network policy that predicts component
weights from the current refinement state (AWR-trained).

.. code-block:: bash

   torchref.refine-policy -s model.pdb -f data.mtz --policy policy.pt -o output/

**Key options:** ``--policy`` path to checkpoint, ``--sample`` enable
stochastic sampling, ``--temperature`` sampling temperature.

:API: :mod:`torchref.cli.refine_everything_policy`

``torchref.refine-random-weights``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refinement with randomly sampled component weights (log-normal
distributions).  Used to generate diverse training trajectories for
policy-network training.

.. code-block:: bash

   torchref.refine-random-weights -s model.pdb -f data.mtz -o output/ --seed 42

:API: :mod:`torchref.cli.refine_everything_random_weights`

Difference Refinement
---------------------

``torchref.difference-refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Difference refinement for time-resolved crystallography.  Refines a mixed
model (dark + light state) against dark and light reflection data using
amplitude-only difference targets with geometry, ADP, and maximum-likelihood
restraints.

.. code-block:: bash

   torchref.difference-refine \
       --dark-pdb dark.pdb --light-pdb light.pdb \
       --dark-mtz dark.mtz --light-mtz light.mtz \
       --fractions 0.63,0.37 -o output/

**Key options:** ``--weight-schedule`` annealing schedule (e.g. ``5,3,2``),
``-n`` macro-cycles.

:API: :mod:`torchref.cli.difference_refine`

Map & Validation Utilities
--------------------------

``torchref.mtz2map``
~~~~~~~~~~~~~~~~~~~~

Convert MTZ map coefficients to a CCP4 map file.  Reads amplitude and phase
columns, expands to P1, and computes a real-space map via FFT.

.. code-block:: bash

   torchref.mtz2map -f refined.mtz -F 2FOFCWT -P PH2FOFCWT -o map.ccp4

**Key options:** ``--high-res``, ``--low-res`` resolution limits,
``--gridsize`` override, ``-n`` normalize to sigma units.

:API: :mod:`torchref.cli.mtz2map`

``torchref.validate-ded``
~~~~~~~~~~~~~~~~~~~~~~~~~

Validate difference electron density by correlating DFo and DFc maps.
Computes real-space correlations and resolution-binned reciprocal-space CC.

.. code-block:: bash

   torchref.validate-ded \
       --dark-mtz dark.mtz --light-mtz light.mtz \
       --dark-pdb dark.pdb --light-pdb light.pdb

**Key options:** ``--fraction``, ``--selection`` (Phenix-style atom
selection), ``--mask-radius``, ``--n-bins``.

:API: :mod:`torchref.cli.validate_ded`

``torchref.phased-difference-map``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute phased difference and extrapolated map coefficients without
refinement.  Uses the same pipeline as ``torchref.difference-refine`` but
the input models are kept as-is.

.. code-block:: bash

   torchref.phased-difference-map \
       --dark-pdb dark.pdb --light-pdb light.pdb \
       --dark-mtz dark.mtz --light-mtz light.mtz \
       --fractions 0.63,0.37 -o output/

:API: :mod:`torchref.cli.phased_difference_map`
