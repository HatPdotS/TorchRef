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

LBFGS crystallographic refinement. Defaults to the maximum-likelihood X-ray
target with a cross-validated Luzzati σ_A term (``ml``) and separated
XYZ-then-ADP optimisation.

.. code-block:: bash

   torchref.refine -m model.pdb -sf reflections.mtz -o output_dir/

Produces refined coordinates (PDB and/or mmCIF), structure factors (MTZ),
and a ``refinement_history.json`` log.

**Key options:**

* ``-n`` / ``--n-cycles`` number of macro cycles (default 5)
* ``--mode`` ``separate`` (separated XYZ then ADP, default) or ``everything``
  (joint XYZ+ADP)
* ``--xray-mode`` ``ml`` (default; maximum-likelihood with a cross-validated
  Luzzati σ_A term), ``bhattacharyya``, ``ls``, ``ls_wunit_k1``, ``gaussian``
  (the legacy ``ml_sigmaa`` is accepted as a deprecated alias for ``ml``)
* ``--sigma-m-scale`` global multiplier on σ_m for the Bhattacharyya target
* ``--dmin`` resolution cutoff
* ``--device`` ``cpu`` / ``cuda``
* ``-v`` verbose

:API: :mod:`torchref.cli.refine`

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
       -dm dark.pdb -lm light.pdb \
       -dsf dark.mtz -lsf light.mtz \
       --fraction 0.37 -o output/

**Key options:** ``-dm``/``--dark-model``, ``-lm``/``--light-model``,
``-dsf``/``--dark-structure-factor``, ``-lsf``/``--light-structure-factor``,
``--fraction`` (light-state population fraction, singular),
``--weight-schedule`` annealing schedule (default ``5,3,2``),
``-n``/``--n-cycles`` macro-cycles.

:API: :mod:`torchref.cli.collection_difference_refine`

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
       -dsf dark.mtz -lsf light.mtz \
       -dm dark.pdb -lm light.pdb

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
       -dm dark.pdb -lm light.pdb \
       -dsf dark.mtz -lsf light.mtz \
       --fraction 0.37 -o results.mtz

:API: :mod:`torchref.cli.phased_difference_map`
