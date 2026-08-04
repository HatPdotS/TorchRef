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
* ``--xray-mode`` one of ``ml`` (default; Read MLF at variance ε·β, conditional
  mean α·``|F_calc|``), ``ml_noalpha`` (the same with the mean coupling fixed at 1),
  ``ml_full`` (marginalises the measurement error rather than inflating the
  variance; ~4× the cost), ``nll_beta`` (the Gaussian large-signal limit of
  ``ml`` — diagnostic), ``nll`` (Gaussian weighted by σ_obs only, no model-error
  term), ``ls`` (unit-weight least squares) or ``ls_wunit_k1`` (Phenix-style, own
  global scale). ``--help`` lists them from the taxonomy table itself.
* ``--sigma-a-max`` upper bound on the per-shell Luzzati σ_A (default 0.99)
* ``--no-shrink`` disable the per-shell σ_A stability shrinkage
* ``--adp-mode`` ``isotropic`` (default) or ``anisotropic``, the latter refining
  6 U components for ``--anisotropic-selection`` (default: non-water heavy
  atoms). Six parameters per atom overfits low-resolution data — check that the
  resolution supports it rather than taking the input model's ``ANISOU`` as
  permission
* ``--weights`` JSON overrides on the loss weights. Defaults are xray=1,
  geometry=0.2, adp=0.02, geometry/ramachandran=0. Weights are **hierarchical
  and multiplicative**: a component's effective weight is the product down its
  path, so ``geometry/ramachandran`` is scaled by ``geometry`` too. Re-enable
  Ramachandran with ``--weights '{"geometry/ramachandran": 1.0}'``
* ``--with-rigid-body`` run rigid-body first (``--rigid-body-iter``,
  ``--rigid-body-cutoffs``)
* ``--wavelength`` Å, for anomalous f'/f''. ``0`` disables anomalous refinement
  and forces a Friedel-merged read of the data
* ``--dmin`` resolution cutoff
* ``--output-format`` ``pdb`` / ``cif`` / ``both`` (default both)
* ``--device`` ``auto`` (default) / ``cpu`` / ``cuda``. ``auto`` picks CUDA only
  when a visible GPU passes the capability and VRAM checks
* ``-v`` ``0`` quiet / ``1`` normal / ``2`` detailed

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

Model Utilities
---------------

``torchref.add-metadata``
~~~~~~~~~~~~~~~~~~~~~~~~~

Add deposition metadata (REMARK 3 / PDBx refinement statistics) to an existing
PDB or mmCIF, for structures refined before the headers were written
automatically. Output format follows the ``-o`` extension.

.. code-block:: bash

   torchref.add-metadata -i refined.pdb -o deposit.cif \
       --title "..." --authors "..." --r-work 0.18 --r-free 0.21

**Key options:** ``--metadata`` (JSON in ``RefinementMetadata`` form, instead of
the individual flags), ``--title``, ``--authors``, ``--r-work``, ``--r-free``,
``--resolution-high``, ``--resolution-low``.

:API: :mod:`torchref.cli.add_metadata`

``torchref.strip-altlocs``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Strip alternate conformations from a PDB, keeping the highest-occupancy
conformer. Takes two positional arguments, not flags.

.. code-block:: bash

   torchref.strip-altlocs input.pdb output.pdb

:API: :mod:`torchref.cli.strip_altlocs`
