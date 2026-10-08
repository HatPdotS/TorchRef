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
* ``--hydrogens {keep,add,strip}`` what loading does with the model's hydrogens:
  keep the ones the file has (default), also generate the missing ones (waters
  included), or strip them all
* ``--hydrogen-mode {atoms,riding}`` refine hydrogens as ordinary atoms (default) or
  let them ride on their parent heavy atoms; ``riding`` with ``--hydrogens strip`` is
  an error
* ``--hydrogens-in-xray`` / ``--no-hydrogens-in-xray`` include hydrogen atoms in the
  structure-factor calculation (default on). Off keeps them in the restraints only;
  the bulk-solvent mask is built from heavy atoms in either case
* ``--mode`` ``separate`` (separated XYZ then ADP, default) or ``everything``
  (joint XYZ+ADP)
* ``--xray-mode`` one of ``ml`` (default; Read MLF at variance ε·β, conditional
  mean α·``|F_calc|``), ``ml_noalpha`` (the same with the mean coupling fixed at 1),
  ``ml_full`` (marginalises the measurement error rather than inflating the
  variance; ~4× the cost), ``nll_beta`` (the Gaussian large-signal limit of
  ``ml`` — diagnostic), ``nll`` (Gaussian weighted by σ_obs only, no model-error
  term), ``nll_i`` (as ``nll`` but on the observed *intensities*, skipping the
  French–Wilson conversion), ``ls`` (unit-weight least squares) or ``ls_wunit_k1``
  (Phenix-style, own global scale). ``--help`` lists them from the taxonomy table
  itself, which is authoritative.
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
* ``--wavelength`` Å, the wavelength the data were collected at. Given, the model
  includes anomalous f'/f'' and reads F(+)/F(-) as Bijvoet pairs; without it (or
  with ``0``) there is no anomalous scattering and the read is Friedel-merged
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
``--n-cycles`` macro-cycles (default 3), ``--difference-target {difference,difference_sd}``
(the difference row the schedule drives; default ``difference``), ``--ded-weight`` and
``--difference-gamma`` for the difference MTZ (see ``torchref.difference-map``).

:API: :mod:`torchref.cli.collection_difference_refine`

Data Utilities
--------------

``torchref.uniform-rfree``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Give any number of structure-factor files (MTZ or SF-mmCIF) of one crystal
form a single shared R-free set, for example before refining and
difference-refining the dark and light datasets of a time-resolved experiment.
Run it before ``torchref.refine`` / ``torchref.difference-refine`` so that no
reflection is free in one dataset and work in another.

- **Existing flags are kept by default.** If any input already has an R-free
  column (CCP4 ``0 = free``, Phenix ``1 = free`` or mmCIF ``status``), its
  free set is inherited and extended, at its own fraction, to the reflections
  it lacks. The source is the first input with flags, or the file named with
  ``--reference``. Without any flags a new set is generated. ``--fresh`` always
  generates one. ``--free-fraction`` and ``--max-free`` size a new set, so they
  are refused while a set is inherited. Replacing a set that a model was
  already refined against makes that model's R-free meaningless.
- **Mixed resolution cutoffs.** If the reference stops short of the data
  resolution, a warning is printed and the higher-resolution shells, plus any
  gaps in the reference, are generated at the reference's free fraction,
  stratified by shell. By default these shells are seeded with a hash of the
  reference's free/work partition. Every extension of the same free set is
  therefore identical, whatever the file format, row order, flag convention
  or the other inputs. A dataset cut at lower resolution gets exactly the
  matching subset of the shared flags. Datasets with fewer than 500 free
  reflections are reported.
- **Reproducibility.** New flags are drawn on the complete reciprocal ASU
  (Friedel mates and symmetry equivalents share a flag), with exactly the free
  fraction in every resolution shell of ``--shell-size`` reflections. Each flag
  depends only on cell, space group, fraction and ``--seed``. The cell is that
  of the reference, or of the first input when there is none, and the flags
  are sensitive to it at the 1e-5 level. To give a dataset added later the same
  flags, run it together with an already flagged file (inherited by default)
  or name that file with ``--reference``. A separate ``--fresh`` run on a
  dataset with its own cell gives a different free set.
- **Excluded reflections** (MTZ flag ``-1``, mmCIF ``status x``) stay excluded
  in the file that marked them, as ``-1`` / ``x``. They are not copied to the
  other files.
- **Output.** MTZ output keeps every input column; existing flag columns are
  replaced unless ``--keep-old-flags`` retains them as ``<name>_orig``. SF-mmCIF
  output retains supported mapped measurements and numeric free-flag values,
  including CCP4 work-set numbers. Unsupported columns, including saved original
  flag columns, cause an error before writing the CIF; use MTZ for these columns.
  Standard CIF aliases may rename measurements (for example ``I`` to ``IMEAN``).

.. code-block:: bash

   # do the existing free sets agree? (writes nothing; exit code 2 if not)
   torchref.uniform-rfree dark.mtz light_*.mtz --check

   # shared free set (inherited if any input has one), MTZ and mmCIF output
   torchref.uniform-rfree dark.mtz light_*.mtz --format mtz cif -o flagged/

   # new set capped at 2000 free reflections, all light data scaled onto dark
   torchref.uniform-rfree dark.mtz light_*.mtz --fresh --max-free 2000 \
       --scale --scale-reference dark -o flagged/

With ``--scale`` the datasets are scaled together (overall plus anisotropic, on
work reflections only) with ``DatasetCollection.scale``. The fitted factor
multiplies the observed amplitude columns and its square multiplies the
observed intensity columns; map coefficients such as ``FWT``/``PHWT`` are left
alone. By default everything goes onto the shared consensus scale;
``--scale-reference`` leaves one input unchanged instead.

**Key options:** ``--check``, ``--reference {auto,FILE}``/``--fresh``,
``--reference-column``, ``--free-fraction`` (new sets, default 0.05),
``--max-free`` (new sets; because the cap depends on resolution, the run prints
the ``--free-fraction`` that reproduces it), ``--seed`` (default: 0 for a new
set, the reference hash when extending), ``--shell-size``,
``--format {mtz,cif}``, ``--suffix``, ``--keep-old-flags``,
``--length-tol``/``--angle-tol``/``--force`` for the cell/space-group check.

:API: :mod:`torchref.cli.uniform_rfree`

Map & Validation Utilities
--------------------------

``torchref.mtz2map``
~~~~~~~~~~~~~~~~~~~~

Convert MTZ map coefficients to a CCP4 map file.  Reads amplitude and phase
columns, expands to P1, and computes a real-space map via FFT.

.. code-block:: bash

   torchref.mtz2map -sf refined.mtz -csf 2FOFCWT -cphi PH2FOFCWT -o map.ccp4
   torchref.mtz2map -sf diff.mtz -csf dF -cw W_Q -cphi PHDELWT -o diff.ccp4
   torchref.mtz2map -sf diff.mtz -csf dF -cw W_Q -cphi PHDELWT --units electrons -o diff_e.ccp4

**Key options:** ``--dmin``/``--dmax`` resolution limits, ``--gridsize`` override,
``-cw``/``--column-weight`` multiplies the amplitudes by a weight column before the
FFT, ``--units {sigma,electrons,raw}`` (``sigma``, the default, gives zero mean and
unit standard deviation; ``electrons`` gives e/A^3 as
``(1/V) sum_h F(h) exp(-2 pi i h.x)`` with the amplitudes divided by the
``-ck``/``--column-scale`` factor, ``KSCALE`` by default). ``-n`` is the deprecated
alias of ``--units sigma``/``raw``.

:API: :mod:`torchref.cli.mtz2map`

``torchref.validate-ded``
~~~~~~~~~~~~~~~~~~~~~~~~~

Validate difference electron density by correlating dFo and dFc maps.
Computes real-space correlations and resolution-binned reciprocal-space CC.

.. code-block:: bash

   torchref.validate-ded \
       -dsf dark.mtz -lsf light.mtz \
       -dm dark.pdb -lm light.pdb

**Key options:** ``--fraction``, ``--selection`` (Phenix-style atom
selection), ``--mask-radius``, ``--n-bins``, ``--ded-weight`` (the headline weight
scheme; every scheme is also reported side by side, real-space in each mask and
reciprocal-space overall, as the ``by_weight`` block of the JSON and a table in the
summary, and a ``q`` fallback to inverse variance is recorded under ``weights``).

:API: :mod:`torchref.cli.validate_ded`

``torchref.difference-map``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute difference and extrapolated map coefficients without refinement.
Uses the same pipeline as ``torchref.difference-refine`` but the input
models are kept as-is.

The default output is the difference map: the amplitude difference ``dF``/``SIGdF``
on the **dark** model's phases ``PHDELWT``, with one mean-one weight column per
registered scheme beside it -- ``W_Q``, the q-weight (the default), and ``W_InVa``,
the inverse variance ``1/sigma^2`` -- and ``KSCALE``, the scaler's factor from model to observed
scale. This is the construction ``torchref.validate-ded`` correlates against. Build the
map with ``torchref.mtz2map -csf dF -cw W_Q -cphi PHDELWT``, adding
``--units electrons`` for e/A^3. It needs no light-state model, so ``-lm`` is optional:

.. code-block:: bash

   torchref.difference-map \
       -dm dark.pdb \
       -dsf dark.mtz -lsf light.mtz -o results.mtz

Supplying ``-lm`` (with ``--fraction``) adds the light state's amplitude and
phase and the extrapolated map ``FWT``/``PHWT``:

.. code-block:: bash

   torchref.difference-map \
       -dm dark.pdb -lm light.pdb \
       -dsf dark.mtz -lsf light.mtz \
       --fraction 0.37 -o results.mtz

``FWT``/``PHWT`` keep the standard labels so Coot auto-opens the map, but here they
are the extrapolated light-state map ``2*FEXT - Fc``, not a ``2mFo-DFc``. The file
records this: the columns sit in named MTZ datasets -- ``observed``, ``difference``,
``light_model``, ``extrapolated_light`` and, when written, ``two_moment`` -- so Coot's
column chooser shows ``/torchref/extrapolated_light/FWT``, and ``gemmi mtz`` prints a
history line per dataset. ``torchref.difference-refine`` writes the same file.

**Key options:** ``--ded-weight {q,inverse_variance,none}`` selects the scheme the
model-phased and two-moment difference columns carry (default ``q``; its fit uses the
intensity differences when the data carry ``I``/``SIGI``, and falls back to inverse
variance with a warning when too few reflections exist to fit); ``--difference-gamma``
fixes the dark-amplitude exponent of the difference power law instead of fitting it;
``--all-columns`` writes every alternative map coefficient and diagnostic -- the
model-phased difference, the two other extrapolations and the intensity block -- at
the cost of two further scale fits.

:API: :mod:`torchref.cli.difference_map`

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
