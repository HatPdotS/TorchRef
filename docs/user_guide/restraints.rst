Geometry Restraints
===================

Geometry restraints keep the model chemically reasonable during refinement.
:class:`~torchref.restraints.Restraints` (the exported alias of
``RestraintsNew``) builds and holds bond, angle, torsion, planarity, chirality,
and non-bonded (VDW) restraints.

Restraint Setup
---------------

You do not normally construct ``Restraints`` yourself. ``model.restraints`` is a
lazy property that builds them from the monomer library — fetched per monomer on
demand — on first access. Point it at extra CIF definitions *before* that first
access:

.. code-block:: python

   model.set_restraints_cif("ligand.cif")     # or a list of paths; chainable
   restraints = model.restraints              # built here, on first access

``Restraints.__init__`` takes a PDB DataFrame plus accessor callables
(``pdb, cif_path, xyz_fn, adp_fn, vdw_radii_fn, cell, spacegroup, links,
verbose``), not a model — that is what the lazy property assembles for you.

Residues for which no restraints could be built are frozen in ``xyz`` rather
than refined unrestrained, so a missing ligand definition shows up as an
immobile ligand, not as distorted geometry.

Links and Their Modifications
-----------------------------

The monomer library defines each amino acid **free**, not in-chain: ``ALA``
carries ``OXT`` and a protonated ``N``, with carboxylate and ammonium geometry.
Forming a bond to a neighbour therefore does more than add restraints across the
link — it also *changes* the residue's own. The ``chem_link`` table names a
modification per partner (``TRANS`` applies ``DEL-OXT`` to the residue donating
its C and ``DEL-HN1`` to the one donating its N; proline uses ``DEL-HNP``), and
TorchRef applies them when it builds the peptide links.

What that means for the numbers you will see: a linked residue is restrained to
``CA-C-O`` 120.6° and ``N-CA`` 1.453 Å, not the free 117.2° and ~1.48 Å, and its
``C-OXT`` bond, ``CA-C-OXT``/``O-C-OXT`` angles and carboxylate plane are gone.
The targets are chosen so the restraints close: the intra-residue ``CA-C-O``
plus the link's ``CA-C-N`` and ``O-C-N`` sum to exactly 360° around the planar
carbonyl carbon, and ``CA-N-H`` plus ``C-N-CA`` and ``C-N-H`` likewise around the
amide nitrogen.

Chain termini are left unmodified on purpose — a real C-terminus keeps its
``OXT`` and carboxylate geometry, a real N-terminus its ammonium — so a
structure's first and last residues legitimately carry different targets from
the ones in between.

Restraint Storage
-----------------

Restraints are reached through a nested-dict interface, ``[type][origin][field]``:

.. code-block:: python

   bond = restraints.restraints["bond"]["all"]
   bond["indices"]      # (N, 2) atom indices
   bond["references"]   # (N,)   ideal values
   bond["sigmas"]       # (N,)   restraint sigmas

   n_bonds = restraints.restraints["bond"]["all"]["indices"].shape[0]

- **Types** with an origin level: ``"bond"``, ``"angle"``, ``"torsion"``,
  ``"plane"``. Note ``"plane"`` in *storage* — the matching *target* and its
  ``stats()`` entry are called ``"planarity"``, so the two keys differ.
- **Origins** are where the restraint came from: ``"intra"``, ``"link"``,
  ``"peptide"``, ``"disulfide"``, and for torsions ``"phi"`` / ``"psi"`` /
  ``"omega"``.
- ``"all"`` is the merged origin the targets actually read. It is built lazily by
  ``restraints.cat_dict()``, so it is missing until a target has run once —
  targets guard with ``if "all" not in ...: self.cat_dict()`` and so should you.
  Bond and angle ``"all"`` merge every origin; torsion ``"all"`` merges only
  ``"intra"`` and ``"disulfide"``, because phi/psi carry no reference values and
  omega has its own target.
- **Flat types** with no origin level: ``"vdw"`` and ``"chiral"`` are indexed
  straight by field, ``restraints.restraints["vdw"]["indices"]``.
- **Fields** beyond the three above, where the restraint type has them:
  ``periods``, ``min_distances``.

The nesting is an accessor over a flat ``TensorDict`` keyed ``bond_all_indices``,
not a real dict — it supports ``[]``, ``keys()``, ``get()`` and ``in``, but
assigning a whole type (``restraints["bond"] = ...``) raises ``TypeError`` for
the nested types. A type absent from the model is absent from ``keys()``, so
probe with ``in`` before indexing.

Restraint Types
---------------

Restraints are negative log-likelihoods, not weighted sums of squares — the
``log σ`` term is kept so that the components are on a common scale and need no
hand-tuned inter-restraint weights. For bonds and angles that is a Gaussian NLL
in the deviation from ideal:

.. math::

   E = \sum_i \left[ \tfrac{1}{2}
       \left( \frac{q_i - q_i^{ideal}}{\sigma_i} \right)^2
       + \log \sigma_i + \tfrac{1}{2}\log 2\pi \right]

with :math:`q` the interatomic distance or the bond angle (angles in radians,
sigmas converted from the CIF's degrees). Torsions use a periodic von Mises NLL,
planarity restrains a group's out-of-plane deviations, chirality preserves
stereochemistry, and the non-bonded term is a steep PROLSQ-style repulsion
(:math:`E \sim \text{violation}^4`) applied to symmetry mates as well as to the
asymmetric unit.

Restraint Statistics
--------------------

Deviations come off the ``Restraints`` object; the summary statistics come off
the geometry *targets*, not off ``Restraints``:

.. code-block:: python

   # Raw per-restraint deviations and their sigmas
   deviations, sigmas = restraints.bond_deviations()
   deviations, sigmas = restraints.angle_deviations()

   # Summary statistics, keyed by component: bond, angle, torsion, planarity,
   # chiral, nonbonded, ramachandran
   stats = refinement.geometry_target.stats()
   stats["bond"]["rms_delta"].value   # RMSD from ideal
   stats["bond"]["rms_z"].value       # RMS Z-score
   stats["bond"]["n"].value           # number of restraints

``stats()`` values are :class:`~torchref.utils.stats.StatEntry` wrappers, not
floats. Their ``__repr__`` prints the bare value, so a print looks right while
arithmetic on the entry fails — take ``.value``.
