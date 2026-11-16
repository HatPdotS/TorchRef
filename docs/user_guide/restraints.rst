Geometry Restraints
===================

Geometry restraints keep the model chemically reasonable during refinement.
:class:`~torchref.topology.restraints.Restraints` builds and holds bond, angle, torsion,
planarity, chirality, and non-bonded (VDW) restraints.

Restraint Setup
---------------

You do not normally construct ``Restraints`` yourself. They live on the model's
context, ``model.ctx.restraints``, and ``model.restraints`` builds them from the
monomer library -- fetched per monomer on demand -- on first access. Point it at
extra CIF definitions *before* that first access:

.. code-block:: python

   model.ctx.set_cif_path("ligand.cif")       # or a list of paths
   restraints = model.restraints              # built here, on first access
   deviations, sigmas = restraints.bond_deviations(model.xyz())

Restraints hold no reference to the model: every evaluation takes the
coordinates (or B-factors) it scores, and the non-bonded pair list is rebuilt
from the coordinates the non-bonded target passes in. ``Restraints.__init__``
takes a node-only topology (``model.ctx.topology``, or
``Topology.from_table(table)``) and the coordinates to build over
(``topology, cif_path, xyz, cell, spacegroup, links, nonbonded,
peptide_links, disulfides``).

Residues for which no restraints could be built are frozen in ``xyz`` rather
than refined unrestrained, so a missing ligand definition shows up as an
immobile ligand, not as distorted geometry.

Restraints Without Coordinates
------------------------------

A model read from a file has its connectivity completed from its coordinates:
sequence neighbours are peptide-bonded when their C and N lie within 2 Å, and two
cysteines are bridged when their SG lie within 2.5 Å. That is right for a deposited
structure and wrong for coordinates that are not yet trustworthy -- a sample from a
structure generator, a prediction before refinement -- where a stretched bond would
silently drop its restraints. Two settings on the context take both decisions away
from the coordinates:

``peptide_links="sequence"``
    Every pair of sequence-adjacent peptide residues is bonded. A chain break has
    to show in the residue numbering.

``disulfides="open"``
    No cysteines are bonded. Instead every pair of cysteine sulfurs may approach to
    the disulfide bond length (and a sulfur to the other cysteine's CB to the
    matching 1-3 distance) without non-bonded repulsion, so a bridge closes where
    the data put the sulfurs together. An open bridge has no bond, angle or torsion
    restraint, so its S-S distance is set by the data alone and is less precise
    than a detected one.

:meth:`ModelContext.from_sequence <torchref.model.context.ModelContext.from_sequence>`
builds a context this way from chains of residues, with no atom table:

.. code-block:: python

   from torchref.model import ModelContext

   ctx, values = ModelContext.from_sequence(
       {"A": "MKVLAAGIVG", "W": ["HOH"] * 20},   # one-letter codes, or residue names
       cell, spacegroup,                          # optional
   )
   restraints = ctx.build_restraints()            # no coordinates needed
   deviations, sigmas = restraints.bond_deviations(xyz)

The atoms are the heavy atoms of each residue's monomer-library template, in
template order; ``ctx.topology.columns()`` gives the ``chain``, ``resseq`` and
``name`` of every row, which is what coordinates from elsewhere are gathered by.
``values`` holds NaN coordinates until they are filled in. A build without
coordinates carries the whole restraint graph and its ideal values but no
non-bonded pair list, which only coordinates can give; it is returned and not
stored on the context, so it can be cached and shared by every model of that
sequence. Three choices that coordinates normally make fall back to the
dictionaries in such a build: a sugar takes its template's first pucker, every
proline the trans Ramachandran surface (the omega restraint itself is a cis/trans
mixture either way), and the C-beta dihedrals are emitted for residues whose
template declares an L centre.

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

- **Types** with an origin level: ``"bond"``, ``"angle"``, ``"torsion"``.
  ``"plane"`` is keyed by plane size instead (``"4_atoms"``, ...; fields
  ``indices`` and ``sigmas``, no ``"all"``). Note ``"plane"`` in *storage* — the
  matching *target* and its ``stats()`` entry are called ``"planarity"``.
- **Origins** are where the restraint came from: ``"intra"``, ``"link"``,
  ``"peptide"``, ``"disulfide"``, and for torsions ``"cbeta"`` / ``"phi"`` /
  ``"psi"`` / ``"omega"``. ``"cbeta"`` holds the two C-beta dihedrals per
  amino acid (``N-C-CA-CB`` and ``C-N-CA-CB``, σ 2.5°, Phenix's targets), which
  fix the in-plane CB tilt that the CA angles and the chiral volume leave loose.
- ``"all"`` is the merged origin the targets actually read. It is built with the
  rest when the restraints are built; a type with nothing to merge (the torsions
  of a glycine-only model without hydrogens) has none, so probe with ``in``.
  Bond and angle ``"all"`` merge every origin; torsion ``"all"`` merges only
  ``"intra"``, ``"disulfide"`` and ``"cbeta"``, because phi/psi carry no
  reference values and omega has its own target.
- **Flat types** with no origin level: ``"vdw"`` and ``"chiral"`` are indexed
  straight by field, ``restraints.restraints["vdw"]["indices"]``.
- **Fields** beyond the three above, where the restraint type has them:
  ``periods``, ``min_distances``; chirals carry ``ideal_volumes`` instead of
  ``references``.

The nesting is a plain nested dict of tensors, assembled once at build time.
Its entries are views into the topology's edge blocks, re-sliced into a new dict
after every device or dtype move, so read ``restraints.restraints`` again after a
``.to()`` instead of keeping an old reference. A type absent from the model is
absent from the dict, so probe with ``in`` before indexing.

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
sigmas converted from the CIF's degrees). Torsions are measured with the IUPAC sign
the monomer library uses (the same as ``gemmi.calculate_dihedral``) and scored with
a periodic von Mises NLL, planarity restrains a group's out-of-plane deviations,
chirality preserves stereochemistry, and the non-bonded term is a steep PROLSQ-style
repulsion
(:math:`E \sim \text{violation}^4`) applied to symmetry mates as well as to the
asymmetric unit.

Not every torsion a dictionary lists is restrained. One screen,
:func:`~torchref.topology.builders.screen_torsions`, decides, and the restraint build logs at
DEBUG (``torchref.set_verbosity(2)``, CLI ``-v 2``) what it skipped per residue type and
why. Three kinds are skipped: those over backbone atoms only (N, CA, C, O, OXT), which would fight the Ramachandran term;
those with a zero sigma (the ``CONST_*`` aromatic-ring torsions, which the planes
already cover); and staggered (period-3) torsions about a bond in a ring of five or
fewer atoms. AceDRG writes its generic sp3–sp3 target (60°/180°) for ring bonds too,
and no pucker of a five-membered ring reaches it, so restraining proline's ring
torsions would strain every proline. Six-membered rings keep theirs: a chair's
torsions sit near the staggered values.

Restraint Statistics
--------------------

Deviations come off the ``Restraints`` object; the summary statistics come off
the geometry *targets*, not off ``Restraints``:

.. code-block:: python

   # Raw per-restraint deviations and their sigmas
   deviations, sigmas = restraints.bond_deviations(model.xyz())
   deviations, sigmas = restraints.angle_deviations(model.xyz())

   # Summary statistics, keyed by component: bond, angle, torsion, planarity,
   # chiral, nonbonded, ramachandran
   stats = refinement.geometry_target.stats()
   stats["bond"]["rms_delta"].value   # RMSD from ideal
   stats["bond"]["rms_z"].value       # RMS Z-score
   stats["bond"]["n"].value           # number of restraints

``stats()`` values are :class:`~torchref.utils.stats.StatEntry` wrappers, not
floats. Their ``__repr__`` prints the bare value, so a print looks right while
arithmetic on the entry fails — take ``.value``.
