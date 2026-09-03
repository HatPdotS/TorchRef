"""
Single source of truth for deposition metadata, renderable as either PDB
REMARK 3 header records or PDBx/mmCIF ``_refine`` category items.

The usual flow: build with ``from_refinement``, optionally
``from_pdb_file(...).merge(meta)`` to carry input headers through, then
``render_pdb_header()`` / ``render_cif_categories()``. The ``from_*`` extractors
swallow every exception, so a field that could not be read is simply absent --
never assume a populated result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from datetime import date
from typing import Any, Dict, List, Optional

#: Input records that describe the crystal, the sample and its chemistry.
#: Refinement moves atoms; it does not invalidate any of these, so they are
#: carried through. Listed in the order the PDB format mandates --
#: ``render_pdb_header`` emits them in this sequence and relies on it.
#: Deliberately absent: AUTHOR and JRNL (they credit the deposited entry, not
#: this refinement), HELIX/SHEET (nothing here computes secondary structure, so
#: they could only be stale or wrong), and HEADER/REVDAT/OBSLTE/CAVEAT/SPLIT
#: (assertions about a PDB entry that this file is not).
_KEEP_RECORDS = (
    "TITLE", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "MDLTYP",
    "DBREF", "DBREF1", "DBREF2", "SEQADV", "SEQRES", "MODRES",
    "HET", "HETNAM", "HETSYN", "FORMUL",
    "SSBOND", "LINK", "CISPEP", "SITE",
)

#: REMARK numbers dropped from the input. 2 is the resolution, which we
#: regenerate; 3 is the refinement, which is ours to write and whose statistics
#: describe a model we just replaced; 500 lists geometry outliers of those same
#: superseded coordinates.
_DROP_REMARKS = {2, 3, 500}

#: mmCIF categories carried over from an input CIF -- entity, sequence and
#: connectivity, i.e. the CIF counterpart of ``_KEEP_RECORDS``. The input's
#: ``_refine`` is NOT here: it is the previous program's statistics.
_KEEP_CIF_CATEGORIES = (
    "_entity", "_entity_poly", "_entity_poly_seq", "_struct_conn",
    "_chem_comp", "_struct_ref", "_struct_ref_seq", "_exptl",
)

#: Width of the label field in a REMARK 3 line, so the colons line up. Matches
#: the longest label we emit ("RESOLUTION RANGE HIGH (ANGSTROMS)").
_REMARK3_LABEL_WIDTH = 33


@dataclass
class RefinementMetadata:
    """Unified metadata for PDB headers and mmCIF categories.

    Each field maps to both a PDB REMARK 3 line and a PDBx/mmCIF ``_refine``
    item; only populated (non-None, non-empty) fields are rendered.

    Attributes
    ----------
    program, program_version, refinement_method : str
        Refinement program identification.
    target_function, optimizer : str
        The function minimised and how, e.g. ``"MAXIMUM LIKELIHOOD"`` and
        ``"LBFGS, 5 MACROCYCLES"``. Rendered only when set.
    resolution_high, resolution_low : float, optional
        Resolution limits ``d_min`` / ``d_max`` in Angstroms.
    n_reflections_work, n_reflections_test, n_reflections_all : int, optional
        Reflection counts; ``percent_free`` is the test-set percentage.
    r_work, r_free : float, optional
        Working- and free-set R-factors.
    b_mean_overall, b_min, b_max : float, optional
        Atomic B-factor statistics in A**2.
    rmsd_bond_lengths, rmsd_bond_angles : float, optional
        Geometry deviations from ideal (Angstroms, degrees).
    n_atoms_total, n_atoms_protein, n_atoms_solvent : int, optional
        Model atom counts.
    solvent_model_ksol, solvent_model_bsol : float, optional
        Bulk-solvent scale, and the equivalent single ``B`` for the fitted falloff.
    cell, spacegroup
        Unit cell ``[a, b, c, alpha, beta, gamma]`` and space-group name.
    title, authors
        Structure title and author names.
    starting_model : str, optional
        Input model this refinement started from (path or PDB ID).
    rfree_selection : str, optional
        Where the free-set flags came from. Filled from
        ``ReflectionData.rfree_source``, so the values are that field's:
        ``"MTZReader FreeR"`` for flags read from the input, or
        ``"Generated (resolution-binned, ASU-grouped, seed N)"`` for a draw this
        run made. Two refinements with different free sets have incomparable
        R-free values, which is why it is recorded rather than inferred.
    output_remarks : str
        Author-supplied free text. Rendered only when non-empty.
    software_chain : list of dict
        Programs applied before this refinement, read from an input mmCIF's
        ``_software`` loop, so ours appends to the chain instead of erasing it.
    passthrough_pdb_remarks, passthrough_pdb_records, passthrough_cif_categories
        Surviving REMARK lines, structural records keyed by record name, and
        mmCIF category items carried over from an input file.
    """

    # Program identification
    program: str = "TORCHREF"
    program_version: str = ""
    refinement_method: str = ""  # e.g. "difference-refine", "LBFGS"
    target_function: str = ""    # e.g. "MAXIMUM LIKELIHOOD"
    optimizer: str = ""          # e.g. "LBFGS, 5 MACROCYCLES"

    # Resolution
    resolution_high: Optional[float] = None  # d_min in Angstroms
    resolution_low: Optional[float] = None  # d_max in Angstroms

    # Reflection counts
    n_reflections_work: Optional[int] = None
    n_reflections_test: Optional[int] = None
    n_reflections_all: Optional[int] = None
    percent_free: Optional[float] = None

    # R-factors
    r_work: Optional[float] = None
    r_free: Optional[float] = None

    # B-factor statistics
    b_mean_overall: Optional[float] = None
    b_min: Optional[float] = None
    b_max: Optional[float] = None

    # Geometry deviations
    rmsd_bond_lengths: Optional[float] = None  # Angstroms
    rmsd_bond_angles: Optional[float] = None  # degrees

    # Model contents
    n_atoms_total: Optional[int] = None
    n_atoms_protein: Optional[int] = None
    n_atoms_solvent: Optional[int] = None

    # Solvent model
    solvent_model_ksol: Optional[float] = None
    solvent_model_bsol: Optional[float] = None

    # Cell and spacegroup
    cell: Optional[List[float]] = None  # [a, b, c, alpha, beta, gamma]
    spacegroup: Optional[str] = None

    # Free-form fields
    title: str = ""
    authors: List[str] = field(default_factory=list)

    # Provenance
    starting_model: Optional[str] = None
    rfree_selection: Optional[str] = None

    # Author-supplied free text, rendered as REMARK 3 OTHER REFINEMENT REMARKS
    # and _refine.details. Never generated -- if the author has nothing to say,
    # the field stays empty and neither is emitted.
    output_remarks: str = ""

    # Programs that touched the model before us, read from the input's
    # _software loop so ours can be appended rather than replacing the chain.
    software_chain: List[Dict[str, str]] = field(default_factory=list)

    # Pass-through from the input file: surviving REMARKs, structural records
    # keyed by record name (see _KEEP_RECORDS), and mmCIF categories.
    passthrough_pdb_remarks: List[str] = field(default_factory=list)
    passthrough_pdb_records: Dict[str, List[str]] = field(default_factory=dict)
    passthrough_cif_categories: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary, dropping None values."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, list) and len(val) == 0:
                continue
            if isinstance(val, dict) and len(val) == 0:
                continue
            if isinstance(val, str) and val == "" and f.name not in ("program",):
                continue
            d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RefinementMetadata:
        """Reconstruct from a dictionary (inverse of ``to_dict``)."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    # ------------------------------------------------------------------ #
    #  Construction from refinement
    # ------------------------------------------------------------------ #

    @classmethod
    def from_refinement(cls, refinement) -> RefinementMetadata:
        """Extract metadata from a completed Refinement, from ``get_rfactor()``,
        ``collect_metrics()`` and the reflection data. Every statistic is
        best-effort: anything unavailable is left unset, silently.
        """
        import torch
        from torchref import __version__

        meta = cls(program_version=__version__)

        # --- R-factors (from scaler, already computed) ---
        try:
            rwork, rfree = refinement.get_rfactor()
            meta.r_work = float(rwork.item() if hasattr(rwork, "item") else rwork)
            meta.r_free = float(rfree.item() if hasattr(rfree, "item") else rfree)
        except Exception:
            pass

        # --- Resolution from reflection data ---
        try:
            rd = refinement.reflection_data
            if rd.resolution is not None:
                meta.resolution_high = float(rd.resolution.min())
                meta.resolution_low = float(rd.resolution.max())
        except Exception:
            pass

        # --- Reflection counts ---
        try:
            rd = refinement.reflection_data
            with torch.no_grad():
                hkl, fobs, sigma, rfree_flags = rd()
            n_all = len(fobs)
            n_test = int(rfree_flags.sum().item()) if rfree_flags.dtype == torch.bool else int((~rfree_flags.bool()).sum().item())
            n_work = n_all - n_test
            # In torchref, rfree_flags=True means WORK set
            n_work = int(rfree_flags.sum().item())
            n_test = n_all - n_work
            meta.n_reflections_all = n_all
            meta.n_reflections_work = n_work
            meta.n_reflections_test = n_test
            meta.percent_free = 100.0 * n_test / n_all if n_all > 0 else None
        except Exception:
            pass

        # --- B-factor statistics from model ---
        try:
            model = refinement.model
            model.update_pdb()
            pdb = model.pdb
            bvals = pdb["tempfactor"]
            meta.b_mean_overall = float(bvals.mean())
            meta.b_min = float(bvals.min())
            meta.b_max = float(bvals.max())
        except Exception:
            pass

        # --- Geometry deviations (silently skip if no restraints) ---
        try:
            model = refinement.model
            if model.ctx.initialized and model._restraints is not None:
                restraints = model.restraints
                if hasattr(restraints, "bond_deviations"):
                    with torch.no_grad():
                        bond_devs, _ = restraints.bond_deviations()
                        meta.rmsd_bond_lengths = float(
                            torch.sqrt((bond_devs**2).mean())
                        )
                if hasattr(restraints, "angle_deviations"):
                    with torch.no_grad():
                        angle_devs, _ = restraints.angle_deviations()
                        meta.rmsd_bond_angles = float(
                            torch.sqrt((angle_devs**2).mean())
                        )
        except Exception:
            pass

        # --- Atom counts ---
        try:
            pdb = refinement.model.pdb
            meta.n_atoms_total = len(pdb)
            protein_mask = pdb["ATOM"] == "ATOM"
            meta.n_atoms_protein = int(protein_mask.sum())
            solvent_mask = pdb["ATOM"] == "HETATM"
            meta.n_atoms_solvent = int(solvent_mask.sum())
        except Exception:
            pass

        # --- Solvent model parameters ---
        # PDB REMARK 3 and mmCIF carry k_sol and a single solvent B; the fitted falloff
        # ``k_sol exp(-ln2 (ss/ss_half)^n)`` has no B, so the reported one is back-fitted
        # from the curve over this dataset's own reflections.
        scaler = getattr(refinement, "scaler", None)
        sm = getattr(scaler, "solvent", None)
        if sm is not None:
            try:
                meta.solvent_model_ksol = float(sm.k_solvent().detach())
                meta.solvent_model_bsol = sm.b_solvent_equivalent(scaler._s_half_sq)
            except (AttributeError, RuntimeError, ValueError) as exc:
                if getattr(refinement, "verbose", 0) > 0:
                    print(f"Could not record solvent parameters: {exc}")

        # --- Provenance ---
        # Both are recorded on the objects already; the header just had no way
        # to say them. rfree_source in particular is what distinguishes a test
        # set read from the input file from one this run drew itself, and hence
        # whether R-free is comparable to the number the input reported.
        try:
            input_file = refinement.model.ctx.input_file
            if input_file:
                meta.starting_model = str(input_file)
        except Exception:
            pass

        try:
            source = refinement.reflection_data.rfree_source
            if source:
                meta.rfree_selection = source
        except Exception:
            pass

        # --- Cell and spacegroup ---
        try:
            model = refinement.model
            if model.cell is not None:
                meta.cell = [float(x) for x in model.cell.data.tolist()]
            if model.spacegroup is not None:
                meta.spacegroup = model.spacegroup.hm
        except Exception:
            pass

        return meta

    # ------------------------------------------------------------------ #
    #  Construction from input files (pass-through)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pdb_file(
        cls, filepath: str, *, supersede_refinement: bool = True
    ) -> RefinementMetadata:
        """Extract the carry-through header of an existing PDB file.

        Captures TITLE, the structural records in ``_KEEP_RECORDS`` and every
        REMARK except those in ``_DROP_REMARKS``. AUTHOR and JRNL are absent
        from ``_KEEP_RECORDS`` and so never collected: they credit whoever
        deposited the entry, not this refinement.

        The input's REMARK 3 is dropped rather than carried, which is the whole
        point -- a refined file that repeats the previous program's R-factors
        alongside its own asserts two different refinements at once.

        Parameters
        ----------
        supersede_refinement : bool, optional
            Whether this file's refinement is about to be replaced -- the
            default, and the case for refinement output. Pass ``False`` when
            annotating a file without re-refining it: nothing supersedes the
            existing REMARK 3 or AUTHOR records then, and dropping them would
            lose statistics and credit that are still accurate.
        """
        meta = cls()
        remarks: List[str] = []
        records: Dict[str, List[str]] = {}
        try:
            with open(filepath, "r") as f:
                for line in f:
                    record = line[:6].strip()
                    # MODEL as well as the atoms: it opens the coordinate
                    # section in a multi-model file.
                    if record in ("ATOM", "HETATM", "MODEL"):
                        break
                    if record == "AUTHOR" and not supersede_refinement:
                        for author in line[10:].strip().split(","):
                            if author.strip():
                                meta.authors.append(author.strip())
                    elif record == "TITLE":
                        # Kept on `title` rather than as a raw record so the
                        # --title override has something to override.
                        title_text = line[10:].strip()
                        meta.title = (
                            meta.title + " " + title_text if meta.title else title_text
                        )
                    elif record == "REMARK":
                        try:
                            number = int(line[7:10])
                        except ValueError:
                            # A REMARK with no parsable number is not one we can
                            # judge, so leave it out.
                            continue
                        if supersede_refinement and number in _DROP_REMARKS:
                            continue
                        remarks.append(line.rstrip("\n"))
                    elif record in _KEEP_RECORDS:
                        records.setdefault(record, []).append(line.rstrip("\n"))
            meta.passthrough_pdb_remarks = remarks
            meta.passthrough_pdb_records = records
        except Exception:
            pass
        return meta

    @classmethod
    def from_cif_file(cls, filepath: str) -> RefinementMetadata:
        """Extract the carry-through metadata of an existing mmCIF file.

        Captures ``_struct.title``, the entity/sequence/connectivity categories
        in ``_KEEP_CIF_CATEGORIES``, and the ``_software`` loop -- the last so
        this refinement can append itself to the chain of programs rather than
        presenting itself as the only one.

        The input's ``_refine`` category is deliberately NOT captured. It holds
        the previous program's R-factors and resolution, which this refinement
        supersedes; carrying them forward is the mmCIF form of the duplicated
        REMARK 3 that ``from_pdb_file`` used to produce.
        """
        meta = cls()
        try:
            import gemmi

            doc = gemmi.cif.read(filepath)
            block = doc[0]

            title = block.find_value("_struct.title")
            if title and title != "?":
                meta.title = gemmi.cif.as_string(title)

            # Prior programs, so ours lands at max(ordinal) + 1.
            chain: List[Dict[str, str]] = []
            table = block.find(
                "_software.",
                ["name", "?version", "?classification", "?pdbx_ordinal",
                 "?description"],
            )
            for row in table:
                name = gemmi.cif.as_string(row[0])
                if not name or name in ("?", "."):
                    continue
                entry = {"name": name}
                for idx, key in ((1, "version"), (2, "classification"),
                                 (3, "pdbx_ordinal"), (4, "description")):
                    if row.has(idx):
                        val = gemmi.cif.as_string(row[idx])
                        if val and val not in ("?", "."):
                            entry[key] = val
                chain.append(entry)
            meta.software_chain = chain

            # Entity, sequence and connectivity categories, in whichever form
            # the input used them (loop or key-value).
            cats: Dict[str, Any] = {}
            for item in block:
                if item.loop is not None:
                    tags = list(item.loop.tags)
                    if tags[0].split(".")[0] not in _KEEP_CIF_CATEGORIES:
                        continue
                    cats[tags[0].split(".")[0]] = {
                        tag: [
                            item.loop[r, c] for r in range(item.loop.length())
                        ]
                        for c, tag in enumerate(tags)
                    }
                elif item.pair is not None:
                    tag, val = item.pair
                    category = tag.split(".")[0]
                    if category in _KEEP_CIF_CATEGORIES:
                        cats.setdefault(category, {})[tag] = val
            meta.passthrough_cif_categories = cats

        except Exception:
            pass
        return meta

    # ------------------------------------------------------------------ #
    #  Merge
    # ------------------------------------------------------------------ #

    def merge(self, other: RefinementMetadata) -> RefinementMetadata:
        """Merge *other* into self, returning a NEW instance (neither is mutated).

        ``other`` wins for any field that is non-None, non-``""`` and non-``[]``;
        the pass-through / authors / custom-remark containers are concatenated
        rather than replaced.
        """
        merged = RefinementMetadata()
        for f in fields(RefinementMetadata):
            self_val = getattr(self, f.name)
            other_val = getattr(other, f.name)

            # For pass-through containers, combine
            if f.name == "passthrough_pdb_remarks":
                merged_list = list(self_val) + [
                    r for r in other_val if r not in self_val
                ]
                setattr(merged, f.name, merged_list)
            elif f.name == "passthrough_cif_categories":
                merged_dict = dict(self_val)
                merged_dict.update(other_val)
                setattr(merged, f.name, merged_dict)
            elif f.name == "authors":
                merged_authors = list(self_val) + [
                    a for a in other_val if a not in self_val
                ]
                setattr(merged, f.name, merged_authors)
            elif f.name == "passthrough_pdb_records":
                merged_records = {k: list(v) for k, v in self_val.items()}
                for key, rows in other_val.items():
                    existing = merged_records.setdefault(key, [])
                    existing.extend([r for r in rows if r not in existing])
                setattr(merged, f.name, merged_records)
            elif f.name == "software_chain":
                # Keyed on name+version: re-refining with the same build should
                # not add a second identical link to the chain.
                merged_chain = list(self_val)
                seen = {(e.get("name"), e.get("version")) for e in merged_chain}
                for entry in other_val:
                    key = (entry.get("name"), entry.get("version"))
                    if key not in seen:
                        merged_chain.append(entry)
                        seen.add(key)
                setattr(merged, f.name, merged_chain)
            else:
                # other takes precedence if non-None and non-default
                if other_val is not None and other_val != "" and other_val != []:
                    setattr(merged, f.name, other_val)
                else:
                    setattr(merged, f.name, self_val)
        return merged

    # ------------------------------------------------------------------ #
    #  PDB header rendering
    # ------------------------------------------------------------------ #

    def render_pdb_header(self) -> str:
        """Render the header as PDB records, ready to precede CRYST1.

        Records come out in the order the PDB format mandates -- TITLE and the
        entry-level records, then REMARKs in ascending numeric order with ours
        slotted in at 3, then sequence, chemistry and connectivity. Only the
        REMARK 3 block is generated; everything else is either carried through
        from the input or supplied by the caller.
        """
        lines: List[str] = []
        records = self.passthrough_pdb_records

        def _emit(*names: str) -> None:
            for name in names:
                lines.extend(records.get(name, []))

        # -- entry level -------------------------------------------------- #
        if self.title:
            _wrap_pdb_record(lines, "TITLE", self.title)
        _emit("COMPND", "SOURCE", "KEYWDS", "EXPDTA", "MDLTYP")

        # Only ever what the caller set explicitly: authors are not inherited
        # from the input, since they credit that deposition and not this run.
        if self.authors:
            _wrap_pdb_record(lines, "AUTHOR", ", ".join(self.authors))

        # -- REMARKs, ascending, ours at 3 -------------------------------- #
        def _remark_number(line: str) -> int:
            try:
                return int(line[7:10])
            except ValueError:
                return 0

        passthrough = sorted(self.passthrough_pdb_remarks, key=_remark_number)
        lines.extend(r for r in passthrough if _remark_number(r) < 3)
        lines.extend(self._render_remark3())
        lines.extend(r for r in passthrough if _remark_number(r) > 3)

        # -- sequence, chemistry, connectivity ---------------------------- #
        _emit("DBREF", "DBREF1", "DBREF2", "SEQADV", "SEQRES", "MODRES",
              "HET", "HETNAM", "HETSYN", "FORMUL",
              "SSBOND", "LINK", "CISPEP", "SITE")

        return "\n".join(lines) + "\n"

    def _render_remark3(self) -> List[str]:
        """Build the REMARK 3 block: this refinement, and only this one."""
        lines: List[str] = []
        lines.append("REMARK   3")
        lines.append("REMARK   3 REFINEMENT.")
        _ident(lines, "PROGRAM", f"{self.program} {self.program_version}".strip())
        if self.refinement_method:
            _ident(lines, "METHOD", self.refinement_method)
        if self.target_function:
            _ident(lines, "TARGET", self.target_function)
        if self.optimizer:
            _ident(lines, "OPTIMIZER", self.optimizer)
        lines.append("REMARK   3")

        lines.append("REMARK   3  DATA USED IN REFINEMENT.")
        _remark3(lines, "RESOLUTION RANGE HIGH (ANGSTROMS)", self.resolution_high, ".2f")
        _remark3(lines, "RESOLUTION RANGE LOW  (ANGSTROMS)", self.resolution_low, ".2f")
        _remark3(lines, "NUMBER OF REFLECTIONS", self.n_reflections_all, "d")
        lines.append("REMARK   3")

        lines.append("REMARK   3  FIT TO DATA USED IN REFINEMENT.")
        # Where the free set came from, before the R-factors it conditions:
        # R-free values from different test sets are not comparable, and the
        # reader has no way to tell without this.
        if self.rfree_selection:
            _remark3(lines, "FREE R VALUE TEST SET SELECTION", self.rfree_selection)
        _remark3(lines, "R VALUE            (WORKING SET)", self.r_work, ".4f")
        _remark3(lines, "FREE R VALUE", self.r_free, ".4f")
        _remark3(lines, "FREE R VALUE TEST SET SIZE   (%)", self.percent_free, ".1f")
        _remark3(lines, "FREE R VALUE TEST SET COUNT", self.n_reflections_test, "d")
        lines.append("REMARK   3")

        lines.append("REMARK   3  B VALUES.")
        # Wilson-plot B is not computed; passing None makes _remark3 render the
        # literal "NULL" here intentionally (not a bug).
        _remark3(lines, "FROM WILSON PLOT           (A**2)", None, ".2f")
        _remark3(lines, "MEAN B VALUE      (OVERALL, A**2)", self.b_mean_overall, ".2f")
        _remark3(lines, "B MIN                      (A**2)", self.b_min, ".2f")
        _remark3(lines, "B MAX                      (A**2)", self.b_max, ".2f")
        lines.append("REMARK   3")

        lines.append("REMARK   3  RMS DEVIATIONS FROM IDEAL VALUES.")
        _remark3(lines, "BOND LENGTHS                 (A)", self.rmsd_bond_lengths, ".3f")
        _remark3(lines, "BOND ANGLES            (DEGREES)", self.rmsd_bond_angles, ".2f")
        lines.append("REMARK   3")

        lines.append("REMARK   3  NUMBER OF NON-HYDROGEN ATOMS USED IN REFINEMENT.")
        _remark3(lines, "PROTEIN ATOMS", self.n_atoms_protein, "d")
        _remark3(lines, "SOLVENT ATOMS", self.n_atoms_solvent, "d")
        _remark3(lines, "TOTAL", self.n_atoms_total, "d")
        lines.append("REMARK   3")

        if self.solvent_model_ksol is not None or self.solvent_model_bsol is not None:
            lines.append("REMARK   3  BULK SOLVENT MODELLING.")
            _remark3(lines, "K_SOL", self.solvent_model_ksol, ".4f")
            _remark3(lines, "B_SOL", self.solvent_model_bsol, ".2f")
            lines.append("REMARK   3")

        if self.starting_model:
            lines.append(f"REMARK   3  STARTING MODEL: {self.starting_model}")
            lines.append("REMARK   3")

        # The only free text in the block, and the caller wrote all of it.
        if self.output_remarks:
            lines.append("REMARK   3  OTHER REFINEMENT REMARKS:")
            for paragraph in self.output_remarks.splitlines():
                _wrap_remark3_text(lines, paragraph.strip())
            lines.append("REMARK   3")

        return lines

    # ------------------------------------------------------------------ #
    #  mmCIF rendering
    # ------------------------------------------------------------------ #

    def render_cif_categories(self) -> Dict[str, Dict[str, str]]:
        """Render metadata as mmCIF category dictionaries.

        Returns a dict of dicts keyed by mmCIF category, with item names
        as keys and string values. Uses official PDBx/mmCIF field names.

        Returns
        -------
        dict
            Nested dictionary ``{category: {field: value}}``.
        """
        cats: Dict[str, Dict[str, str]] = {}

        # _software: every program applied to this model, in order, ours last.
        # Always a loop, even with one entry -- that is what lets the next
        # refinement append a link rather than overwrite the chain, which is the
        # only record of prior work that mmCIF actually has room for.
        chain = list(self.software_chain)
        ordinals = []
        for entry in chain:
            try:
                ordinals.append(int(entry.get("pdbx_ordinal", 0)))
            except (TypeError, ValueError):
                pass
        description = self.refinement_method or ", ".join(
            part for part in (self.target_function, self.optimizer) if part
        )
        ours = {
            "name": self.program,
            "classification": "refinement",
            "pdbx_ordinal": str(max(ordinals, default=len(chain)) + 1),
        }
        if self.program_version:
            ours["version"] = self.program_version
        if description:
            ours["description"] = description
        chain.append(ours)
        columns = [
            key
            for key in ("pdbx_ordinal", "name", "version", "classification",
                        "description")
            if any(key in entry for entry in chain)
        ]
        cats["_software"] = {
            f"_software.{key}": [entry.get(key, "?") for entry in chain]
            for key in columns
        }

        # _struct
        if self.title:
            cats["_struct"] = {"_struct.title": self.title}

        # _audit_author (loop)
        if self.authors:
            cats["_audit_author"] = {
                "_audit_author.name": self.authors,
            }

        # _refine
        ref = {}
        if self.r_work is not None:
            ref["_refine.ls_R_factor_R_work"] = f"{self.r_work:.4f}"
        if self.r_free is not None:
            ref["_refine.ls_R_factor_R_free"] = f"{self.r_free:.4f}"
        if self.resolution_high is not None:
            ref["_refine.ls_d_res_high"] = f"{self.resolution_high:.2f}"
        if self.resolution_low is not None:
            ref["_refine.ls_d_res_low"] = f"{self.resolution_low:.2f}"
        if self.n_reflections_all is not None:
            ref["_refine.ls_number_reflns_all"] = str(self.n_reflections_all)
        if self.n_reflections_work is not None:
            ref["_refine.ls_number_reflns_R_work"] = str(self.n_reflections_work)
        if self.n_reflections_test is not None:
            ref["_refine.ls_number_reflns_R_free"] = str(self.n_reflections_test)
        if self.percent_free is not None:
            ref["_refine.ls_percent_reflns_R_free"] = f"{self.percent_free:.1f}"
        if self.b_mean_overall is not None:
            ref["_refine.B_iso_mean"] = f"{self.b_mean_overall:.2f}"
        if self.b_min is not None:
            ref["_refine.B_iso_min"] = f"{self.b_min:.2f}"
        if self.b_max is not None:
            ref["_refine.B_iso_max"] = f"{self.b_max:.2f}"
        if self.solvent_model_ksol is not None:
            ref["_refine.solvent_model_param_ksol"] = f"{self.solvent_model_ksol:.4f}"
        if self.solvent_model_bsol is not None:
            ref["_refine.solvent_model_param_bsol"] = f"{self.solvent_model_bsol:.2f}"
        # Free-set provenance sits beside the R-factors it conditions: the two
        # R-free values either side of a changed test set are not comparable.
        if self.rfree_selection:
            ref["_refine.pdbx_R_Free_selection_details"] = self.rfree_selection
        if self.starting_model:
            ref["_refine.pdbx_starting_model"] = self.starting_model
        if self.refinement_method:
            ref["_refine.pdbx_method_to_determine_struct"] = self.refinement_method
        if self.output_remarks:
            ref["_refine.details"] = self.output_remarks
        if ref:
            cats["_refine"] = ref

        # What this refinement started from. Standard category, and the only
        # structured place to say it.
        if self.starting_model:
            cats["_pdbx_initial_refinement_model"] = _initial_model_category(
                self.starting_model
            )

        # _refine_ls_restr (geometry deviations, as loop)
        if self.rmsd_bond_lengths is not None or self.rmsd_bond_angles is not None:
            restr_types = []
            restr_devs = []
            if self.rmsd_bond_lengths is not None:
                restr_types.append("f_bond_d")
                restr_devs.append(f"{self.rmsd_bond_lengths:.4f}")
            if self.rmsd_bond_angles is not None:
                restr_types.append("f_angle_d")
                restr_devs.append(f"{self.rmsd_bond_angles:.4f}")
            cats["_refine_ls_restr"] = {
                "_refine_ls_restr.type": restr_types,
                "_refine_ls_restr.dev_ideal": restr_devs,
            }

        # _refine_hist (atom counts)
        if self.n_atoms_total is not None:
            hist = {}
            hist["_refine_hist.number_atoms_total"] = str(self.n_atoms_total)
            if self.n_atoms_protein is not None:
                hist["_refine_hist.number_atoms_protein"] = str(self.n_atoms_protein)
            if self.n_atoms_solvent is not None:
                hist["_refine_hist.number_atoms_solvent"] = str(self.n_atoms_solvent)
            cats["_refine_hist"] = hist

        # Pass-through CIF categories
        for cat_name, items in self.passthrough_cif_categories.items():
            if cat_name not in cats:
                cats[cat_name] = items

        return cats


# ====================================================================== #
#  Private helpers
# ====================================================================== #


def _initial_model_category(starting_model: str) -> Dict[str, str]:
    """Describe the starting model the way deposited entries do.

    A four-character stem that looks like a PDB ID (digit then three
    alphanumerics, e.g. ``3GR5.pdb``) is reported as an accession code; anything
    else is named in ``details`` and left unaccessioned rather than guessed at.
    """
    import os
    import re

    basename = os.path.basename(starting_model)
    stem = os.path.splitext(basename)[0]
    cat = {
        "_pdbx_initial_refinement_model.id": "1",
        "_pdbx_initial_refinement_model.type": "experimental model",
        "_pdbx_initial_refinement_model.details": basename,
    }
    if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", stem):
        cat["_pdbx_initial_refinement_model.source_name"] = "PDB"
        cat["_pdbx_initial_refinement_model.accession_code"] = stem.upper()
    return cat


def _remark3(
    lines: List[str], label: str, value: Any, fmt: str = ""
) -> None:
    """Append a REMARK 3 ``label : value`` line.

    The line is always emitted; when ``value`` is None it is rendered as
    the literal ``NULL`` rather than being skipped.
    """
    if value is not None:
        formatted = f"{value:{fmt}}"
    else:
        formatted = "NULL"
    # Pad the label so the colons align down the block. Callers used to have to
    # pre-pad their own labels, and the ones that forgot rendered ragged.
    lines.append(f"REMARK   3   {label:<{_REMARK3_LABEL_WIDTH}} : {formatted}")


def _ident(lines: List[str], label: str, value: str) -> None:
    """Append a ``REMARK   3   LABEL      : value`` line, wrapped if long.

    Overflow continues on a further line whose label field is blank and whose
    colon stays in the same column, which is what REFMAC does with its own
    long values::

        REMARK   3   AUTHORS     : MURSHUDOV,SKUBAK,LEBEDEV,PANNU,STEINER,
        REMARK   3               : NICHOLLS,WINN,LONG,VAGIN

    Without this an optimizer description naming the cycles, mode, ADP model and
    scale target runs past column 80.
    """
    head = f"REMARK   3   {label:<12}: "
    cont = f"REMARK   3   {'':<12}: "
    width = 80 - len(head)
    prefix, current = head, ""
    for word in value.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(prefix + current)
            prefix, current = cont, word
        else:
            current = current + " " + word if current else word
    if current or prefix is head:
        lines.append(prefix + current)


def _wrap_remark3_text(lines: List[str], text: str) -> None:
    """Append free text as continuation-free ``REMARK   3`` lines.

    REMARK records have no continuation-number field -- unlike TITLE or AUTHOR,
    they simply repeat the same number -- so this wraps on width alone. An empty
    paragraph becomes a bare ``REMARK   3`` spacer.
    """
    prefix = "REMARK   3   "
    if not text:
        lines.append("REMARK   3")
        return
    width = 80 - len(prefix)
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(prefix + current)
            current = word
        else:
            current = current + " " + word if current else word
    if current:
        lines.append(prefix + current)


def _wrap_pdb_record(lines: List[str], record: str, text: str) -> None:
    """Wrap long text into multiple PDB records (80-char lines).

    Continuation lines use record + spaces + continuation number.
    """
    max_text = 80 - 10  # 10 chars for record + spaces
    words = text.split()
    current = ""
    continuation = 0

    for word in words:
        if current and len(current) + 1 + len(word) > max_text:
            if continuation == 0:
                lines.append(f"{record:<10}{current}")
            else:
                lines.append(f"{record:<8}{continuation + 1:>2}{current}")
            current = word
            continuation += 1
        else:
            current = current + " " + word if current else word

    if current:
        if continuation == 0:
            lines.append(f"{record:<10}{current}")
        else:
            lines.append(f"{record:<8}{continuation + 1:>2}{current}")
