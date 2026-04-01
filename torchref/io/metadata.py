"""
Unified metadata dictionary for PDB deposition headers and mmCIF categories.

This module provides a ``RefinementMetadata`` dataclass that serves as the
single source of truth for all deposition-related information. The same
metadata can be rendered as PDB REMARK 3 header records or as PDBx/mmCIF
``_refine`` category fields.

Examples
--------
::

    from torchref.io.metadata import RefinementMetadata

    # Build from a completed refinement
    meta = RefinementMetadata.from_refinement(refinement)

    # Render for output
    pdb_header = meta.render_pdb_header()
    cif_dict   = meta.render_cif_categories()

    # Merge pass-through headers from input file
    input_meta = RefinementMetadata.from_pdb_file("input.pdb")
    meta = input_meta.merge(meta)

    # Serialization
    d = meta.to_dict()
    meta2 = RefinementMetadata.from_dict(d)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class RefinementMetadata:
    """Unified metadata for PDB headers and mmCIF categories.

    Fields map to both PDB REMARK 3 lines and PDBx/mmCIF ``_refine``
    category items.  Only populated (non-None) fields are rendered.

    Parameters
    ----------
    program : str
        Refinement program name.
    program_version : str
        Program version string.
    """

    # Program identification
    program: str = "TORCHREF"
    program_version: str = ""
    refinement_method: str = ""  # e.g. "difference-refine", "LBFGS"

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

    # Pass-through: raw header lines from input file
    passthrough_pdb_remarks: List[str] = field(default_factory=list)
    passthrough_cif_categories: Dict[str, Any] = field(default_factory=dict)

    # Custom remarks
    custom_remarks: List[str] = field(default_factory=list)

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
        """Extract metadata from a completed Refinement object.

        Reuses existing statistics from ``collect_metrics()``,
        ``get_rfactor()``, and reflection data attributes.
        Silently skips any unavailable statistics.

        Parameters
        ----------
        refinement : torchref.refinement.Refinement
            A refinement object (after refinement is complete).
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
            # Let's use the same convention as populate_state_meta
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
            if model.initialized and model._restraints is not None:
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
        try:
            scaler = refinement.scaler
            if hasattr(scaler, "solvent_model") and scaler.solvent_model is not None:
                sm = scaler.solvent_model
                meta.solvent_model_ksol = float(
                    torch.exp(sm.log_k_solvent).item()
                )
                meta.solvent_model_bsol = float(sm.b_solvent.item())
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
    def from_pdb_file(cls, filepath: str) -> RefinementMetadata:
        """Extract header metadata from an existing PDB file.

        Captures TITLE, AUTHOR, and REMARK records for pass-through.
        """
        meta = cls()
        remarks = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    record = line[:6].strip()
                    if record in ("ATOM", "HETATM"):
                        break
                    if record == "TITLE":
                        title_text = line[10:].strip()
                        if meta.title:
                            meta.title += " " + title_text
                        else:
                            meta.title = title_text
                    elif record == "AUTHOR":
                        author_text = line[10:].strip()
                        # Authors are comma-separated in PDB
                        for author in author_text.split(","):
                            author = author.strip()
                            if author:
                                meta.authors.append(author)
                    elif record.startswith("REMARK"):
                        remarks.append(line.rstrip("\n"))
            meta.passthrough_pdb_remarks = remarks
        except Exception:
            pass
        return meta

    @classmethod
    def from_cif_file(cls, filepath: str) -> RefinementMetadata:
        """Extract refinement metadata from an existing mmCIF file.

        Captures ``_struct.title``, ``_audit_author.name``, and
        ``_refine`` category items for pass-through.
        """
        meta = cls()
        try:
            import gemmi

            doc = gemmi.cif.read(filepath)
            block = doc[0]

            # Title
            title = block.find_value("_struct.title")
            if title and title != "?":
                meta.title = gemmi.cif.as_string(title)

            # Authors
            author_loop = block.find(["_audit_author.name"])
            if author_loop:
                for row in author_loop:
                    name = gemmi.cif.as_string(row[0])
                    if name and name != "?":
                        meta.authors.append(name)

            # _refine category pass-through
            refine_cats = {}
            for tag in block.find(["_refine."]):
                # Collect all _refine.* pairs
                pass
            # Use find_values for individual items
            for item_name in [
                "_refine.ls_R_factor_R_work",
                "_refine.ls_R_factor_R_free",
                "_refine.ls_d_res_high",
                "_refine.ls_d_res_low",
                "_refine.ls_number_reflns_R_work",
                "_refine.ls_number_reflns_R_free",
                "_refine.B_iso_mean",
            ]:
                val = block.find_value(item_name)
                if val and val not in ("?", "."):
                    refine_cats[item_name] = gemmi.cif.as_string(val)

            if refine_cats:
                meta.passthrough_cif_categories["_refine"] = refine_cats

        except Exception:
            pass
        return meta

    # ------------------------------------------------------------------ #
    #  Merge
    # ------------------------------------------------------------------ #

    def merge(self, other: RefinementMetadata) -> RefinementMetadata:
        """Merge *other* into self. Non-None values in *other* take precedence.

        Pass-through containers are combined (not replaced).

        Parameters
        ----------
        other : RefinementMetadata
            Metadata to merge in (takes precedence for non-None fields).

        Returns
        -------
        RefinementMetadata
            A new merged instance.
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
            elif f.name == "custom_remarks":
                merged_remarks = list(self_val) + list(other_val)
                setattr(merged, f.name, merged_remarks)
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
        """Render metadata as PDB header records (REMARK 3, TITLE, AUTHOR).

        Returns
        -------
        str
            Multi-line string ready to insert into a PDB file.
        """
        lines: List[str] = []

        # Pass-through remarks first (from input file)
        for remark in self.passthrough_pdb_remarks:
            lines.append(remark)

        # TITLE
        if self.title:
            _wrap_pdb_record(lines, "TITLE", self.title)

        # AUTHOR
        if self.authors:
            author_str = ", ".join(self.authors)
            _wrap_pdb_record(lines, "AUTHOR", author_str)

        # REMARK 3 - Refinement statistics
        lines.append("REMARK   3")
        lines.append("REMARK   3 REFINEMENT.")
        lines.append(
            f"REMARK   3   PROGRAM     : {self.program} {self.program_version}".rstrip()
        )
        if self.refinement_method:
            lines.append(
                f"REMARK   3   METHOD      : {self.refinement_method}"
            )
        lines.append("REMARK   3")

        # Data used in refinement
        lines.append("REMARK   3  DATA USED IN REFINEMENT.")
        _remark3(lines, "RESOLUTION RANGE HIGH (ANGSTROMS)", self.resolution_high, ".2f")
        _remark3(lines, "RESOLUTION RANGE LOW  (ANGSTROMS)", self.resolution_low, ".2f")
        _remark3(lines, "NUMBER OF REFLECTIONS", self.n_reflections_all, "d")
        lines.append("REMARK   3")

        # Fit to data
        lines.append("REMARK   3  FIT TO DATA USED IN REFINEMENT.")
        _remark3(lines, "R VALUE            (WORKING SET)", self.r_work, ".4f")
        _remark3(lines, "FREE R VALUE", self.r_free, ".4f")
        _remark3(lines, "FREE R VALUE TEST SET SIZE   (%)", self.percent_free, ".1f")
        _remark3(lines, "FREE R VALUE TEST SET COUNT", self.n_reflections_test, "d")
        lines.append("REMARK   3")

        # B-values
        lines.append("REMARK   3  B VALUES.")
        _remark3(lines, "FROM WILSON PLOT           (A**2)", None, ".2f")
        _remark3(lines, "MEAN B VALUE      (OVERALL, A**2)", self.b_mean_overall, ".2f")
        _remark3(lines, "B MIN                      (A**2)", self.b_min, ".2f")
        _remark3(lines, "B MAX                      (A**2)", self.b_max, ".2f")
        lines.append("REMARK   3")

        # RMS deviations
        lines.append("REMARK   3  RMS DEVIATIONS FROM IDEAL VALUES.")
        _remark3(lines, "BOND LENGTHS                 (A)", self.rmsd_bond_lengths, ".3f")
        _remark3(lines, "BOND ANGLES            (DEGREES)", self.rmsd_bond_angles, ".2f")
        lines.append("REMARK   3")

        # Model contents
        lines.append("REMARK   3  NUMBER OF NON-HYDROGEN ATOMS USED IN REFINEMENT.")
        _remark3(lines, "PROTEIN ATOMS", self.n_atoms_protein, "d")
        _remark3(lines, "SOLVENT ATOMS", self.n_atoms_solvent, "d")
        _remark3(lines, "TOTAL", self.n_atoms_total, "d")
        lines.append("REMARK   3")

        # Solvent model
        if self.solvent_model_ksol is not None or self.solvent_model_bsol is not None:
            lines.append("REMARK   3  BULK SOLVENT MODELLING.")
            _remark3(lines, "K_SOL", self.solvent_model_ksol, ".4f")
            _remark3(lines, "B_SOL", self.solvent_model_bsol, ".2f")
            lines.append("REMARK   3")

        # Custom remarks
        for remark in self.custom_remarks:
            lines.append(f"REMARK   3 {remark}")

        lines.append("REMARK   3")

        return "\n".join(lines) + "\n"

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

        # _software
        sw = {}
        sw["_software.name"] = self.program
        if self.program_version:
            sw["_software.version"] = self.program_version
        sw["_software.classification"] = "refinement"
        if self.refinement_method:
            sw["_software.description"] = self.refinement_method
        sw["_software.pdbx_ordinal"] = "1"
        cats["_software"] = sw

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
        if ref:
            cats["_refine"] = ref

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


def _remark3(
    lines: List[str], label: str, value: Any, fmt: str = ""
) -> None:
    """Append a REMARK 3 line with label : value formatting."""
    if value is not None:
        formatted = f"{value:{fmt}}"
    else:
        formatted = "NULL"
    line = f"REMARK   3   {label} : {formatted}"
    lines.append(line)


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
