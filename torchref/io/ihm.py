"""
IHM mmCIF reader and writer for kinetic ensemble data.

Provides ``IHMReader`` and ``IHMWriter`` for reading and writing IHM
(Integrative/Hybrid Methods) mmCIF files that describe multi-state
kinetic ensembles.

Requires the optional ``python-ihm`` dependency::

    pip install torchref[ihm]

Detection of IHM files (via ``is_ihm_file``) uses only gemmi and does
not require ``python-ihm``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import gemmi
import numpy as np
import pandas as pd

from torchref.io.ihm_mapping import IHMEnsembleMapping, IHMModelGroupInfo, IHMStateInfo

if TYPE_CHECKING:
    import torch

    from torchref.model.model_collection import ModelCollection
    from torchref.model.model_ft import ModelFT


def _check_ihm_available():
    """Raise ImportError with install instructions if python-ihm is missing."""
    try:
        import ihm  # noqa: F401
    except ImportError:
        raise ImportError(
            "python-ihm is required for IHM mmCIF support. "
            "Install with: pip install torchref[ihm]"
        )


class IHMReader:
    """
    Read IHM mmCIF files into torchref ModelCollection + IHMEnsembleMapping.

    Uses ``python-ihm`` to parse IHM-specific categories
    (``_ihm_multi_state_modeling``, ``_ihm_model_group``, etc.) and gemmi /
    ``ModelCIFReader`` to parse standard ``_atom_site`` data.

    Parameters
    ----------
    filepath : str
        Path to IHM mmCIF file.
    verbose : int
        Verbosity level (0=silent, 1=info, 2=debug).

    Examples
    --------
    ::

        reader = IHMReader("ensemble.cif")
        model_collection, mapping = reader(max_res=1.5)

        # Or step by step:
        mapping = reader.read_mapping()
        mapping.atom_data_per_state = reader.read_atom_data(mapping)
        model_collection = reader.build_model_collection(mapping)
    """

    def __init__(self, filepath: str, verbose: int = 0):
        _check_ihm_available()
        self.filepath = Path(filepath)
        self.verbose = verbose
        if not self.filepath.exists():
            raise FileNotFoundError(f"IHM file not found: {self.filepath}")

    # ------------------------------------------------------------------
    # Static detection (gemmi only, no python-ihm needed)
    # ------------------------------------------------------------------

    @staticmethod
    def is_ihm_file(filepath: str) -> bool:
        """
        Quick check whether a CIF file contains IHM categories.

        Uses gemmi to avoid requiring ``python-ihm`` for detection.
        Looks for ``_ihm_model_list`` or ``_ihm_multi_state_modeling`` loops.

        Parameters
        ----------
        filepath : str
            Path to a CIF/mmCIF file.

        Returns
        -------
        bool
        """
        try:
            doc = gemmi.cif.read(str(filepath))
        except Exception:
            return False

        for block in doc:
            if block.find(["_ihm_model_list.model_id"]):
                return True
            if block.find(["_ihm_multi_state_modeling.state_id"]):
                return True
        return False

    # ------------------------------------------------------------------
    # Read IHM metadata -> IHMEnsembleMapping
    # ------------------------------------------------------------------

    def read_mapping(self) -> IHMEnsembleMapping:
        """
        Parse IHM categories and build an ``IHMEnsembleMapping``.

        Reads the following IHM categories:
        - ``_ihm_multi_state_modeling`` -> states
        - ``_ihm_model_list`` -> model enumeration
        - ``_ihm_model_group`` + ``_ihm_model_group_link`` -> groups
        - ``_ihm_multi_state_model_group_link`` -> state-group fractions

        Also extracts cell and spacegroup from standard mmCIF categories.

        Returns
        -------
        IHMEnsembleMapping
        """
        import ihm.reader

        with open(self.filepath) as f:
            systems = ihm.reader.read(f)

        if not systems:
            raise ValueError(f"No IHM system found in {self.filepath}")

        system = systems[0]

        # --- Extract named states (skip unnamed container states) ---
        states = []
        state_groups = getattr(system, "state_groups", [])
        named_states = []  # (ihm.model.State, model_num)

        if state_groups:
            model_num = 1
            for sg in state_groups:
                for state in sg:
                    name = getattr(state, "name", None)
                    stype = getattr(state, "type", None)
                    # Skip unnamed states that are just model containers
                    if name is None and stype is None:
                        continue
                    details = getattr(state, "details", "") or ""
                    states.append(
                        IHMStateInfo(
                            state_id=model_num,
                            name=name or f"state_{model_num}",
                            details=details,
                            model_num=model_num,
                        )
                    )
                    named_states.append((state, model_num))
                    model_num += 1

        # Fallback: if no named states, infer from models
        if not states:
            states = self._infer_states_from_models(system)

        # --- Extract model groups from all state groups ---
        model_groups = []
        seen_ids = set()  # deduplicate by Python object identity
        group_id = 1

        if state_groups:
            for sg in state_groups:
                for state in sg:
                    for mg in state:
                        if id(mg) in seen_ids:
                            continue
                        seen_ids.add(id(mg))
                        mg_name = getattr(mg, "name", None)
                        # Skip unnamed placeholder groups with no models
                        if mg_name is None and len(mg) == 0:
                            continue
                        mg_name = mg_name or f"group_{group_id}"
                        model_groups.append(
                            IHMModelGroupInfo(
                                group_id=group_id,
                                name=mg_name,
                                state_fractions={},
                            )
                        )
                        group_id += 1

        # Fallback: build groups from model groups directly
        if not model_groups:
            model_groups = self._infer_groups_from_system(system, states)

        # --- Fill in fractions from _ihm_multi_state_model_group_link ---
        # Parse fractions directly from gemmi since python-ihm may not
        # fully expose them on the State objects
        self._fill_fractions_from_cif(states, model_groups)

        # --- Extract cell and spacegroup ---
        cell = None
        spacegroup = None
        doc = gemmi.cif.read(str(self.filepath))
        for block in doc:
            cell = self._extract_cell_from_block(block)
            spacegroup = self._extract_spacegroup_from_block(block)
            if cell is not None:
                break

        mapping = IHMEnsembleMapping(
            states=states,
            model_groups=model_groups,
            cell=cell,
            spacegroup=spacegroup,
        )

        if self.verbose > 0:
            print(f"IHM mapping: {len(states)} states, {len(model_groups)} groups")
            for s in states:
                print(f"  State {s.state_id}: {s.name} (model_num={s.model_num})")
            for g in model_groups:
                print(f"  Group {g.group_id}: {g.name} fractions={g.state_fractions}")

        return mapping

    def _infer_states_from_models(self, system) -> List[IHMStateInfo]:
        """Infer states from ihm.System model groups when state_groups is empty."""
        states = []
        model_nums = set()

        for mg in getattr(system, "model_groups", []):
            for model in mg:
                num = getattr(model, "_id", len(states) + 1)
                if num not in model_nums:
                    model_nums.add(num)
                    name = getattr(model, "name", None) or f"state_{num}"
                    states.append(
                        IHMStateInfo(
                            state_id=num, name=name, details="", model_num=num
                        )
                    )

        if not states:
            # Last resort: use atom_site model numbers
            states = self._infer_states_from_atom_site()

        return states

    def _infer_states_from_atom_site(self) -> List[IHMStateInfo]:
        """Infer states from pdbx_PDB_model_num in _atom_site."""
        from torchref.io.cif_readers import ModelCIFReader

        reader = ModelCIFReader(str(self.filepath), verbose=0)
        by_model = reader.get_atom_data_by_model()

        return [
            IHMStateInfo(
                state_id=num,
                name=f"state_{num}",
                details="",
                model_num=num,
            )
            for num in sorted(by_model.keys())
        ]

    def _infer_groups_from_system(
        self, system, states: List[IHMStateInfo]
    ) -> List[IHMModelGroupInfo]:
        """Build model groups with equal fractions when IHM categories are sparse."""
        n_states = len(states)
        if n_states == 0:
            return []

        equal_frac = 1.0 / n_states
        fracs = {s.state_id: equal_frac for s in states}

        return [
            IHMModelGroupInfo(
                group_id=1,
                name="ensemble",
                state_fractions=fracs,
            )
        ]

    def _fill_fractions_from_cif(
        self,
        states: List[IHMStateInfo],
        model_groups: List[IHMModelGroupInfo],
    ) -> None:
        """
        Parse ``_ihm_multi_state_model_group_link`` via gemmi to fill
        population fractions in model groups.
        """
        doc = gemmi.cif.read(str(self.filepath))
        for block in doc:
            table = block.find(
                [
                    "_ihm_multi_state_model_group_link.state_id",
                    "_ihm_multi_state_model_group_link.group_id",
                    "_ihm_multi_state_model_group_link.population_fraction",
                ]
            )
            if not table:
                continue

            # Build lookup: group_id -> IHMModelGroupInfo
            group_by_id = {g.group_id: g for g in model_groups}
            state_ids = [s.state_id for s in states]

            # Initialize all fractions to 0
            for g in model_groups:
                g.state_fractions = {sid: 0.0 for sid in state_ids}

            for row in table:
                sid = int(row[0])
                gid = int(row[1])
                frac_str = row[2]
                frac = float(frac_str) if frac_str not in (".", "?") else 0.0
                if gid in group_by_id and sid in state_ids:
                    group_by_id[gid].state_fractions[sid] = frac
            return

        # No link table found — use equal fractions as fallback
        n_states = len(states)
        if n_states > 0:
            equal = 1.0 / n_states
            for g in model_groups:
                g.state_fractions = {s.state_id: equal for s in states}

    def _extract_cell_from_block(self, block) -> Optional[List[float]]:
        """Extract unit cell from a gemmi CIF block."""
        try:
            a = block.find_value("_cell.length_a")
            b = block.find_value("_cell.length_b")
            c = block.find_value("_cell.length_c")
            alpha = block.find_value("_cell.angle_alpha")
            beta = block.find_value("_cell.angle_beta")
            gamma = block.find_value("_cell.angle_gamma")
            if a and b and c:
                return [
                    float(a), float(b), float(c),
                    float(alpha) if alpha else 90.0,
                    float(beta) if beta else 90.0,
                    float(gamma) if gamma else 90.0,
                ]
        except (ValueError, TypeError):
            pass
        return None

    def _extract_spacegroup_from_block(self, block) -> Optional[str]:
        """Extract space group from a gemmi CIF block."""
        for tag in [
            "_symmetry.space_group_name_H-M",
            "_space_group.name_H-M_alt",
        ]:
            val = block.find_value(tag)
            if val and val not in ("?", "."):
                return val.strip("'\"")
        return None

    # ------------------------------------------------------------------
    # Read atom data split by state
    # ------------------------------------------------------------------

    def read_atom_data(self, mapping: IHMEnsembleMapping) -> Dict[int, pd.DataFrame]:
        """
        Extract per-state atom DataFrames using ``pdbx_PDB_model_num``.

        Reuses ``ModelCIFReader`` for robust ``_atom_site`` parsing, then
        splits by model number and maps to state IDs.

        Parameters
        ----------
        mapping : IHMEnsembleMapping
            Mapping with state info (specifically ``model_num`` per state).

        Returns
        -------
        dict of int -> pandas.DataFrame
            Mapping of ``state_id`` -> atom DataFrame.
        """
        from torchref.io.cif_readers import ModelCIFReader

        reader = ModelCIFReader(str(self.filepath), verbose=0)
        by_model = reader.get_atom_data_by_model()

        result = {}
        for state in mapping.states:
            if state.model_num not in by_model:
                raise ValueError(
                    f"State '{state.name}' expects pdbx_PDB_model_num={state.model_num} "
                    f"but file only contains model numbers: {sorted(by_model.keys())}"
                )
            result[state.state_id] = by_model[state.model_num]

        # Validate atom consistency across states
        self._validate_atom_consistency(result, mapping)

        return result

    def _validate_atom_consistency(
        self, atom_data: Dict[int, pd.DataFrame], mapping: IHMEnsembleMapping
    ) -> None:
        """Check that all states have the same atoms in the same order."""
        state_ids = sorted(atom_data.keys())
        if len(state_ids) < 2:
            return

        ref_id = state_ids[0]
        ref_df = atom_data[ref_id]
        ref_count = len(ref_df)

        for sid in state_ids[1:]:
            df = atom_data[sid]
            if len(df) != ref_count:
                ref_name = mapping.get_state_by_id(ref_id).name
                this_name = mapping.get_state_by_id(sid).name
                raise ValueError(
                    f"Atom count mismatch: state '{ref_name}' has {ref_count} atoms "
                    f"but state '{this_name}' has {len(df)} atoms. "
                    f"All states must have identical atom lists for kinetic refinement."
                )

    # ------------------------------------------------------------------
    # Build ModelCollection from mapping + atom data
    # ------------------------------------------------------------------

    def build_model_collection(
        self,
        mapping: IHMEnsembleMapping,
        max_res: float = 1.5,
        radius_angstrom: float = 4.0,
        device: "Optional[torch.device]" = None,
    ) -> "ModelCollection":
        """
        Build a ``ModelCollection`` from parsed IHM data.

        For each state, creates a ``ModelFT`` and loads atom data.
        Then assembles a ``ModelCollection`` with one timepoint per model group.

        Parameters
        ----------
        mapping : IHMEnsembleMapping
            Must have ``atom_data_per_state`` populated.
        max_res : float
            Maximum resolution for FFT grid setup.
        radius_angstrom : float
            Radius for electron density calculation.
        device : torch.device, optional
            Device for model tensors.

        Returns
        -------
        ModelCollection
        """
        import torch

        from torchref.model.model_collection import ModelCollection
        from torchref.model.model_ft import ModelFT

        if mapping.atom_data_per_state is None:
            raise ValueError(
                "mapping.atom_data_per_state is None. "
                "Call read_atom_data() first."
            )

        if device is None:
            device = torch.device("cpu")

        # Build one ModelFT per state
        base_models = []
        for state in sorted(mapping.states, key=lambda s: s.state_id):
            df = mapping.atom_data_per_state[state.state_id]
            model = ModelFT(
                max_res=max_res,
                radius_angstrom=radius_angstrom,
                device=device,
            )
            # Build a lightweight reader-like callable for Model.load()
            cell = mapping.cell
            spacegroup = mapping.spacegroup or "P 1"
            sg = gemmi.SpaceGroup(spacegroup)
            reader = _DataFrameReader(df, cell, sg)
            model.load(reader)

            if self.verbose > 0:
                print(
                    f"  Loaded state '{state.name}': "
                    f"{len(df)} atoms"
                )
            base_models.append(model)

        # Create ModelCollection
        collection = ModelCollection(
            base_models=base_models,
            verbose=self.verbose,
        )

        # Add timepoints from model groups
        dark_name = mapping.identify_dark_group()
        for group in sorted(mapping.model_groups, key=lambda g: g.group_id):
            state_ids = mapping.get_state_ids()
            fractions = [
                group.state_fractions.get(sid, 0.0) for sid in state_ids
            ]

            # Normalize fractions
            total = sum(fractions)
            if total > 0:
                fractions = [f / total for f in fractions]
            else:
                fractions = [1.0 / len(fractions)] * len(fractions)

            is_dark = (group.name == dark_name)
            if is_dark:
                collection.add_dark(fractions=fractions)
            else:
                collection.add_timepoint(group.name, fractions=fractions)

        return collection

    # ------------------------------------------------------------------
    # Convenience: read everything in one call
    # ------------------------------------------------------------------

    def __call__(
        self,
        max_res: float = 1.5,
        radius_angstrom: float = 4.0,
        device: "Optional[torch.device]" = None,
    ) -> Tuple["ModelCollection", IHMEnsembleMapping]:
        """
        Read IHM file and return ``(ModelCollection, IHMEnsembleMapping)``.

        Parameters
        ----------
        max_res : float
            Maximum resolution for FFT grid.
        radius_angstrom : float
            Radius for electron density calculation.
        device : torch.device, optional
            Device for tensors.

        Returns
        -------
        tuple of (ModelCollection, IHMEnsembleMapping)
        """
        mapping = self.read_mapping()
        mapping.atom_data_per_state = self.read_atom_data(mapping)
        model_collection = self.build_model_collection(
            mapping,
            max_res=max_res,
            radius_angstrom=radius_angstrom,
            device=device,
        )
        return model_collection, mapping


class _DataFrameReader:
    """
    Minimal reader-like object for ``Model.load()``.

    Wraps a DataFrame + cell + spacegroup to match the protocol
    expected by ``Model.load(reader)`` where ``reader()`` returns
    ``(dataframe, cell, spacegroup)``.
    """

    def __init__(self, df: pd.DataFrame, cell: Optional[List[float]], spacegroup):
        self.df = df
        self.cell = cell
        self.spacegroup = spacegroup

    def __call__(self):
        return self.df, self.cell, self.spacegroup


class IHMWriter:
    """
    Write a torchref ``ModelCollection`` to IHM mmCIF format.

    Uses ``python-ihm`` to build a complete IHM System object and write
    it as a compliant mmCIF file.

    Parameters
    ----------
    model_collection : ModelCollection
        The collection to write.
    mapping : IHMEnsembleMapping, optional
        Original mapping for round-tripping metadata. If ``None``,
        creates a minimal mapping from the collection structure.
    verbose : int
        Verbosity level.

    Examples
    --------
    ::

        writer = IHMWriter(model_collection, mapping=mapping)
        writer.write("refined_ensemble.cif")

        # Or without a pre-existing mapping:
        writer = IHMWriter(model_collection)
        writer.write("refined_ensemble.cif")
    """

    def __init__(
        self,
        model_collection: "ModelCollection",
        mapping: Optional[IHMEnsembleMapping] = None,
        datasets: Optional[Dict[str, "ReflectionData"]] = None,
        verbose: int = 0,
    ):
        _check_ihm_available()
        self.model_collection = model_collection
        self.mapping = mapping or self._create_default_mapping()
        self.datasets = datasets
        self.verbose = verbose

    def _create_default_mapping(self) -> IHMEnsembleMapping:
        """
        Create a minimal ``IHMEnsembleMapping`` from ``ModelCollection`` structure.

        Each base model becomes a state; each timepoint becomes a model group
        with current fractions.
        """
        mc = self.model_collection

        # States from base models
        states = []
        for i in range(mc.n_base_models):
            states.append(
                IHMStateInfo(
                    state_id=i + 1,
                    name=f"state_{i + 1}",
                    details="",
                    model_num=i + 1,
                )
            )

        # Model groups from timepoints
        state_ids = [s.state_id for s in states]
        model_groups = []
        for group_id, (name, mixed) in enumerate(mc, start=1):
            fracs = mixed.fractions.detach().cpu().tolist()
            state_fractions = dict(zip(state_ids, fracs))
            model_groups.append(
                IHMModelGroupInfo(
                    group_id=group_id,
                    name=name,
                    state_fractions=state_fractions,
                )
            )

        # Cell and spacegroup from first base model
        cell = None
        spacegroup = None
        if mc.n_base_models > 0:
            model0 = mc.base_models[0]
            if hasattr(model0, "cell") and model0.cell is not None:
                cell_obj = model0.cell
                if hasattr(cell_obj, "parameters"):
                    cell = cell_obj.parameters.tolist()
                elif hasattr(cell_obj, "tolist"):
                    cell = cell_obj.tolist()
            if hasattr(model0, "spacegroup") and model0.spacegroup is not None:
                sg = model0.spacegroup
                if hasattr(sg, "hm"):
                    spacegroup = sg.hm
                elif hasattr(sg, "xhm"):
                    spacegroup = sg.xhm()
                else:
                    spacegroup = str(sg)

        return IHMEnsembleMapping(
            states=states,
            model_groups=model_groups,
            cell=cell,
            spacegroup=spacegroup,
        )

    def write(self, filepath: str) -> None:
        """
        Write IHM mmCIF file.

        Builds a ``python-ihm`` System object with:
        - Entity and assembly from base model atom data
        - Multi-state definitions from mapping states
        - Model groups from mapping groups with population fractions
        - Atom coordinates per state via ``pdbx_PDB_model_num``

        Parameters
        ----------
        filepath : str
            Output file path.
        """
        import ihm
        import ihm.dumper
        import ihm.model

        system = ihm.System(title="TorchRef kinetic ensemble")

        mc = self.model_collection
        mapping = self.mapping

        # --- Build entities and asym units from first base model ---
        import ihm.representation

        lpep = ihm.LPeptideAlphabet()
        asym_units = []

        if mc.n_base_models > 0:
            model0 = mc.base_models[0]
            for chain_id, seq_str in model0.chain_sequences:
                seq = []
                for char in seq_str:
                    if char == "?":
                        continue  # skip gaps
                    try:
                        seq.append(lpep[char])
                    except KeyError:
                        seq.append(lpep["UNK"])
                if seq:
                    entity = ihm.Entity(seq, description=f"Chain {chain_id}")
                    system.entities.append(entity)
                    asym = ihm.AsymUnit(entity, details=f"Chain {chain_id}")
                    system.asym_units.append(asym)
                    asym_units.append(asym)

        if not asym_units:
            entity = ihm.Entity(
                [lpep["UNK"]],
                description="Crystallographic model",
            )
            system.entities.append(entity)
            asym = ihm.AsymUnit(entity)
            system.asym_units.append(asym)
            asym_units.append(asym)

        # --- Assembly and representation ---
        assembly = ihm.Assembly(asym_units, name="Complete assembly")
        system.orphan_assemblies.append(assembly)

        rep = ihm.representation.Representation(
            [ihm.representation.AtomicSegment(a, rigid=False) for a in asym_units]
        )
        system.orphan_representations.append(rep)

        # --- Build states ---
        state_group = ihm.model.StateGroup()
        ihm_states = {}

        for state_info in sorted(mapping.states, key=lambda s: s.state_id):
            ihm_state = ihm.model.State(name=state_info.name)
            ihm_states[state_info.state_id] = ihm_state
            state_group.append(ihm_state)

        system.state_groups.append(state_group)

        # --- Build model groups and link to states ---
        for group_info in sorted(mapping.model_groups, key=lambda g: g.group_id):
            mg = ihm.model.ModelGroup(name=group_info.name)

            for state_id, fraction in group_info.state_fractions.items():
                if state_id in ihm_states:
                    ihm_state = ihm_states[state_id]
                    ihm_state.append(mg)
                    if fraction > 0:
                        ihm_state.population_fraction = fraction

        # --- Write atom coordinates as a separate _atom_site block ---
        # python-ihm writes IHM categories; we append atom_site via gemmi
        filepath = Path(filepath)

        # First write the IHM system
        with open(filepath, "w") as f:
            ihm.dumper.write(f, [system])

        # Then append multi-model atom_site data via gemmi
        self._append_atom_site(filepath, mapping)

        # Append per-timepoint reflection data blocks if provided
        if self.datasets:
            self._append_refln_blocks(filepath, mapping)

        if self.verbose > 0:
            print(f"Wrote IHM mmCIF: {filepath}")
            print(f"  {len(mapping.states)} states, {len(mapping.model_groups)} groups")
            if self.datasets:
                print(f"  {len(self.datasets)} reflection dataset(s)")

    @classmethod
    def _from_mixed_model(
        cls,
        mixed_model,
        mapping: IHMEnsembleMapping,
        verbose: int = 0,
    ) -> "IHMWriter":
        """
        Create an IHMWriter from a MixedModel (instead of ModelCollection).

        Wraps the MixedModel's constituent models in a lightweight adapter
        that provides the interface expected by the writer.

        Parameters
        ----------
        mixed_model : MixedModel
            The mixed model to write.
        mapping : IHMEnsembleMapping
            Mapping with state/group metadata.
        verbose : int
            Verbosity level.

        Returns
        -------
        IHMWriter
        """
        adapter = _MixedModelAdapter(mixed_model)
        writer = cls.__new__(cls)
        _check_ihm_available()
        writer.model_collection = adapter
        writer.mapping = mapping
        writer.verbose = verbose
        return writer

    def _append_atom_site(
        self, filepath: Path, mapping: IHMEnsembleMapping
    ) -> None:
        """
        Append ``_atom_site`` loop with ``pdbx_PDB_model_num`` to the CIF file.

        Iterates over base models, extracts current coordinates,
        and writes combined atom data with model numbers distinguishing states.
        """
        mc = self.model_collection

        # Read existing CIF
        doc = gemmi.cif.read(str(filepath))
        block = doc[0] if len(doc) > 0 else doc.add_new_block("torchref")

        # Add cell and spacegroup before atom_site so readers find them early
        if mapping.cell:
            a, b, c, alpha, beta, gamma = mapping.cell
            block.set_pair("_cell.length_a", f"{a:.4f}")
            block.set_pair("_cell.length_b", f"{b:.4f}")
            block.set_pair("_cell.length_c", f"{c:.4f}")
            block.set_pair("_cell.angle_alpha", f"{alpha:.4f}")
            block.set_pair("_cell.angle_beta", f"{beta:.4f}")
            block.set_pair("_cell.angle_gamma", f"{gamma:.4f}")

        if mapping.spacegroup:
            sg_val = mapping.spacegroup
            if " " in sg_val and not sg_val.startswith("'"):
                sg_val = f"'{sg_val}'"
            block.set_pair("_symmetry.space_group_name_H-M", sg_val)

        # Collect all atom data with model numbers
        all_rows = []
        states_sorted = sorted(mapping.states, key=lambda s: s.state_id)

        for i, state in enumerate(states_sorted):
            if i >= mc.n_base_models:
                break

            model = mc.base_models[i]
            # Update PDB DataFrame with current refined coordinates
            if hasattr(model, "update_pdb"):
                model.update_pdb()

            pdb_df = model.pdb

            for _, row in pdb_df.iterrows():
                atom_name = str(row.get("name", "CA"))
                altloc = str(row.get("altloc", ".")) or "."
                resname = str(row.get("resname", "UNK"))
                chainid = str(row.get("chainid", "A"))
                resseq = str(row.get("resseq", 1))
                icode = str(row.get("icode", ".")) or "."

                all_rows.append(
                    {
                        "group_PDB": row.get("ATOM", "ATOM"),
                        "id": str(row.get("serial", 0)),
                        "type_symbol": str(row.get("element", "C")),
                        "label_atom_id": atom_name,
                        "label_alt_id": altloc,
                        "label_comp_id": resname,
                        "label_asym_id": chainid,
                        "label_seq_id": resseq,
                        "pdbx_PDB_ins_code": icode,
                        "Cartn_x": f"{row.get('x', 0.0):.3f}",
                        "Cartn_y": f"{row.get('y', 0.0):.3f}",
                        "Cartn_z": f"{row.get('z', 0.0):.3f}",
                        "occupancy": f"{row.get('occupancy', 1.0):.2f}",
                        "B_iso_or_equiv": f"{row.get('tempfactor', 20.0):.2f}",
                        "auth_seq_id": resseq,
                        "auth_comp_id": resname,
                        "auth_asym_id": chainid,
                        "auth_atom_id": atom_name,
                        "pdbx_PDB_model_num": str(state.model_num),
                    }
                )

        if not all_rows:
            return

        # Build the loop — include auth_* columns required by Coot/viewers
        tags = [
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
        ]

        loop = block.init_loop("_atom_site.", [t.split(".")[-1] for t in tags])
        for row in all_rows:
            loop.add_row(
                [
                    row["group_PDB"],
                    row["id"],
                    row["type_symbol"],
                    row["label_atom_id"],
                    row["label_alt_id"],
                    row["label_comp_id"],
                    row["label_asym_id"],
                    row["label_seq_id"],
                    row["pdbx_PDB_ins_code"],
                    row["Cartn_x"],
                    row["Cartn_y"],
                    row["Cartn_z"],
                    row["occupancy"],
                    row["B_iso_or_equiv"],
                    row["auth_seq_id"],
                    row["auth_comp_id"],
                    row["auth_asym_id"],
                    row["auth_atom_id"],
                    row["pdbx_PDB_model_num"],
                ]
            )

        doc.write_file(str(filepath))

    # ------------------------------------------------------------------
    # Append per-timepoint reflection data
    # ------------------------------------------------------------------

    def _append_refln_blocks(
        self, filepath: Path, mapping: IHMEnsembleMapping
    ) -> None:
        """
        Append per-timepoint ``_refln`` data blocks to the CIF file.

        Each dataset gets its own CIF data block with cell, spacegroup,
        and a ``_refln`` loop containing HKL, F, σF, and R-free status.
        """
        doc = gemmi.cif.read(str(filepath))

        datasets = self.datasets
        # Support both dict and DatasetCollection (which is iterable)
        if hasattr(datasets, "items"):
            items = datasets.items()
        else:
            items = iter(datasets)

        for name, dataset in items:
            block = doc.add_new_block(name)

            # Cell and spacegroup per block
            if mapping.cell:
                a, b, c, alpha, beta, gamma = mapping.cell
                block.set_pair("_cell.length_a", f"{a:.4f}")
                block.set_pair("_cell.length_b", f"{b:.4f}")
                block.set_pair("_cell.length_c", f"{c:.4f}")
                block.set_pair("_cell.angle_alpha", f"{alpha:.4f}")
                block.set_pair("_cell.angle_beta", f"{beta:.4f}")
                block.set_pair("_cell.angle_gamma", f"{gamma:.4f}")

            if mapping.spacegroup:
                sg_val = mapping.spacegroup
                if " " in sg_val and not sg_val.startswith("'"):
                    sg_val = f"'{sg_val}'"
                block.set_pair("_symmetry.space_group_name_H-M", sg_val)

            # Build _refln loop
            if dataset.hkl is None:
                continue

            hkl = dataset.hkl.detach().cpu().numpy()
            n_refln = len(hkl)

            has_F = dataset.F is not None
            has_sigF = dataset.F_sigma is not None
            has_I = dataset.I is not None
            has_sigI = dataset.I_sigma is not None
            has_rfree = dataset.rfree_flags is not None

            # Prepare numpy arrays for efficient access
            F_np = dataset.F.detach().cpu().numpy() if has_F else None
            sigF_np = dataset.F_sigma.detach().cpu().numpy() if has_sigF else None
            I_np = dataset.I.detach().cpu().numpy() if has_I else None
            sigI_np = dataset.I_sigma.detach().cpu().numpy() if has_sigI else None
            rfree_np = dataset.rfree_flags.detach().cpu().numpy() if has_rfree else None

            # Build tag list
            tags = ["index_h", "index_k", "index_l"]
            if has_F:
                tags.append("F_meas_au")
            if has_sigF:
                tags.append("F_meas_sigma_au")
            if has_I:
                tags.append("intensity_meas")
            if has_sigI:
                tags.append("intensity_sigma")
            if has_rfree:
                tags.append("status")

            loop = block.init_loop("_refln.", tags)

            for i in range(n_refln):
                row = [str(int(hkl[i, 0])), str(int(hkl[i, 1])), str(int(hkl[i, 2]))]
                if has_F:
                    row.append(f"{F_np[i]:.4f}")
                if has_sigF:
                    row.append(f"{sigF_np[i]:.4f}")
                if has_I:
                    row.append(f"{I_np[i]:.4f}")
                if has_sigI:
                    row.append(f"{sigI_np[i]:.4f}")
                if has_rfree:
                    # CIF convention: 'f'=free, 'o'=working
                    row.append("f" if int(rfree_np[i]) == 0 else "o")
                loop.add_row(row)

            if self.verbose > 0:
                print(f"  Dataset '{name}': {n_refln} reflections")

        doc.write_file(str(filepath))

    # ------------------------------------------------------------------
    # Convenience for building mapping from ModelCollection + metadata
    # ------------------------------------------------------------------

    @staticmethod
    def mapping_from_kinetic_refinement(
        model_collection: "ModelCollection",
        state_names: Optional[List[str]] = None,
        time_delays: Optional[Dict[str, float]] = None,
    ) -> IHMEnsembleMapping:
        """
        Build an ``IHMEnsembleMapping`` from a ``ModelCollection`` with
        optional kinetic metadata.

        Parameters
        ----------
        model_collection : ModelCollection
            The refined model collection.
        state_names : list of str, optional
            Names for each base model / state. Default: state_1, state_2, ...
        time_delays : dict, optional
            Mapping of timepoint name -> time delay in seconds.

        Returns
        -------
        IHMEnsembleMapping
        """
        mc = model_collection

        # States
        states = []
        for i in range(mc.n_base_models):
            name = state_names[i] if state_names and i < len(state_names) else f"state_{i + 1}"
            states.append(
                IHMStateInfo(
                    state_id=i + 1,
                    name=name,
                    model_num=i + 1,
                )
            )

        # Model groups from timepoints
        state_ids = [s.state_id for s in states]
        model_groups = []
        for group_id, (name, mixed) in enumerate(mc, start=1):
            fracs = mixed.fractions.detach().cpu().tolist()
            state_fractions = dict(zip(state_ids, fracs))
            delay = time_delays.get(name) if time_delays else None
            model_groups.append(
                IHMModelGroupInfo(
                    group_id=group_id,
                    name=name,
                    state_fractions=state_fractions,
                    time_delay=delay,
                )
            )

        return IHMEnsembleMapping(
            states=states,
            model_groups=model_groups,
        )


class _MixedModelAdapter:
    """
    Adapter that makes a MixedModel look like a ModelCollection for IHMWriter.

    IHMWriter accesses ``model_collection.base_models`` and
    ``model_collection.n_base_models``. This adapter provides those
    properties from a MixedModel's ``models`` attribute.
    """

    def __init__(self, mixed_model):
        self._mixed_model = mixed_model

    @property
    def base_models(self):
        return self._mixed_model.models

    @property
    def n_base_models(self):
        return len(self._mixed_model.models)

    def __iter__(self):
        """Yield (name, mixed_model) — single group."""
        yield "ensemble", self._mixed_model

    def __len__(self):
        return 1
