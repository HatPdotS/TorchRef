"""
Intermediate representation for IHM mmCIF ensemble data.

Provides dataclasses that map between IHM categories and torchref's
ModelCollection / DatasetCollection structures. This mapping can be:

1. Populated by IHMReader from an IHM mmCIF file
2. Constructed programmatically (e.g., from a kinetic model)
3. Passed to IHMWriter for round-trip output

Concept Mapping
---------------
IHM state  ->  base model (ModelFT) in ModelCollection
IHM model group  ->  timepoint entry (_SharedMixedModel) in ModelCollection
IHM population fraction  ->  fraction_params in _SharedMixedModel
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class IHMStateInfo:
    """
    Metadata for a single structural state (e.g., ground state, intermediate).

    Parameters
    ----------
    state_id : int
        Unique identifier matching ``_ihm_multi_state_modeling.state_id``.
    name : str
        Human-readable name (e.g., ``"ground_state"``, ``"intermediate_1"``).
    details : str
        Free-text description of this state.
    model_num : int
        ``pdbx_PDB_model_num`` in the ``_atom_site`` loop that corresponds
        to this state's coordinates.
    """

    state_id: int
    name: str
    details: str = ""
    model_num: int = 1


@dataclass
class IHMModelGroupInfo:
    """
    Metadata for a model group (experimental condition / timepoint).

    Parameters
    ----------
    group_id : int
        Unique identifier matching ``_ihm_model_group.id``.
    name : str
        Human-readable name (e.g., ``"dark"``, ``"1ps"``, ``"5ps"``).
    state_fractions : Dict[int, float]
        Mapping of ``state_id`` -> population fraction for this group.
        Fractions should sum to 1.0.
    time_delay : float, optional
        Time delay in ``time_delay_units`` (for time-resolved experiments).
    time_delay_units : str, optional
        Units for ``time_delay``. Default ``"s"`` (seconds).
    """

    group_id: int
    name: str
    state_fractions: Dict[int, float] = field(default_factory=dict)
    time_delay: Optional[float] = None
    time_delay_units: str = "s"


@dataclass
class IHMEnsembleMapping:
    """
    Complete mapping between IHM mmCIF categories and torchref structures.

    This is the central interchange object: both ``IHMReader`` and
    ``IHMWriter`` operate through it.  It can also be constructed manually
    for programmatic workflows (e.g., building an IHM file from a
    ``KineticRefinement`` result without reading one first).

    Parameters
    ----------
    states : List[IHMStateInfo]
        Structural states (one per base model in ModelCollection).
    model_groups : List[IHMModelGroupInfo]
        Model groups / timepoints (one per timepoint in ModelCollection).
    cell : list of float, optional
        Unit cell parameters ``[a, b, c, alpha, beta, gamma]``.
    spacegroup : str, optional
        Space group name (Hermann-Mauguin notation).
    atom_data_per_state : dict, optional
        Mapping of ``state_id`` -> pandas DataFrame with atom data.
        Populated by ``IHMReader.read_atom_data()``.
    """

    states: List[IHMStateInfo] = field(default_factory=list)
    model_groups: List[IHMModelGroupInfo] = field(default_factory=list)
    cell: Optional[List[float]] = None
    spacegroup: Optional[str] = None
    atom_data_per_state: Optional[Dict[int, pd.DataFrame]] = None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_state_ids(self) -> List[int]:
        """Return state IDs ordered by ``state_id``."""
        return sorted(s.state_id for s in self.states)

    def get_timepoint_names(self) -> List[str]:
        """Return model group names ordered by ``group_id``."""
        return [g.name for g in sorted(self.model_groups, key=lambda g: g.group_id)]

    def get_fractions_for_group(self, group_name: str) -> List[float]:
        """
        Return population fractions for a model group, ordered by state_id.

        Parameters
        ----------
        group_name : str
            Name of the model group.

        Returns
        -------
        list of float
            Fractions ordered by ascending ``state_id``.

        Raises
        ------
        KeyError
            If no group with the given name exists.
        """
        for group in self.model_groups:
            if group.name == group_name:
                state_ids = self.get_state_ids()
                return [group.state_fractions.get(sid, 0.0) for sid in state_ids]
        raise KeyError(f"No model group named '{group_name}'")

    def identify_dark_group(self) -> Optional[str]:
        """
        Heuristic: identify the reference / dark group.

        Returns the name of the first model group where a single state
        has population fraction >= 0.95, or ``None`` if no such group exists.
        """
        for group in sorted(self.model_groups, key=lambda g: g.group_id):
            fracs = list(group.state_fractions.values())
            if fracs and max(fracs) >= 0.95:
                return group.name
        return None

    def get_state_by_id(self, state_id: int) -> IHMStateInfo:
        """
        Look up a state by its ID.

        Raises
        ------
        KeyError
            If no state with the given ID exists.
        """
        for state in self.states:
            if state.state_id == state_id:
                return state
        raise KeyError(f"No state with id {state_id}")

    def get_group_by_name(self, name: str) -> IHMModelGroupInfo:
        """
        Look up a model group by name.

        Raises
        ------
        KeyError
            If no group with the given name exists.
        """
        for group in self.model_groups:
            if group.name == name:
                return group
        raise KeyError(f"No model group named '{name}'")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Check internal consistency.

        Raises
        ------
        ValueError
            If states are empty, fractions don't reference valid states,
            or fractions don't sum to ~1.0 for any group.
        """
        if not self.states:
            raise ValueError("IHMEnsembleMapping has no states defined.")
        if not self.model_groups:
            raise ValueError("IHMEnsembleMapping has no model groups defined.")

        state_ids = set(s.state_id for s in self.states)

        for group in self.model_groups:
            # Check that all referenced state IDs exist
            for sid in group.state_fractions:
                if sid not in state_ids:
                    raise ValueError(
                        f"Model group '{group.name}' references state_id={sid} "
                        f"which is not in the states list."
                    )
            # Check fractions sum to ~1.0
            total = sum(group.state_fractions.values())
            if abs(total - 1.0) > 0.05:
                raise ValueError(
                    f"Model group '{group.name}' fractions sum to {total:.4f}, "
                    f"expected ~1.0."
                )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state_names = [s.name for s in self.states]
        group_names = [g.name for g in self.model_groups]
        return (
            f"IHMEnsembleMapping("
            f"states={state_names}, "
            f"groups={group_names}, "
            f"spacegroup='{self.spacegroup}')"
        )
