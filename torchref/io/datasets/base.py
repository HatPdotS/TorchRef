"""
Base dataclass for crystallographic datasets.

This module defines the CrystalDataset dataclass that provides:
- All possible tensor fields for crystallographic data (optional)
- Device management (to, cuda, cpu)
- Serialization (save, load)

On the base class the ``spacegroup`` field is an ``Optional[str]`` name;
concrete subclasses (e.g. ReflectionData, FcalcDataset) override it to hold
``torchref.symmetry.SpaceGroup`` objects.
"""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import gemmi
import torch

from torchref.config import get_default_device
from torchref.symmetry import Cell
from torchref.utils.device_mixin import DeviceMovementMixin

if TYPE_CHECKING:
    pass


@dataclass
class CrystalDataset(DeviceMovementMixin):
    """
    Base dataclass for crystallographic datasets.

    Defines all possible tensor fields (optional) and handles device management
    and serialization. Subclasses add domain-specific methods.

    This lightweight design enables scaling to 1000s of datasets without
    the overhead of torch.nn.Module.

    Parameters
    ----------
    device : torch.device
        Device for tensors ('cpu', 'cuda', etc.). Defaults to the configured device.current.
    verbose : int
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.

    Examples
    --------
    Basic usage::

        data = CrystalDataset(device='cuda')
        data.hkl = torch.tensor([[1, 0, 0], [0, 1, 0]])
        data.cpu()  # Move all tensors to CPU
    """

    # === Core reflection tensors ===
    hkl: Optional[torch.Tensor] = None  # Miller indices (N, 3), int32
    F: Optional[torch.Tensor] = None  # Structure factor amplitudes (N,)
    F_sigma: Optional[torch.Tensor] = None  # Amplitude uncertainties (N,)
    I: Optional[torch.Tensor] = None  # Intensities (N,)
    I_sigma: Optional[torch.Tensor] = None  # Intensity uncertainties (N,)
    rfree_flags: Optional[torch.Tensor] = None  # R-free test set flags (N,), int32
    # Reserved validation set: per-reflection bool. None => no validation
    # reflections (the validation subset is empty). When populated, these
    # reflections are carved out of BOTH the work and free sets (disjoint).
    validation_flags: Optional[torch.Tensor] = None  # (N,), bool
    resolution: Optional[torch.Tensor] = None  # Resolution per reflection (N,)
    bin_indices: Optional[torch.Tensor] = None  # Resolution bin assignments (N,), int32
    outlier_flags: Optional[torch.Tensor] = None  # Outlier flags (N,), bool
    phase: Optional[torch.Tensor] = None  # Phases in radians (N,)
    fom: Optional[torch.Tensor] = None  # Figure of merit (N,)
    _centric_flags: Optional[torch.Tensor] = None  # Centric flags (N,), bool
    # Anomalous (Bijvoet) bookkeeping, populated during canonicalization:
    #   friedel_flags: True where the canonical mapping conjugated a Friedel mate
    #   hkl_anomalous: signed Miller indices (canonical for +, negated for flagged
    #     mates) used for structure-factor evaluation so the two members of a
    #     Bijvoet pair get distinct |F_calc|. self.hkl stays the canonical ASU index.
    friedel_flags: Optional[torch.Tensor] = None  # (N,), bool
    hkl_anomalous: Optional[torch.Tensor] = None  # (N, 3), int32
    # Scalar merge state (NOT per-reflection): True when the data are Friedel-merged
    # (one row per ASU reflection), False when anomalous F(+)/F(-) have been loaded as
    # explicit Bijvoet pairs (separate signed-HKL rows). Gates the model's f'' term.
    friedel_merged: bool = True

    # === E-value and anisotropy correction fields ===
    E: Optional[torch.Tensor] = None  # E-values (N,)
    E_squared: Optional[torch.Tensor] = None  # E² values (N,)
    F_squared_corrected: Optional[torch.Tensor] = None  # Anisotropy-corrected F² (N,)
    U_aniso: Optional[torch.Tensor] = None  # Fitted anisotropy parameters (6,)
    radial_shell_indices: Optional[torch.Tensor] = None  # Shell assignments (N,)

    # === Unit cell and symmetry ===
    cell: Optional[Cell] = None  # Cell object with [a, b, c, alpha, beta, gamma]
    spacegroup: Optional[str] = None  # Space group name string

    # === Metadata ===
    device: torch.device = field(default_factory=get_default_device)
    verbose: int = 1

    # === Source tracking ===
    rfree_source: Optional[str] = None
    amplitude_source: Optional[str] = None
    intensity_source: Optional[str] = None
    phase_source: Optional[str] = None

    # === Wilson B-factors ===
    wilson_b: Optional[float] = None
    wilson_b_structure: Optional[float] = None
    wilson_b_solvent: Optional[float] = None
    wilson_k_sol: Optional[float] = None

    # === Outlier detection parameters ===
    outlier_detection_params: Optional[Dict[str, Any]] = None

    # === Masks (initialized in __post_init__) ===
    # Note: masks is not a dataclass field to avoid serialization issues
    # It's initialized in __post_init__ and handled specially

    def __post_init__(self):
        """Initialize non-field attributes after dataclass init."""
        # Ensure device is a torch.device object
        if isinstance(self.device, str):
            object.__setattr__(self, "device", torch.device(self.device))
        # Import here to avoid circular imports
        from torchref.utils.utils import TensorMasks

        # Initialize masks as TensorMasks (dict subclass)
        if not hasattr(self, "masks") or self.masks is None:
            self.masks = TensorMasks(device=self.device)

    # ========== DEVICE MANAGEMENT ==========

    def _tensor_fields(self):
        """
        Yield (name, tensor) for all tensor attributes.

        Yields
        ------
        Tuple[str, torch.Tensor]
            Field name and tensor value for each tensor field.

        Note
        ----
        Cell objects are excluded; they are handled separately in to().
        """
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                yield f.name, val

    # ========== SERIALIZATION ==========

    def _get_state(self) -> Dict[str, Any]:
        """
        Get serializable state dictionary.

        Returns
        -------
        Dict[str, Any]
            State dictionary with all tensor and metadata fields.
        """

        state = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                state[f.name] = val.cpu()
            elif f.name == "cell" and val is not None:
                # Store Cell tensor data for serialization
                state[f.name] = val.data.cpu()
            elif f.name == "device":
                # Store device as string
                state[f.name] = str(val)
            elif f.name == "spacegroup" and val is not None:
                # Store spacegroup as string for serialization
                state[f.name] = val.xhm()  # Extended Hermann-Mauguin
            else:
                state[f.name] = val
        # Handle masks specially
        if hasattr(self, "masks") and self.masks is not None:
            state["masks"] = {k: v.cpu() for k, v in self.masks.items()}
        return state

    @classmethod
    def _from_state(cls, state: Dict[str, Any], device=None) -> "CrystalDataset":
        """
        Reconstruct from state dictionary.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary from _get_state().
        device : torch.device, optional
            Device to load tensors onto. If None, defaults to the configured
            device via get_default_device().

        Returns
        -------
        CrystalDataset
            Reconstructed dataset.
        """
        from torchref.utils.utils import TensorMasks

        if device is None:
            device = get_default_device()

        # Extract masks before creating object
        masks_state = state.pop("masks", {})

        # Convert device string back to torch.device
        if "device" in state:
            state["device"] = torch.device(state["device"])

        # Spacegroup is stored as string — keep as-is
        # (no conversion needed since CrystalDataset.spacegroup is now str)

        # Convert cell tensor back to Cell object
        if "cell" in state and state["cell"] is not None:
            if isinstance(state["cell"], torch.Tensor):
                state["cell"] = Cell(state["cell"], dtype=torch.float32, device=device)

        # Create object with remaining state
        obj = cls(**state)

        # Restore masks
        if masks_state:
            obj.masks = TensorMasks(data=masks_state, device=device)

        return obj.to(device)

    def save_state(self, path: str) -> None:
        """
        Save dataset state to file.

        Parameters
        ----------
        path : str
            Output file path.

        Examples
        --------
        Save to file::

            data.save_state('reflection_data.pt')
        """
        state = self._get_state()
        state["__class__"] = self.__class__.__name__
        torch.save(state, path)
        if self.verbose > 0:
            print(f"Saved {self.__class__.__name__} to {path}")

    @classmethod
    def load_state(cls, path: str, device=None) -> "CrystalDataset":
        """
        Load dataset state from file.

        Parameters
        ----------
        path : str
            Input file path.
        device : torch.device, optional
            Device to load tensors onto. If None, defaults to the configured
            device via get_default_device().

        Returns
        -------
        CrystalDataset
            Loaded dataset.

        Examples
        --------
        Load from file::

            data = ReflectionData.load_state('reflection_data.pt', device='cuda')
        """
        state = torch.load(path, map_location="cpu")
        # Remove class marker if present
        state.pop("__class__", None)
        obj = cls._from_state(state, device)
        if obj.verbose > 0:
            print(f"Loaded {cls.__name__} from {path}")
        return obj

    # ========== UTILITY METHODS ==========

    def __len__(self) -> int:
        """Return number of reflections in dataset."""
        if self.hkl is not None:
            return len(self.hkl)
        return 0

    def __repr__(self) -> str:
        """String representation of dataset."""
        n_refl = len(self)
        sg = self.spacegroup if self.spacegroup else "unknown"
        return f"{self.__class__.__name__}(n_reflections={n_refl}, spacegroup='{sg}', device={self.device})"

    @property
    def spacegroup_name(self) -> Optional[str]:
        """Get space group name as string (short form, e.g., 'P212121')."""
        if self.spacegroup is None:
            return None
        return gemmi.SpaceGroup(self.spacegroup).short_name()

    @property
    def spacegroup_hm(self) -> Optional[str]:
        """Get space group Hermann-Mauguin name with spaces (e.g., 'P 21 21 21')."""
        if self.spacegroup is None:
            return None
        return gemmi.SpaceGroup(self.spacegroup).hm

    @property
    def spacegroup_number(self) -> Optional[int]:
        """Get space group number (1-230)."""
        if self.spacegroup is None:
            return None
        return gemmi.SpaceGroup(self.spacegroup).number
