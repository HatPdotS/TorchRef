"""
Base dataclass for crystallographic datasets: every optional tensor field,
device management and save/load.

Beware the ``spacegroup`` field: annotated ``Optional[str]`` here, but at
runtime ``FcalcDataset`` *and* ``ReflectionData`` (which does not override the
annotation) both store a ``torchref.symmetry.SpaceGroup`` object in it.
"""

import warnings
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import gemmi
import torch

from torchref.config import get_default_device, get_float_dtype, normalize_device
from torchref.symmetry import Cell
from torchref.utils.device_mixin import DeviceMovementMixin

if TYPE_CHECKING:
    pass


@dataclass
class CrystalDataset(DeviceMovementMixin):
    """
    Base dataclass for crystallographic datasets.

    Declares every optional tensor field and handles device movement and
    serialization; subclasses add the domain methods. Deliberately not an
    ``nn.Module``, so thousands of datasets stay affordable.

    Parameters
    ----------
    device : torch.device
        Device for tensors. Defaults to ``get_default_device()``.
    verbose : int
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.
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

    # === Masks (initialized in __post_init__) ===
    # Note: masks is not a dataclass field to avoid serialization issues
    # It's initialized in __post_init__ and handled specially

    def __post_init__(self):
        """Initialize non-field attributes after dataclass init."""
        # Canonicalise, don't merely coerce: ``torch.device("mps")`` has no
        # index and so compares unequal to the ``mps:0`` every tensor reports,
        # which makes ``resolve_device``'s equality checks misfire.
        object.__setattr__(self, "device", normalize_device(self.device))
        # Imported here to avoid a circular import.
        from torchref.utils.utils import TensorMasks

        if not hasattr(self, "masks") or self.masks is None:
            self.masks = TensorMasks(device=self.device)

    # ========== DEVICE MANAGEMENT ==========

    def _tensor_fields(self):
        """Yield ``(name, tensor)`` for every tensor field.

        ``Cell`` objects are NOT included -- ``to()`` moves those separately.
        """
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                yield f.name, val

    # ========== SERIALIZATION ==========

    def _get_state(self) -> Dict[str, Any]:
        """State dict of all fields, tensors on CPU, cell/device/spacegroup
        flattened to tensor/str, plus a ``"masks"`` entry.
        """

        state = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                state[f.name] = val.cpu()
            elif f.name == "cell" and val is not None:
                state[f.name] = val.data.cpu()
            elif f.name == "device":
                state[f.name] = str(val)
            elif f.name == "spacegroup" and val is not None:
                state[f.name] = val.xhm()  # Extended Hermann-Mauguin
            else:
                state[f.name] = val
        # Masks are not a dataclass field, so handle them separately.
        if hasattr(self, "masks") and self.masks is not None:
            state["masks"] = {k: v.cpu() for k, v in self.masks.items()}
        return state

    @classmethod
    def _drop_stale_state_keys(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        """Drop state keys that are no longer dataclass fields, warning about each.

        :meth:`_from_state` splats the state straight into the constructor, so a field
        removed after a checkpoint was written would otherwise make that checkpoint
        permanently unloadable with a bare ``TypeError``. Returns a filtered copy; the
        caller's dict is left alone.
        """
        known = {f.name for f in fields(cls)}
        stale = sorted(set(state) - known)
        if stale:
            warnings.warn(
                f"{cls.__name__}: dropping state keys that are no longer fields: "
                f"{stale}. Written by an older TorchRef version.",
                stacklevel=3,
            )
            state = {k: v for k, v in state.items() if k in known}
        return state

    @classmethod
    def _from_state(cls, state: Dict[str, Any], device=None) -> "CrystalDataset":
        """Rebuild from a :meth:`_get_state` dict, on ``device`` (default the
        configured one). Pops ``"masks"`` from ``state``, so the caller's dict
        is mutated.
        """
        from torchref.utils.utils import TensorMasks

        device = normalize_device(device)

        masks_state = state.pop("masks", {})
        state = cls._drop_stale_state_keys(state)

        if "device" in state:
            state["device"] = torch.device(state["device"])

        # Spacegroup stays a string here; subclasses that want an object rewrap.
        if "cell" in state and state["cell"] is not None:
            if isinstance(state["cell"], torch.Tensor):
                # Conform the reloaded cell to the config float dtype rather than
                # pinning float32: a dataset saved and reloaded under a float64
                # config otherwise carries a float32 cell into reciprocal-basis math.
                state["cell"] = Cell(
                    state["cell"], dtype=get_float_dtype(), device=device
                )

        obj = cls(**state)

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
            Loaded dataset, of the class this was called on.
        """
        state = torch.load(path, map_location="cpu")
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
