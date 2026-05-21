"""
Reflection data container for crystallographic datasets.

This module provides the ReflectionData class for handling single-crystal
reflection data including Miller indices, structure factor amplitudes,
intensities, and R-free flags.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from torchref.config import dtypes, get_default_device
from torch.nn import Parameter

from torchref.symmetry import SpaceGroup

import numpy as np
import pandas as pd
import torch

from torchref.io import cif, mtz
from torchref.io.datasets.base import CrystalDataset
from torchref.base import math_torch
from torchref.base.french_wilson import FrenchWilson
from torchref.symmetry import Cell
from torchref.utils.debug_utils import DebugMixin

if TYPE_CHECKING:
    from torchref.model.model_ft import ModelFT

# Suppress PyTorch MaskedTensor prototype warnings globally
# MaskedTensor is stable enough for our use case (aggregations, element-wise ops)
warnings.filterwarnings(
    "ignore", message=".*MaskedTensors is in prototype stage.*", category=UserWarning
)

if TYPE_CHECKING:
    from torch.masked import MaskedTensor


@dataclass
class ReflectionData(CrystalDataset, DebugMixin):
    """
    Container for crystallographic reflection data.

    This class handles loading, processing, and accessing reflection data
    including Miller indices, structure factor amplitudes, intensities,
    and R-free flags. All data is stored as PyTorch tensors for GPU
    acceleration.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level for logging (0=silent, 1=normal, 2=debug). Default is 1.
    device : str, optional
        Device to store tensors on ('cpu', 'cuda', 'cuda:0', etc.). Defaults to the configured device.current.

    Attributes
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N, 3), dtype int32.
    F : torch.Tensor
        Structure factor amplitudes of shape (N,), dtype float32.
    F_sigma : torch.Tensor
        Amplitude uncertainties of shape (N,), dtype float32.
    I : torch.Tensor
        Intensities of shape (N,), dtype float32.
    I_sigma : torch.Tensor
        Intensity uncertainties of shape (N,), dtype float32.
    rfree_flags : torch.Tensor
        R-free test set flags of shape (N,), dtype bool.
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str
        Space group symbol.
    resolution : torch.Tensor
        Resolution per reflection in Ångströms of shape (N,).
    wilson_b : float
        Overall Wilson B-factor in Ų.

    Examples
    --------
    Load reflection data from an MTZ file::

        data = ReflectionData(verbose=1, device='cuda')
        data.load_mtz('data.mtz')
        print(f"Loaded {len(data.hkl)} reflections")
        print(f"Resolution range: {data.resolution.min():.2f} - {data.resolution.max():.2f} Å")
    """

    # Additional fields specific to ReflectionData (beyond CrystalDataset)
    # Note: Most fields are inherited from CrystalDataset dataclass

    # Cached properties (not serialized)
    _centric: Optional[torch.Tensor] = field(default=None, repr=False)
    _n_bins: Optional[int] = field(default=None, repr=False)
    _FrenchWilson: Optional[FrenchWilson] = field(default=None, repr=False)

    # Dynamic fields used by various methods
    source: Optional["ReflectionData"] = field(default=None, repr=False)
    dataset: Optional[pd.DataFrame] = field(default=None, repr=False)
    last_op: Optional[str] = field(default=None, repr=False)
    reader: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        """
        Initialize non-dataclass attributes after dataclass init.

        This is called automatically after the dataclass __init__.
        """
        # Call parent __post_init__ to initialize masks
        super().__post_init__()
        self.setup_scale()
        self.setup_anisotropy()

    def _canonicalize_in_place(self) -> None:
        """Remap HKL to canonical CCP4 ASU form and reorder all data in-place."""
        from dataclasses import fields as dc_fields
        from torchref.symmetry.reciprocal_symmetry import canonicalize_hkl

        if self.hkl is None or self.spacegroup is None:
            return

        canonical_hkl, phase_shifts, friedel_flags, sort_indices = canonicalize_hkl(
            self.hkl, self.spacegroup, include_friedel=True, device=self.device
        )

        n_refl = len(self.hkl)

        # Reorder every per-reflection tensor field in-place
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor) and val.shape and val.shape[0] == n_refl:
                setattr(self, f.name, val[sort_indices])

        # Overwrite HKL with canonical form
        self.hkl = canonical_hkl

        # Apply phase correction
        if self.phase is not None:
            self.phase = torch.where(friedel_flags, -self.phase, self.phase) + phase_shifts

        # Recalculate resolution from canonical HKL
        if self.cell is not None:
            self._calculate_resolution()

        # Reorder masks
        if hasattr(self, "masks") and self.masks is not None:
            for name in list(self.masks.keys()):
                mask_tensor = self.masks[name]
                if mask_tensor is not None:
                    # Bypass __setitem__ validation (reordering preserves True count)
                    dict.__setitem__(self.masks, name, mask_tensor[sort_indices])
            self.masks._updated = True

        # Reorder DataFrame
        if hasattr(self, "dataset") and self.dataset is not None:
            self.dataset = self.dataset.iloc[sort_indices.cpu().numpy()].copy()

        # Record anomalous bookkeeping. friedel_flags is already returned in the
        # sorted (canonical) order, matching self.hkl. hkl_anomalous carries the
        # signed Miller index used for structure-factor evaluation: the canonical
        # index for the (+) member and its negation for the conjugated (-) mate,
        # so the model produces a genuine Bijvoet difference when anomalous
        # scattering is present. See ReflectionData.hkl_for_sf.
        self.friedel_flags = friedel_flags
        self.hkl_anomalous = torch.where(
            friedel_flags.unsqueeze(-1), -self.hkl, self.hkl
        )

    def hkl_for_sf(self) -> torch.Tensor:
        """Signed Miller indices for structure-factor evaluation.

        Returns ``hkl_anomalous`` when present so the two members of a Bijvoet
        pair (which share a canonical ASU index in :attr:`hkl`) are evaluated at
        their true ``+h``/``-h`` positions and therefore get distinct
        ``|F_calc|`` under anomalous scattering. Falls back to the canonical
        :attr:`hkl` when anomalous bookkeeping is unavailable (e.g. empty init).

        Returns
        -------
        torch.Tensor
            Miller indices of shape (N, 3), row-aligned with :attr:`hkl`.
        """
        if self.hkl_anomalous is not None:
            return self.hkl_anomalous
        return self.hkl

    def load(self, reader):
        """
        Load reflection data using a data reader.

        Parameters
        ----------
        reader : callable
            Data reader object that returns (data_dict, cell, spacegroup) when called.
            Can be MTZ, ReflectionCIFReader, or other compatible reader.

        Returns
        -------
        ReflectionData
            Self, for method chaining.

        Raises
        ------
        ValueError
            If unit cell parameters are missing or no amplitude/intensity data found.
        """

        data_dict, cell, spacegroup = reader()

        hkl = torch.tensor(
            data_dict["HKL"], dtype=dtypes.int, device=self.device, requires_grad=False
        )

        self.hkl = hkl

        if cell is not None:
            self.cell = Cell(cell, dtype=dtypes.float, device=self.device)
        else:
            raise ValueError(
                "Unit cell parameters are required in the data and could not be read."
            )

        if spacegroup is not None:
            self.spacegroup = SpaceGroup(spacegroup)
        self._calculate_resolution()

        if "I" in data_dict:
            self.I = torch.tensor(
                data_dict["I"],
                dtype=dtypes.float,
                device=self.device,
                requires_grad=False,
            )
            if "SIGI" in data_dict:
                self.I_sigma = torch.tensor(
                    data_dict["SIGI"],
                    dtype=dtypes.float,
                    device=self.device,
                    requires_grad=False,
                )
            self.intensity_source = data_dict.get("I_col", "Unknown")
            self._FrenchWilson = FrenchWilson(
                self.hkl, self.cell.data, self.spacegroup, verbose=self.verbose
            )
            F, F_sigma = self._FrenchWilson(self.I, self.I_sigma)
            self.F = F
            self.F_sigma = F_sigma
        elif "F" in data_dict:
            self.F = torch.tensor(
                data_dict["F"],
                dtype=dtypes.float,
                device=self.device,
                requires_grad=False,
            )
            if "SIGF" in data_dict:
                if data_dict["SIGF"] is not None:
                    self.F_sigma = torch.tensor(
                        data_dict["SIGF"],
                        dtype=dtypes.float,
                        device=self.device,
                        requires_grad=False,
                    )
                else:
                    sigF = math_torch.estimate_sigma_F(self.F)
                    self.F_sigma = sigF
            else:
                sigF = math_torch.estimate_sigma_F(self.F)
                self.F_sigma = sigF
            self.amplitude_source = data_dict.get("F_col", "Unknown")

        else:
            raise ValueError("No amplitude or intensity data found in MTZ file")

        if "R-free-flags" in data_dict:
            rfree = torch.tensor(
                data_dict["R-free-flags"], device=self.device, requires_grad=False
            )
            flagged = rfree < 0
            rfree = rfree.clip(min=0, max=1).to(torch.bool)
            self.rfree_flags = rfree
            self.masks["flagged_initial"] = ~flagged
        else:
            flagged = torch.zeros(
                len(self.hkl), dtype=torch.bool, device=self.device, requires_grad=False
            )
            self.masks["flagged_initial"] = ~flagged
            self._generate_rfree_flags(free_fraction=0.02, n_bins=20, min_per_bin=100)

        self._post_load_cleanup()

        return self

    def _post_load_cleanup(self) -> "ReflectionData":
        """
        Run all post-load processing steps on the reflection data.

        This is called automatically after ``load()`` and ``from_tensors()``.
        It performs:
        1. Resolution calculation from HKL + cell
        2. Initial flagging mask (marks all reflections as valid if not set)
        3. Canonicalization of HKL to CCP4 ASU
        4. Sanitization of F (mask NaN/Inf/non-positive)
        5. Suspicious sigma detection

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        if self.resolution is None:
            self._calculate_resolution()

        if "flagged_initial" not in self.masks:
            self.masks["flagged_initial"] = torch.ones(
                len(self.hkl), dtype=torch.bool, device=self.device
            )

        self._canonicalize_in_place()
        self.sanitize_F()
        self.flag_suspicious_sigma()
        return self

    @classmethod
    def from_tensors(
        cls,
        hkl: torch.Tensor,
        F: torch.Tensor,
        F_sigma: torch.Tensor,
        cell: "Cell",
        spacegroup: "SpaceGroup",
        rfree_flags: Optional[torch.Tensor] = None,
        device=get_default_device(),
        verbose: int = 1,
    ) -> "ReflectionData":
        """
        Construct ReflectionData directly from tensors.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices of shape (N, 3).
        F : torch.Tensor
            Structure factor amplitudes of shape (N,).
        F_sigma : torch.Tensor
            Amplitude uncertainties of shape (N,).
        cell : Cell
            Unit cell parameters.
        spacegroup : SpaceGroup
            Space group.
        rfree_flags : torch.Tensor, optional
            R-free flags of shape (N,), dtype bool. If None, flags are
            generated automatically (2% free fraction).
        device : str, optional
            Device for tensors. Defaults to the configured device.current.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        ReflectionData
            Fully initialized reflection data with all cleanup applied.
        """
        data = cls(device=device, verbose=verbose)
        data.hkl = hkl.to(device=data.device)
        data.F = F.to(device=data.device)
        data.F_sigma = F_sigma.to(device=data.device)
        data.cell = cell.to(device=data.device) if hasattr(cell, 'to') else Cell(cell, device=data.device)
        data.spacegroup = spacegroup if isinstance(spacegroup, SpaceGroup) else SpaceGroup(spacegroup)

        if rfree_flags is not None:
            data.rfree_flags = rfree_flags.to(device=data.device, dtype=torch.bool)
        else:
            data._calculate_resolution()
            data._generate_rfree_flags(free_fraction=0.02, n_bins=20, min_per_bin=100)

        data._post_load_cleanup()
        return data

    def load_mtz(
        self, path: str, column_names: Optional[dict] = None
    ) -> "ReflectionData":
        """
        Load reflection data from MTZ file.

        Parameters
        ----------
        path : str
            Path to MTZ file.
        column_names : dict, optional
            Explicit column name mapping to override automatic detection.
            Supported keys: ``"F"``, ``"SIGF"``, ``"I"``, ``"SIGI"``.
            Example: ``{"F": "DFo", "SIGF": "sig_DFo"}``.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        reader = mtz.MTZReader(
            verbose=self.verbose, column_names=column_names
        ).read(path)
        return self.load(reader)

    def load_cif(self, path: str, data_block: Optional[str] = None) -> "ReflectionData":
        """
        Load reflection data from CIF file.

        Parameters
        ----------
        path : str
            Path to CIF file.
        data_block : str, optional
            Specific data block name to read (e.g., 'r1vlmsf'). If None, reads
            the first data block. Useful for multi-dataset CIF files.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        self.reader = cif.ReflectionCIFReader(
            path, verbose=self.verbose, data_block=data_block
        )
        return self.load(self.reader)

    @staticmethod
    def list_cif_data_blocks(path: str) -> List[str]:
        """
        List all data blocks available in a CIF file without loading data.

        Useful for multi-dataset CIF files to inspect available blocks
        before loading a specific one.

        Parameters
        ----------
        path : str
            Path to CIF file.

        Returns
        -------
        list of str
            Names of all data blocks in the CIF file.

        Examples
        --------
        List and load a specific data block::

            blocks = ReflectionData.list_cif_data_blocks('1VLM-sf.cif')
            print(blocks)  # ['r1vlmsf', 'r1vlmAsf', 'r1vlmBsf', ...]
            data = ReflectionData().load_cif('1VLM-sf.cif', data_block=blocks[1])
        """
        return cif.list_data_blocks(path)

    def _generate_rfree_flags(
        self,
        free_fraction: float = 0.02,
        n_bins: int = 20,
        min_per_bin: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        """
        Generate R-free flags with resolution-stratified sampling.

        Ensures free reflections are evenly distributed across resolution
        shells for unbiased cross-validation.

        Parameters
        ----------
        free_fraction : float, optional
            Fraction of reflections to mark as free (0.02 = 2%). Default is 0.02.
        n_bins : int, optional
            Target number of resolution bins. Default is 20.
        min_per_bin : int, optional
            Minimum reflections per resolution bin. Default is 100.
        seed : int, optional
            Random seed for reproducibility. Default is None.

        Notes
        -----
        Algorithm:
        1. Bin reflections by resolution
        2. Ensure bins have at least min_per_bin reflections
        3. Randomly select free_fraction from each bin
        4. This ensures even distribution across all resolution ranges

        Raises
        ------
        ValueError
            If resolution information is not available.
        """
        if self.resolution is None:
            raise ValueError("Resolution information required to generate R-free flags")

        print("Generating R-free flags:")
        print(f"  Target free fraction: {free_fraction*100:.1f}%")
        print(f"  Target bins: {n_bins}")
        print(f"  Minimum per bin: {min_per_bin} reflections")

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        n_refl = len(self.resolution)

        # Create resolution bins
        bin_indices, actual_n_bins = self.get_bins(
            n_bins=n_bins, min_per_bin=min_per_bin
        )

        print(f"  Created {actual_n_bins} resolution bins")

        # Initialize all flags as work set (1)
        flags = torch.ones(n_refl, dtype=dtypes.int, device=self.device, requires_grad=False)

        # Sample free reflections from each bin
        total_free = 0
        for bin_idx in range(actual_n_bins):
            bin_mask = bin_indices == bin_idx
            bin_size = bin_mask.sum().item()

            if bin_size == 0:
                continue

            # Number of free reflections in this bin
            # Ensure at least 1, but respect the free_fraction
            n_free_in_bin = max(1, int(bin_size * free_fraction))

            # Get indices of reflections in this bin
            bin_refl_indices = torch.where(bin_mask)[0]

            # Randomly select free reflections
            perm = torch.randperm(bin_size)[:n_free_in_bin]
            free_indices = bin_refl_indices[perm]

            # Mark as free (0)
            flags[free_indices] = 0
            total_free += n_free_in_bin

        # Move to correct device and assign
        flags_tensor = flags.to(dtype=dtypes.int, device=self.device)
        self.rfree_flags = flags_tensor
        self.rfree_source = "Generated (resolution-binned)"

        n_free = (flags == 0).sum().item()
        n_work = (flags != 0).sum().item()
        free_pct = 100.0 * n_free / n_refl

        print(
            f"  ✓ Generated flags: {n_free} free ({free_pct:.1f}%), {n_work} work ({100-free_pct:.1f}%)"
        )
        print("  Flags are resolution-binned for unbiased validation")

    def get_bins(
        self, n_bins: int = 20, min_per_bin: int = 100
    ) -> Tuple[torch.Tensor, int]:
        """
        Create resolution bins with approximately equal reflection counts.

        Parameters
        ----------
        n_bins : int, optional
            Target number of resolution bins. Default is 20.
        min_per_bin : int, optional
            Minimum reflections per bin. Default is 100.

        Returns
        -------
        bin_indices : torch.Tensor
            Tensor of shape (N,) with bin index for each reflection.
        n_bins : int
            Actual number of bins created (may be less than target for small datasets).
        """
        n_refl = len(self.resolution)
        valid_mask = self.masks()
        total_valid = valid_mask.sum().item()

        # Calculate how many bins we can actually create given min_per_bin constraint
        max_possible_bins = max(1, total_valid // min_per_bin)
        actual_n_bins = min(n_bins, max_possible_bins)

        if actual_n_bins < n_bins and self.verbose > 0:
            print(
                f"  Note: Adjusted bins from {n_bins} to {actual_n_bins} (min {min_per_bin} refl/bin)"
            )

        # Sort reflections by resolution
        _, sort_indices = torch.sort(self.resolution)

        # Create bins with approximately equal number of VALID reflections
        bin_indices = torch.zeros(n_refl, dtype=dtypes.int, device=self.device)
        reflections_per_bin = total_valid // actual_n_bins

        # Get the valid mask in sorted order
        valid_mask_sorted = valid_mask[sort_indices]

        # Cumulative sum of valid reflections in sorted order
        cumsum_valid = torch.cumsum(valid_mask_sorted.to(dtypes.int), dim=0)

        # Create bin edges based on cumulative count of valid reflections
        # Each bin should contain approximately reflections_per_bin valid reflections
        bin_edges = [0]
        for bin_idx in range(1, actual_n_bins):
            target_count = bin_idx * reflections_per_bin
            # Find first index where cumsum >= target_count
            edge_candidates = torch.where(cumsum_valid >= target_count)[0]
            if len(edge_candidates) > 0:
                bin_edges.append(edge_candidates[0].item())
        bin_edges.append(n_refl)

        # Assign bin indices to sorted reflections, then map back to original order
        for bin_idx in range(len(bin_edges) - 1):
            start, end = bin_edges[bin_idx], bin_edges[bin_idx + 1]
            bin_indices[sort_indices[start:end]] = bin_idx

        if self.verbose > 1:
            # Print bin statistics
            print("  Resolution bins:")
            for bin_idx in range(min(actual_n_bins, 20)):  # Show all bins (up to 20)
                bin_mask = bin_indices == bin_idx
                if bin_mask.sum() > 0:
                    valid_reflexes = bin_mask & valid_mask
                    bin_res = self.resolution[bin_mask]
                    print(
                        f"    Bin {bin_idx:2d}: {valid_reflexes.sum():6d} valid refl, "
                        f"resolution {bin_res.min():.2f}-{bin_res.max():.2f} Å"
                    )
            if actual_n_bins > 20:
                print(f"    ... ({actual_n_bins - 20} more bins)")
        self.bin_indices = bin_indices
        self._n_bins = actual_n_bins
        return bin_indices, actual_n_bins

    def mean_res_per_bin(self) -> torch.Tensor:
        """
        Calculate mean resolution for each bin.

        Returns
        -------
        torch.Tensor
            Mean resolution for each bin in Ångströms.

        Raises
        ------
        ValueError
            If bins have not been created yet.
        """
        if self.bin_indices is None or self.resolution is None:
            raise ValueError("Bins have not been created yet")

        mean_resolutions = torch.zeros(
            self._n_bins, dtype=dtypes.float, device=self.device
        )
        count_per_bin = torch.zeros(self._n_bins, dtype=dtypes.int, device=self.device)
        mask = self.masks()
        mean_resolutions = torch.scatter_add(
            mean_resolutions,
            0,
            self.bin_indices[mask].to(torch.int64),
            self.resolution[mask],
        )
        count_per_bin = torch.scatter_add(
            count_per_bin,
            0,
            self.bin_indices[mask].to(torch.int64),
            torch.ones_like(self.resolution[mask], dtype=dtypes.int),
        )
        mean_resolutions = mean_resolutions / count_per_bin.clamp(min=1).float()
        return mean_resolutions

    def mean_F_per_bin(self) -> torch.Tensor:
        """
        Calculate mean structure factor amplitude per resolution bin.

        Returns
        -------
        torch.Tensor
            Mean F per bin of shape (n_bins,).

        Raises
        ------
        ValueError
            If bins have not been created yet.
        """
        if self.bin_indices is None:
            self.get_bins()
        if self.F is None:
            raise ValueError("No amplitude data loaded")

        mean_F = torch.zeros(self._n_bins, dtype=dtypes.float, device=self.device)
        count_per_bin = torch.zeros(self._n_bins, dtype=dtypes.int, device=self.device)
        mask = self.masks()
        mean_F = torch.scatter_add(
            mean_F, 0, self.bin_indices[mask].to(torch.int64), self.F[mask]
        )
        count_per_bin = torch.scatter_add(
            count_per_bin,
            0,
            self.bin_indices[mask].to(torch.int64),
            torch.ones_like(self.F[mask], dtype=dtypes.int),
        )
        mean_F = mean_F / count_per_bin.clamp(min=1).float()
        return mean_F

    def mean_sigma_per_bin(self) -> Optional[torch.Tensor]:
        """
        Calculate mean structure factor uncertainty per resolution bin.

        Returns
        -------
        torch.Tensor or None
            Mean sigma_F per bin of shape (n_bins,), or None if no uncertainties.

        Raises
        ------
        ValueError
            If bins have not been created yet.
        """
        if self.bin_indices is None:
            self.get_bins()
        if self.F is None:
            raise ValueError("No amplitude data loaded")

        mean_sigma = torch.zeros(self._n_bins, dtype=dtypes.float, device=self.device)
        count_per_bin = torch.zeros(self._n_bins, dtype=dtypes.int, device=self.device)
        mask = self.masks()
        mean_sigma = torch.scatter_add(
            mean_sigma, 0, self.bin_indices[mask].to(torch.int64), self.F_sigma[mask]
        )
        count_per_bin = torch.scatter_add(
            count_per_bin,
            0,
            self.bin_indices[mask].to(torch.int64),
            torch.ones_like(self.F_sigma[mask], dtype=dtypes.int),
        )
        mean_sigma = mean_sigma / count_per_bin.clamp(min=1).float()
        return mean_sigma

    def regenerate_rfree_flags(
        self,
        free_fraction: float = 0.02,
        n_bins: int = 20,
        min_per_bin: int = 100,
        seed: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """
        Regenerate R-free flags with resolution-stratified sampling.

        Parameters
        ----------
        free_fraction : float, optional
            Fraction of reflections to mark as free. Default is 0.02 (2%).
        n_bins : int, optional
            Target number of resolution bins. Default is 20.
        min_per_bin : int, optional
            Minimum reflections per resolution bin. Default is 100.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        force : bool, optional
            If True, regenerate even if flags already exist. Default is False.

        Examples
        --------
        Generate R-free flags::

            # Generate 2% free reflections with reproducible seed
            data.regenerate_rfree_flags(free_fraction=0.02, n_bins=20, seed=42)
            # Generate 5% free with 10 bins, overwriting existing
            data.regenerate_rfree_flags(free_fraction=0.05, n_bins=10, force=True)
        """
        if self.rfree_flags is not None and not force:
            print("⚠️  WARNING: R-free flags already exist!")
            print(f"   Current source: {self.rfree_source}")
            print("   Use force=True to overwrite existing flags")
            return

        if self.rfree_flags is not None and force:
            print("⚠️  WARNING: Overwriting existing R-free flags")
            print(f"   Old source: {self.rfree_source}")

        self._generate_rfree_flags(
            free_fraction=free_fraction,
            n_bins=n_bins,
            min_per_bin=min_per_bin,
            seed=seed,
        )

    def _calculate_resolution(self) -> None:
        """
        Calculate resolution for each reflection from Miller indices.

        Sets the `resolution` buffer with d-spacing values in Ångströms.

        Raises
        ------
        ValueError
            If Miller indices or unit cell parameters are missing.
        """
        if self.hkl is None:
            raise ValueError(
                "Miller indices (hkl) are required to calculate resolution"
            )
        if self.cell is None:
            raise ValueError(
                "Unit cell parameters are required to calculate resolution"
            )
        s = math_torch.get_scattering_vectors(self.hkl, self.cell.data)
        resolution = 1.0 / torch.linalg.norm(s, axis=1)
        self.resolution = resolution

    def _calculate_wilson_b(self, n_bins: int = 30) -> None:
        """
        Calculate Wilson B-factors from structure factor amplitudes.

        Fits a two-component model separating structure and solvent contributions:
        <F²> ∝ A_struct * exp(-2*B_struct*s²) + A_sol * exp(-2*B_sol*s²)

        The fitting proceeds in three stages:
        1. Estimate B_structure from high-resolution data (d < 3.5 Å) where solvent is negligible
        2. Estimate B_solvent from low-resolution data (d > 6 Å) where solvent dominates
        3. Refine both together with two-exponential fit across all data

        Args:
            n_bins: Number of resolution bins for averaging (default: 30)

        Sets:
            self.wilson_b: Overall Wilson B-factor (weighted average) in Å²
            self.wilson_b_structure: Structure B-factor from high-res in Å²
            self.wilson_b_solvent: Solvent B-factor from low-res in Å²
            self.wilson_k_sol: Relative solvent contribution (0-1)
        """
        if self.F is None or self.resolution is None:
            return

        # Get valid reflections
        F = self.F
        d = self.resolution
        valid = torch.isfinite(F) & (F > 0) & torch.isfinite(d)

        if valid.sum() < 100:
            if self.verbose > 0:
                print(
                    f"  Wilson B: insufficient data ({valid.sum()} reflections), skipping"
                )
            return

        F_valid = F[valid]
        d_valid = d[valid]

        # Calculate s² = 1/(4d²)
        s_sq = 1.0 / (4.0 * d_valid**2)
        F_sq = F_valid**2

        # Bin the data for noise reduction
        s_sq_min, s_sq_max = s_sq.min(), s_sq.max()
        bin_edges = torch.linspace(s_sq_min, s_sq_max, n_bins + 1, device=self.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = torch.bucketize(s_sq, bin_edges[1:-1])

        # Calculate mean F² per bin
        bin_sums = torch.zeros(n_bins, device=self.device, dtype=F_sq.dtype)
        bin_counts = torch.zeros(n_bins, device=self.device, dtype=F_sq.dtype)
        bin_sums.scatter_add_(0, bin_idx, F_sq)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(F_sq))

        valid_bins = bin_counts > 5
        if valid_bins.sum() < 5:
            if self.verbose > 0:
                print(f"  Wilson B: insufficient bins ({valid_bins.sum()}), skipping")
            return

        mean_F_sq = bin_sums[valid_bins] / bin_counts[valid_bins]
        s_sq_bins = bin_centers[valid_bins]

        # Convert s² back to d-spacing for resolution-based selection
        d_bins = 1.0 / (2.0 * torch.sqrt(s_sq_bins))

        # Stage 1: Fit high-resolution region (d < 3.5 Å) for structure B
        high_res_mask = d_bins < 3.5
        B_struct = self._fit_single_wilson(
            s_sq_bins, mean_F_sq, high_res_mask, "high-res"
        )

        # Stage 2: Fit low-resolution region (d > 6 Å) for solvent B
        low_res_mask = d_bins > 6.0
        B_sol = self._fit_single_wilson(s_sq_bins, mean_F_sq, low_res_mask, "low-res")

        # Stage 3: Two-component fit across all data
        B_struct_final, B_sol_final, k_sol = self._fit_two_component_wilson(
            s_sq_bins, mean_F_sq, B_struct, B_sol
        )

        # Store results
        self.wilson_b_structure = B_struct_final
        self.wilson_b_solvent = B_sol_final
        self.wilson_k_sol = k_sol

        # Overall Wilson B is the structure B (what people usually mean by "Wilson B")
        self.wilson_b = B_struct_final

        if self.verbose > 0:
            print(f"  Wilson B-factor (structure): {B_struct_final:.1f} Å²")
            print(f"  Wilson B-factor (solvent):   {B_sol_final:.1f} Å²")
            print(f"  Solvent fraction (k_sol):    {k_sol:.3f}")

    def _fit_single_wilson(
        self,
        s_sq: torch.Tensor,
        mean_F_sq: torch.Tensor,
        mask: torch.Tensor,
        label: str,
    ) -> float:
        """
        Fit single-exponential Wilson plot to selected resolution range.

        Args:
            s_sq: s² values for bins
            mean_F_sq: Mean F² values for bins
            mask: Boolean mask selecting which bins to use
            label: Label for error messages

        Returns:
            B-factor from fit (Å²)
        """
        if mask.sum() < 3:
            # Not enough data, return reasonable default
            if self.verbose > 1:
                print(
                    f"    Wilson {label}: insufficient bins ({mask.sum()}), using default"
                )
            return 50.0 if "struct" in label else 200.0

        x = s_sq[mask]
        y = torch.log(mean_F_sq[mask])

        # Linear regression: ln(F²) = const - 2B*s²
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator < 1e-12:
            return 50.0 if "struct" in label else 200.0

        slope = numerator / denominator
        B = -slope.item() / 2.0

        # Sanity bounds
        B = max(0.0, min(B, 300.0))

        return B

    def _fit_two_component_wilson(
        self,
        s_sq: torch.Tensor,
        mean_F_sq: torch.Tensor,
        B_struct_init: float,
        B_sol_init: float,
        n_iter: int = 50,
    ) -> Tuple[float, float, float]:
        """
        Fit two-component Wilson model using iterative refinement.

        Model: F² = A_struct * exp(-2*B_struct*s²) + A_sol * exp(-2*B_sol*s²)

        Parameterized as:
            F² = A * [(1-k)*exp(-2*B_struct*s²) + k*exp(-2*B_sol*s²)]

        where k is the relative solvent contribution at s²=0.

        Parameters
        ----------
        s_sq : torch.Tensor
            s² values for bins.
        mean_F_sq : torch.Tensor
            Mean F² values for bins.
        B_struct_init : float
            Initial structure B-factor.
        B_sol_init : float
            Initial solvent B-factor.
        n_iter : int, optional
            Number of refinement iterations. Default is 50.

        Returns
        -------
        B_struct : float
            Refined structure B-factor.
        B_sol : float
            Refined solvent B-factor.
        k_sol : float
            Relative solvent contribution.
        """
        # Normalize F² for numerical stability
        F_sq_max = mean_F_sq.max()
        y = mean_F_sq / F_sq_max
        x = s_sq

        # Initialize parameters
        B_struct = torch.tensor(B_struct_init, device=self.device, dtype=x.dtype)
        B_sol = torch.tensor(B_sol_init, device=self.device, dtype=x.dtype)

        # Estimate initial k from ratio of low-res to high-res decay
        # At low resolution, solvent contributes more
        d_from_s = 1.0 / (2.0 * torch.sqrt(x))
        low_res_val = y[d_from_s > 5.0].mean() if (d_from_s > 5.0).any() else y[0]
        high_res_val = y[d_from_s < 3.0].mean() if (d_from_s < 3.0).any() else y[-1]

        # k estimates solvent fraction - if low res is much higher than expected
        # from structure alone, there's solvent contribution
        struct_decay = torch.exp(-2 * B_struct * x)
        expected_low = (
            struct_decay[d_from_s > 5.0].mean()
            if (d_from_s > 5.0).any()
            else struct_decay[0]
        )

        if expected_low > 1e-6 and low_res_val > expected_low:
            k_init = min(0.5, (low_res_val - expected_low).item() / low_res_val.item())
        else:
            k_init = 0.1

        k = torch.tensor(max(0.01, min(0.5, k_init)), device=self.device, dtype=x.dtype)

        # Simple gradient descent refinement
        lr = 0.1

        for _ in range(n_iter):
            # Compute model
            struct_term = (1 - k) * torch.exp(-2 * B_struct * x)
            sol_term = k * torch.exp(-2 * B_sol * x)
            model = struct_term + sol_term

            # Compute scale factor analytically
            A = (y * model).sum() / (model * model).sum()
            model_scaled = A * model

            # Compute gradients (simplified, using finite differences for robustness)
            eps = 0.1

            # B_struct gradient
            model_plus = A * ((1 - k) * torch.exp(-2 * (B_struct + eps) * x) + sol_term)
            model_minus = A * (
                (1 - k) * torch.exp(-2 * (B_struct - eps) * x) + sol_term
            )
            loss_plus = ((y - model_plus) ** 2).sum()
            loss_minus = ((y - model_minus) ** 2).sum()
            grad_B_struct = (loss_plus - loss_minus) / (2 * eps)

            # B_sol gradient
            model_plus = A * (struct_term + k * torch.exp(-2 * (B_sol + eps) * x))
            model_minus = A * (struct_term + k * torch.exp(-2 * (B_sol - eps) * x))
            loss_plus = ((y - model_plus) ** 2).sum()
            loss_minus = ((y - model_minus) ** 2).sum()
            grad_B_sol = (loss_plus - loss_minus) / (2 * eps)

            # k gradient
            eps_k = 0.01
            k_plus = min(0.9, k + eps_k)
            k_minus = max(0.01, k - eps_k)
            model_plus = A * (
                (1 - k_plus) * torch.exp(-2 * B_struct * x)
                + k_plus * torch.exp(-2 * B_sol * x)
            )
            model_minus = A * (
                (1 - k_minus) * torch.exp(-2 * B_struct * x)
                + k_minus * torch.exp(-2 * B_sol * x)
            )
            loss_plus = ((y - model_plus) ** 2).sum()
            loss_minus = ((y - model_minus) ** 2).sum()
            grad_k = (loss_plus - loss_minus) / (2 * eps_k)

            # Update parameters
            B_struct = B_struct - lr * grad_B_struct
            B_sol = B_sol - lr * grad_B_sol
            k = k - lr * 0.1 * grad_k  # Slower learning rate for k

            # Enforce constraints
            B_struct = torch.clamp(B_struct, 1.0, 200.0)
            B_sol = torch.clamp(B_sol, 50.0, 500.0)
            k = torch.clamp(k, 0.01, 0.9)

            # Ensure B_sol > B_struct (solvent is more disordered)
            if B_sol < B_struct + 20:
                B_sol = B_struct + 20

        return B_struct.item(), B_sol.item(), k.item()

    def get_structure_factors(self, as_complex: bool = False) -> torch.Tensor:
        """
        Get structure factors, optionally as complex numbers.

        Parameters
        ----------
        as_complex : bool, optional
            If True and phases available, return F*exp(i*phi). Default is False.

        Returns
        -------
        torch.Tensor
            Structure factor amplitudes or complex structure factors.

        Raises
        ------
        ValueError
            If no amplitude data is loaded.
        """
        if self.F is None:
            raise ValueError("No amplitude data loaded")

        if as_complex and self.phase is not None:
            return self.F * torch.exp(1j * self.phase)
        else:
            return self.F

    def get_structure_factors_with_sigma(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factor amplitudes and their uncertainties.

        Returns
        -------
        F : torch.Tensor
            Structure factor amplitudes of shape (N,).
        F_sigma : torch.Tensor or None
            Uncertainties of shape (N,), or None if not available.

        Raises
        ------
        ValueError
            If no amplitude data is loaded.

        Examples
        --------
        Get amplitudes with uncertainties::

            F, sigma_F = data.get_structure_factors_with_sigma()
            if sigma_F is not None:
                weighted_residual = (F_obs - F_calc) / sigma_F
        """
        if self.F is None:
            raise ValueError("No amplitude data loaded")

        return self.F, self.F_sigma

    def get_hkl(self):
        """
        Return Miller indices for valid reflections.

        Returns
        -------
        torch.Tensor
            Miller indices of shape (N, 3), dtype int32.

        Raises
        ------
        ValueError
            If no Miller indices are loaded.
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded")
        return self.hkl[self.masks()]

    def filter_by_resolution(
        self, d_min: Optional[float] = None, d_max: Optional[float] = None
    ) -> "ReflectionData":
        """
        Filter reflections by resolution range.

        Adds a boolean mask to self.masks for the specified resolution range.

        Parameters
        ----------
        d_min : float, optional
            Minimum resolution / high resolution cutoff (e.g., 1.5 Å).
        d_max : float, optional
            Maximum resolution / low resolution cutoff (e.g., 50.0 Å).

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        if self.resolution is None:
            self._calculate_resolution()

        mask = torch.ones(len(self.hkl), dtype=torch.bool, device=self.device)

        if d_min is not None:
            mask &= self.resolution >= d_min
        if d_max is not None:
            mask &= self.resolution <= d_max

        self.masks["resolution"] = mask

        valid = self.masks().sum().item()
        print(
            f"Filtering: {mask.sum()}/{len(mask)} reflections in range "
            f"[{d_max if d_max else 'inf'} - {d_min if d_min else 'inf'}] "
            f"\u00c5 ({valid} valid after all masks)"
        )

        return self

    def get_mask(self):
        """
        Return combined mask from all active filters.

        Returns
        -------
        torch.Tensor
            Boolean mask combining all filter conditions.
        """

    def cut_res(
        self, highres: Optional[float] = None, lowres: Optional[float] = None
    ) -> "ReflectionData":
        """
        Filter reflections by resolution range.

        Alias for filter_by_resolution with more intuitive naming.

        Parameters
        ----------
        highres : float, optional
            High resolution cutoff (small d-spacing, e.g., 1.5 Å).
            Keeps reflections with d >= highres.
        lowres : float, optional
            Low resolution cutoff (large d-spacing, e.g., 50.0 Å).
            Keeps reflections with d <= lowres.

        Returns
        -------
        ReflectionData
            Self, for method chaining.

        Examples
        --------
        Filter by resolution::

            # Keep reflections between 50 Å and 1.5 Å
            filtered = data.cut_res(highres=1.5, lowres=50.0)
            # Keep only high-resolution data (< 2 Å)
            high_res = data.cut_res(highres=1.0, lowres=2.0)
        """
        return self.filter_by_resolution(d_min=highres, d_max=lowres)

    def get_rfree_masks(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get boolean masks for work and test (free) sets.

        Returns
        -------
        work_mask : torch.Tensor or None
            Boolean tensor for work set (flag != 0).
        test_mask : torch.Tensor or None
            Boolean tensor for test/free set (flag == 0).
            Both are None if no R-free flags are available.

        Examples
        --------
        Separate work and test sets::

            work_mask, test_mask = data.get_rfree_masks()
            if work_mask is not None:
                F_work = data.F[work_mask]
                F_test = data.F[test_mask]
        """
        if self.rfree_flags is None:
            return None, None

        work_mask = self.rfree_flags != 0
        test_mask = self.rfree_flags == 0

        return work_mask, test_mask

    def get_work_set(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factors for the work set (R-free flag != 0).

        Returns
        -------
        F_work : torch.Tensor
            Structure factors for work set.
        sigma_work : torch.Tensor or None
            Uncertainties for work set, or None if not available.

        Notes
        -----
        Returns full dataset with warning if no R-free flags available.
        """
        if self.rfree_flags is None:
            print("WARNING: No R-free flags available, returning full dataset")
            return self.F, self.F_sigma

        work_mask = self.rfree_flags != 0
        F_work = self.F[work_mask] if self.F is not None else None
        sigma_work = self.F_sigma[work_mask] if self.F_sigma is not None else None

        return F_work, sigma_work

    def get_test_set(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factors for the test set (R-free flag == 0).

        Returns
        -------
        F_test : torch.Tensor
            Structure factors for test/free set.
        sigma_test : torch.Tensor or None
            Uncertainties for test set, or None if not available.

        Raises
        ------
        ValueError
            If no R-free flags are available.
        """
        if self.rfree_flags is None:
            raise ValueError("No R-free flags available in dataset")

        test_mask = self.rfree_flags == 0
        F_test = self.F[test_mask] if self.F is not None else None
        sigma_test = self.F_sigma[test_mask] if self.F_sigma is not None else None

        return F_test, sigma_test

    def get_max_res(self) -> Optional[float]:
        """
        Return maximum resolution (lowest d-spacing).

        Returns
        -------
        float
            Maximum resolution in Ångströms.
        """
        if self.resolution is None:
            self._calculate_resolution()
        mask = self.masks()
        return float(self.resolution[mask].min().item())
    
    def get_min_res(self) -> Optional[float]:
        """
        Return minimum resolution (highest d-spacing).

        Returns
        -------
        float
            Minimum resolution in Ångströms.
        """
        if self.resolution is None:
            self._calculate_resolution()
        mask = self.masks()
        return float(self.resolution[mask].max().item())

    def __len__(self) -> int:
        """
        Return number of reflections.

        Returns
        -------
        int
            Number of reflections in the dataset.
        """
        return len(self.hkl) if self.hkl is not None else 0

    @property
    def d_min(self) -> Optional[float]:
        """
        Return maximum resolution (lowest d-spacing).

        Returns
        -------
        float
            Maximum resolution in Ångströms.
        """
        return self.get_max_res()


    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            Summary of reflection data including count, sources, and resolution.
        """
        if self.hkl is None:
            return "ReflectionData(empty)"

        parts = [f"ReflectionData(n={len(self.hkl)}"]
        if self.amplitude_source:
            parts.append(f"F={self.amplitude_source}")
        if self.phase_source:
            parts.append(f"φ={self.phase_source}")
        if self.resolution is not None:
            parts.append(f"d={self.resolution.min():.2f}-{self.resolution.max():.2f}Å")
        parts.append(f"sg={self.spacegroup}")

        return ", ".join(parts) + ")"

    def get_valid_mask(self) -> torch.Tensor:
        """
        Return the combined validity mask for all reflections.

        This is the mask used to filter reflections in forward(). True indicates
        a valid (included) reflection, False indicates an excluded one.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (N,) where N is the total number of reflections.
            True = valid/included, False = invalid/excluded.

        Examples
        --------
        Check validity::

            mask = data.get_valid_mask()
            print(f"{mask.sum()} of {len(mask)} reflections are valid")
        """
        return self.masks()

    def data_indexed(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Return reflection data as indexed (filtered) tensors.

        This method filters out invalid reflections and returns smaller tensors
        containing only valid data. Useful for operations that don't support
        MaskedTensors or for writing output files.

        Returns
        -------
        hkl : torch.Tensor
            Miller indices of shape (M, 3) where M is number of valid reflections.
        F : torch.Tensor
            Structure factor amplitudes of shape (M,).
        F_sigma : torch.Tensor or None
            Uncertainties of shape (M,) or None.
        rfree_flags : torch.Tensor or None
            R-free flags of shape (M,) or None.

        See Also
        --------
        forward : Main method returning MaskedTensors.

        Examples
        --------
        Get indexed data for file writing::

            hkl, F, sigma, rfree = data.data_indexed()
            F_np = F.cpu().numpy()  # Safe for writing to files
        """
        to_mask = self.masks()

        hkl = self.hkl[to_mask]
        F = self.F[to_mask]
        F_sigma = self.F_sigma[to_mask] if self.F_sigma is not None else None
        rfree_flags = (
            self.rfree_flags[to_mask].to(torch.bool)
            if self.rfree_flags is not None
            else None
        )

        return hkl, F, F_sigma, rfree_flags

    def __call__(
        self, mask: bool = True, scale: bool = True
    ) -> Tuple[torch.Tensor, "MaskedTensor", "MaskedTensor", torch.Tensor]:
        """
        Return core reflection data with MaskedTensors for F and sigma.

        F and F_sigma are returned as MaskedTensors which keep all reflections
        but mark invalid ones as masked. Aggregation operations (sum, mean, etc.)
        automatically skip masked values. HKL and rfree_flags remain regular tensors.

        Parameters
        ----------
        mask : bool, optional
            If True, apply current masks to F and sigma. Default is True.
        scale : bool, optional
            If True, apply scaling to F and sigma before returning.

        Returns
        -------
        hkl : torch.Tensor
            Miller indices of shape (N, 3). Full size, unfiltered.
        F : MaskedTensor
            Structure factor amplitudes of shape (N,) with invalid reflections masked.
        F_sigma : MaskedTensor or None
            Uncertainties of shape (N,) with invalid reflections masked, or None.
        rfree_flags : torch.Tensor or None
            R-free flags of shape (N,) or None. Full size, unfiltered. 1=work, 0=free.

        Notes
        -----
        MaskedTensors:

        - Are PyTorch tensors with an associated boolean mask
        - Aggregations (sum, mean, etc.) skip masked values automatically
        - Element-wise operations preserve the mask
        - Use .get_data() and .get_mask() to access underlying data
        - Use .to_tensor(fill_value) to convert back to regular tensor
        - Note: MaskedTensor is in prototype stage in PyTorch

        Loss functions and targets extract valid data from MaskedTensors before
        computation to work correctly with complex F_calc values.

        Examples
        --------
        Access reflection data with MaskedTensors::

            hkl, F, sigma, rfree = data()
            print(F.shape)  # Full shape
            print(F.sum())  # Only sums valid (unmasked) values

            # Access underlying data
            valid_mask = F.get_mask()
            F_values = F.get_data()[valid_mask]
        """
        from torch.masked import MaskedTensor
        hkl, F, F_sigma, rfree_flags = self.hkl, self.F, self.F_sigma, self.rfree_flags

        if scale:
            F, F_sigma = self.get_corrected_data()
            
        if mask:
            to_mask = self.masks()
            if to_mask.sum() == 0:
                raise RuntimeError(
                    "All reflections are masked! Check your filters/masks."
                )
            F = MaskedTensor(F.detach().clone(), to_mask)
            if F_sigma is not None:
                F_sigma = MaskedTensor(F_sigma.detach().clone(), to_mask)
        return hkl, F, F_sigma, rfree_flags

    def data_fill_masked(
        self, mode="mean"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return data tensors with missing or flagged reflections filled with.

        args:
            mode: str, optional
                'mean' : fill missing/flagged with mean of present data
                'zero' : fill missing/flagged with zero
        """
        hkl, F, F_sigma, rfree = self()

        if mode == "mean":
            mean_F = self.mean_F_per_bin()
            mean_F_sigma = self.mean_sigma_per_bin()
            F_data = F.get_data().clone()
            F_sigma_data = F_sigma.get_data().clone()
            mask = F.get_mask()
            F_data[~mask] = mean_F[self.bin_indices[~mask]]
            F_sigma_data[~mask] = mean_F_sigma[self.bin_indices[~mask]]
            rfree[~mask] = True  # set missing to work set
            return hkl, F_data, F_sigma_data, rfree

        elif mode == "zero":
            mask = F.get_mask()
            F_data = F.get_data().clone()
            F_sigma_data = F_sigma.get_data().clone()
            F_data[~mask] = 0.0
            F_sigma_data[~mask] = 0.0
            rfree[~mask] = True  # set missing to work set
            return hkl, F_data, F_sigma_data, rfree

        else:
            raise ValueError(f"Unknown fill mode: {mode}")

    def __getitem__(self, key):
        """
        Index into the reflection dataset.

        Parameters
        ----------
        key : torch.Tensor
            Boolean mask or integer indices for selection.

        Returns
        -------
        ReflectionData
            New ReflectionData object with selected reflections.
        """
        if isinstance(key, torch.Tensor):
            return self.__select__(key)
        raise TypeError(f"Unsupported index type: {type(key)}")

    def __select__(self, indices: torch.Tensor, op=None) -> "ReflectionData":
        """
        Select reflections by boolean mask or integer indices.

        Iterates over all dataclass fields generically, so new tensor fields
        are handled automatically.

        Parameters
        ----------
        indices : torch.Tensor
            Boolean mask of shape (N,) or integer indices for selection.
        op : str, optional
            Operation name for tracking purposes.

        Returns
        -------
        ReflectionData
            New ReflectionData object with selected reflections.
        """
        from dataclasses import fields as dc_fields
        from torchref.utils.utils import TensorMasks

        n_refl = len(self.hkl) if self.hkl is not None else 0

        # Create new instance with same device
        selected = ReflectionData(verbose=self.verbose, device=self.device)

        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                if val.shape and val.shape[0] == n_refl:
                    setattr(selected, f.name, val[indices])
                else:
                    # Non-matching tensor (e.g. U_aniso shape (6,)): copy as-is
                    setattr(selected, f.name, val.clone())
            elif isinstance(val, Cell):
                setattr(selected, f.name, val.clone())
            else:
                # Scalars, strings, None, gemmi objects, etc.
                setattr(selected, f.name, val)

        # Handle masks (not a dataclass field)
        if hasattr(self, "masks") and self.masks is not None and len(self.masks) > 0:
            new_masks = TensorMasks(device=self.device)
            for name, mask_tensor in self.masks.items():
                if mask_tensor is not None:
                    new_masks[name] = mask_tensor[indices]
            selected.masks = new_masks

        # Handle DataFrame
        if hasattr(self, "dataset") and self.dataset is not None:
            idx_np = indices.cpu().numpy()
            selected.dataset = self.dataset.iloc[idx_np].copy()

        selected.source = self
        selected.last_op = op
        return selected

    def sanitize_F(self):
        """
        Remove invalid values from structure factors.

        Adds a mask to filter out NaN, Inf, and non-positive values
        from F and F_sigma.
        """
        mask = torch.zeros(len(self.F), dtype=torch.bool, device=self.device)
        if self.F is not None:
            if self.verbose > 0:
                print("found nan F values: ", torch.isnan(self.F).sum().item())
            mask |= torch.isnan(self.F)
        if self.F_sigma is not None:
            if self.verbose > 0:
                print(
                    "found nan F_sigma values: ", torch.isnan(self.F_sigma).sum().item()
                )
            mask |= torch.isnan(self.F_sigma)
        neg_mask = self.F <= 0
        if torch.any(neg_mask):
            warnings.warn(
                f"Found {neg_mask.sum().item()} non-positive F values, masking them out. This really should not happen!")
            mask |= neg_mask
        self.masks["sanity_F"] = ~mask
        # Zero out invalid values so they can't leak NaN through autograd
        # (masked indexing produces 0 gradients, but 0 * NaN = NaN in IEEE 754)
        if mask.any():
            self.F[mask] = 0.0
            if self.F_sigma is not None:
                self.F_sigma[mask] = 0.0
        return self

    def check_all_data_types(self):
        for key in self.__dict__:
            if self.__dict__[key] is not None and isinstance(
                self.__dict__[key], torch.Tensor
            ):
                print(
                    f"{key}: {self.__dict__[key].dtype}, shape: {self.__dict__[key].shape}"
                )
            elif self.__dict__[key] is not None:
                print(f"{key}: {type(self.__dict__[key])}, value: {self.__dict__[key]}")
            else:
                print(f"{key}: None")

    def validate_hkl(self, hkl_ref: torch.Tensor) -> "ReflectionData":
        """
        Expand dataset to match a reference HKL set.

        Reorders and expands the current dataset to match the reference HKL set.
        Reflections present in the reference but missing from this dataset are
        filled with placeholder values and masked out. This ensures all datasets
        aligned to the same reference have identical shapes and can be processed
        together without data loss from intersection operations.

        Parameters
        ----------
        hkl_ref : torch.Tensor
            Reference Miller indices tensor of shape (N, 3), dtype int32.
            This defines the canonical HKL ordering for all aligned datasets.

        Returns
        -------
        ReflectionData
            Self, modified in-place with expanded arrays matching hkl_ref.

        Notes
        -----
        After calling this method:
        - self.hkl will equal hkl_ref exactly
        - All data arrays (F, F_sigma, rfree_flags, etc.) are reordered/expanded
        - Missing reflections are filled with 0 (or appropriate defaults)
        - A mask 'hkl_present' is added marking which reflections have real data
        - forward() will return MaskedTensors that skip missing reflections

        This approach avoids the problem where intersecting many datasets with
        different outliers/missing reflections causes exponential data loss.

        Examples
        --------
        Align multiple datasets to a common HKL set::

            reference_hkl = data1.hkl.clone()
            data1.validate_hkl(reference_hkl)
            data2.validate_hkl(reference_hkl)
            # Now data1 and data2 have identical shapes
            assert data1.hkl.shape == data2.hkl.shape
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded in ReflectionData")

        if not isinstance(hkl_ref, torch.Tensor):
            raise TypeError(f"hkl_ref must be a torch.Tensor, got {type(hkl_ref)}")

        if hkl_ref.shape[-1] != 3:
            raise ValueError(f"hkl_ref must have shape (N, 3), got {hkl_ref.shape}")

        # Ensure hkl_ref is 2D and int32
        if hkl_ref.dim() == 1:
            hkl_ref = hkl_ref.unsqueeze(0)
        hkl_ref = hkl_ref.to(dtype=dtypes.int, device=self.device)

        n_ref = len(hkl_ref)
        n_data = len(self.hkl)

        # Build lookup from data HKL to index
        # Use a dictionary with tuple keys for fast lookup
        hkl_data_np = self.hkl.cpu().numpy()
        data_hkl_to_idx = {tuple(hkl): idx for idx, hkl in enumerate(hkl_data_np)}

        # For each reference HKL, find the corresponding data index (or -1 if missing)
        hkl_ref_np = hkl_ref.cpu().numpy()
        ref_to_data_idx = np.array(
            [data_hkl_to_idx.get(tuple(hkl), -1) for hkl in hkl_ref_np], dtype=np.int64
        )

        # Create presence mask: True where data exists
        presence_mask = torch.from_numpy(ref_to_data_idx >= 0).to(device=self.device)
        valid_indices = torch.from_numpy(ref_to_data_idx).to(device=self.device)

        # Helper to expand a tensor to reference size
        def expand_tensor(tensor, fill_value=0.0):
            if tensor is None:
                return None
            expanded = torch.full(
                (n_ref,) + tensor.shape[1:],
                fill_value,
                dtype=tensor.dtype,
                device=self.device,
            )
            # Copy existing data to correct positions
            mask = valid_indices >= 0
            expanded[mask] = tensor[valid_indices[mask]]
            return expanded

        # Expand all data arrays
        old_F = self.F
        old_F_sigma = self.F_sigma
        old_I = self.I
        old_I_sigma = getattr(self, "I_sigma", None)
        old_rfree = self.rfree_flags
        old_phase = getattr(self, "phase", None)
        old_fom = getattr(self, "fom", None)

        # Replace HKL with reference
        self.hkl = hkl_ref

        # Expand data tensors
        self.F = expand_tensor(old_F, fill_value=0.0)
        self.F_sigma = expand_tensor(old_F_sigma, fill_value=1.0)

        if old_I is not None:
            self.I = expand_tensor(old_I, fill_value=0.0)
        if old_I_sigma is not None:
            self.I_sigma = expand_tensor(old_I_sigma, fill_value=1.0)

        # For rfree, default missing to work set (1)
        if old_rfree is not None:
            rfree_expanded = torch.ones(
                n_ref, dtype=old_rfree.dtype, device=self.device
            )
            mask = valid_indices >= 0
            rfree_expanded[mask] = old_rfree[valid_indices[mask]]
            self.rfree_flags = rfree_expanded

        # Recalculate resolution for new HKL set
        self._calculate_resolution()

        # Expand phase and fom if present
        if old_phase is not None:
            self.phase = expand_tensor(old_phase, fill_value=0.0)
        if old_fom is not None:
            self.fom = expand_tensor(old_fom, fill_value=0.0)

        # Transfer existing masks to new indexing
        old_masks = dict(self.masks.items())
        # Clear existing masks
        self.masks.clear()
        self.masks._updated = True

        for name, old_mask in old_masks.items():
            if old_mask is not None and len(old_mask) == n_data:
                # Expand mask: missing reflections are masked out (False)
                new_mask = torch.zeros(n_ref, dtype=torch.bool, device=self.device)
                mask = valid_indices >= 0
                new_mask[mask] = old_mask[valid_indices[mask]]
                self.masks[name] = new_mask

        # Add presence mask - this is the key mask that marks real vs placeholder data
        self.masks["hkl_present"] = presence_mask

        n_present = presence_mask.sum().item()
        n_missing = n_ref - n_present

        if self.verbose > 0:
            print("HKL validation (expand mode):")
            print(f"  Original dataset: {n_data} reflections")
            print(f"  Reference set: {n_ref} reflections")
            print(f"  Present in data: {n_present} ({100*n_present/n_ref:.1f}%)")
            print(f"  Missing (masked): {n_missing} ({100*n_missing/n_ref:.1f}%)")

        return self

    def find_outliers(
        self, model: "ModelFT", scaler, z_threshold: float = 4.0
    ) -> torch.Tensor:
        """
        Identify outlier reflections based on log-ratio distribution.

        Uses the fact that log(F_obs) - log(F_calc) should be normally distributed.
        Outliers are reflections where |log_ratio - mean| > z_threshold * std_dev.

        Parameters
        ----------
        model : ModelFT
            ModelFT object to compute structure factors.
        scaler : Scaler
            Scaler object to scale calculated structure factors.
        z_threshold : float, optional
            Z-score threshold to classify outliers. Default is 4.0.

        Returns
        -------
        torch.Tensor
            Boolean mask where True indicates outliers.
        """
        hkl, F_obs, _, _ = self(mask=False)
        log_ratio = self.get_log_ratio(model, scaler)
        eps = 1e-10

        # Remove any infinite or NaN values for statistics
        valid_mask = torch.isfinite(log_ratio)
        if valid_mask.sum() == 0:
            if self.verbose > 0:
                print("Warning: No valid log-ratios found for outlier detection")
            return torch.zeros_like(F_obs, dtype=torch.bool, device=self.device)

        to_use = valid_mask

        log_ratio_valid = log_ratio[to_use]

        # Compute mean and standard deviation of log-ratio distribution
        mean_log_ratio = torch.mean(log_ratio_valid)
        std_log_ratio = torch.std(log_ratio_valid, unbiased=True)

        # Identify outliers using Z-score criterion
        z_scores = torch.abs(log_ratio - mean_log_ratio) / (std_log_ratio + eps)
        outlier_mask = z_scores > z_threshold

        # Set invalid ratios as outliers too
        outlier_mask = outlier_mask | ~valid_mask

        if self.verbose > 0:
            n_outliers = outlier_mask.sum().item()
            n_total = len(F_obs)
            print(
                f"Outlier detection: {n_outliers}/{n_total} ({100*n_outliers/n_total:.2f}%) outliers found"
            )
            print(
                f"  Log-ratio statistics: mean={mean_log_ratio:.3f}, std={std_log_ratio:.3f}"
            )
            print(f"  Z-score threshold: {z_threshold:.1f}")

        # Ensure outlier_mask is on correct device and register
        outlier_mask = outlier_mask.to(self.device)
        self.masks["outliers"] = ~outlier_mask
        if self.verbose > 0:
            print(
                f"Outlier detection: {outlier_mask.sum().item()} reflections flagged as outliers out of {len(outlier_mask)}."
            )

    def get_log_ratio(self, model: "ModelFT", scaler) -> torch.Tensor:
        """
        Compute log-ratio between observed and calculated structure factors.

        Parameters
        ----------
        model : ModelFT
            ModelFT object to compute structure factors.
        scaler : Scaler
            Scaler object to scale calculated structure factors.

        Returns
        -------
        torch.Tensor
            Log-ratio values: log(F_obs) - log(F_calc).
        """
        # Get observed and calculated structure factors
        eps = 1e-6
        hkl, F_obs, _, _ = self(mask=False)
        F_calc_complex = model.forward(hkl)  # Complex structure factors
        F_calc_scaled = torch.abs(
            scaler(F_calc_complex, use_mask=False)
        )  # Scaled amplitudes
        # Avoid log of zero by adding small epsilon
        F_obs_safe = torch.clamp(F_obs, min=eps)
        F_calc_safe = torch.clamp(F_calc_scaled, min=eps)
        # Compute log-ratio distribution: log(F_obs) - log(F_calc)
        log_ratio = torch.log(F_obs_safe) - torch.log(F_calc_safe)
        return log_ratio

    def get_outlier_statistics(self) -> Dict:
        """
        Get statistics about flagged outliers.

        Returns
        -------
        dict
            Dictionary containing:
            - n_outliers : int
            - n_total : int
            - fraction_outliers : float
            - detection_params : dict or None
            - outlier_resolution_stats : dict (if resolution available)
        """
        if self.outlier_flags is None:
            return {"n_outliers": 0, "n_total": 0, "fraction_outliers": 0.0}

        n_outliers = self.outlier_flags.sum().item()
        n_total = len(self.outlier_flags)

        stats = {
            "n_outliers": n_outliers,
            "n_total": n_total,
            "fraction_outliers": n_outliers / n_total if n_total > 0 else 0.0,
            "detection_params": self.outlier_detection_params,
        }

        if self.resolution is not None:
            # Add resolution-dependent statistics
            outlier_resolutions = (
                self.resolution[self.outlier_flags]
                if n_outliers > 0
                else torch.tensor([])
            )
            if len(outlier_resolutions) > 0:
                stats["outlier_resolution_stats"] = {
                    "min": outlier_resolutions.min().item(),
                    "max": outlier_resolutions.max().item(),
                    "mean": outlier_resolutions.mean().item(),
                    "median": outlier_resolutions.median().item(),
                }

        return stats

    def unpack_one(self):
        """
        Unpack one level of source.

        Does not recurse fully and does not flag.

        Returns
        -------
        ReflectionData
            Parent source or self if no source.
        """
        if self.source is not None:
            return self.source
        return self

    def flag_suspicious_sigma(self, z_threshold: float = 5.0) -> None:
        """
        Flag sigma values that deviate significantly from expected distribution.

        Sigma values from a detector should follow a log-normal distribution.
        Values with z-scores beyond threshold are flagged as suspicious.

        Parameters
        ----------
        z_threshold : float, optional
            Z-score threshold for flagging outliers. Default is 5.0.
        """
        sigmas = self.F_sigma
        log_sigmas = torch.log(sigmas)
        flagged_initial = torch.isnan(log_sigmas) | torch.isinf(log_sigmas)
        mean_log_sigma = torch.mean(log_sigmas[~flagged_initial])
        std_log_sigma = torch.std(log_sigmas[~flagged_initial]) + 1e-5 * mean_log_sigma
        z_scores = (log_sigmas - mean_log_sigma) / std_log_sigma
        flagged = torch.abs(z_scores) > z_threshold
        flagged = flagged | flagged_initial
        if self.verbose > 0:
            n_flagged = flagged.sum().item()
            n_total = len(sigmas)
            print(
                f"Suspicious sigma detection: {n_flagged}/{n_total} ({100*n_flagged/n_total:.2f}%) reflections flagged"
            )
        self.masks["flagged_sigma"] = ~flagged

    def dump(self):
        """
        Dump all reflection data to console for debugging.

        Prints type, shape, and device information for all attributes.
        """
        print("ReflectionData dump:")
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, torch.Tensor):
                print(
                    f"  {key}: dtype={value.dtype}, shape={value.shape}, device={value.device}"
                )
            else:
                print(f"  {key}: type={type(value)}, value={value}")

    def _build_anomalous_dataframe(self, fcalc: torch.Tensor) -> pd.DataFrame:
        """Build a phenix-style anomalous MTZ DataFrame on the canonical ASU.

        Bijvoet mates share a canonical ASU index in :attr:`hkl`; here they are
        (a) merged by mean amplitude for the display maps / ``F-obs`` / ``F-model``
        and (b) unstacked into ``(+)/(-)`` columns. No negative-ASU Miller
        indices are emitted, so the display maps render normally in Coot while
        the anomalous columns are available for anomalous difference maps.

        Parameters
        ----------
        fcalc : torch.Tensor
            Complex per-row structure factors evaluated at the *signed* HKL
            (``hkl_for_sf``), row-aligned with :attr:`hkl`.

        Returns
        -------
        pandas.DataFrame
            One row per unique canonical ASU reflection.
        """
        if fcalc is None or not torch.is_complex(fcalc):
            raise ValueError("anomalous=True requires a complex fcalc tensor")
        if self.friedel_flags is None:
            raise ValueError(
                "anomalous output requires canonicalized data with friedel_flags; "
                "load via load_mtz so Friedel bookkeeping is populated."
            )

        hkl = self.hkl.detach().cpu()
        N = hkl.shape[0]
        flag = self.friedel_flags.detach().cpu()

        # Group rows by unique canonical ASU index; inverse maps row -> group.
        uniq, inverse = torch.unique(hkl, dim=0, return_inverse=True)
        M = uniq.shape[0]

        # The (+) member is the unconjugated row, (-) is the Friedel-flagged row.
        arange = torch.arange(N)
        plus_idx = torch.full((M,), -1, dtype=torch.long)
        minus_idx = torch.full((M,), -1, dtype=torch.long)
        plus_idx[inverse[~flag]] = arange[~flag]
        minus_idx[inverse[flag]] = arange[flag]
        has_plus = (plus_idx >= 0).numpy()
        has_minus = (minus_idx >= 0).numpy()
        pi = plus_idx.clamp(min=0).numpy()
        mi = minus_idx.clamp(min=0).numpy()

        # Centric flags per ASU group (centrics obey Friedel's law: F(+)=F(-)).
        cen_full = self.centric
        centric = np.zeros(M, dtype=bool)
        if cen_full is not None:
            cen_full = cen_full.detach().cpu().numpy()
            centric[has_plus] = cen_full[pi][has_plus]
            centric[has_minus] = cen_full[mi][has_minus]

        fc = fcalc.detach().cpu().numpy()
        Fc_amp = np.abs(fc)
        Fc_ph = np.angle(fc, deg=True)
        F = self.F.detach().cpu().numpy()
        Fsig = self.F_sigma.detach().cpu().numpy() if self.F_sigma is not None else None
        rfree = (
            self.rfree_flags.detach().cpu().numpy().astype(int)
            if self.rfree_flags is not None
            else None
        )

        def plus_of(src):
            out = np.full(M, np.nan, dtype=np.float64)
            out[has_plus] = src[pi][has_plus]
            return out

        def minus_of(src):
            out = np.full(M, np.nan, dtype=np.float64)
            out[has_minus] = src[mi][has_minus]
            return out

        # Raw (+/-) values before centric mirroring (used for the merged mean).
        Fobs_p, Fobs_m = plus_of(F), minus_of(F)
        Fmod_p, Fmod_m = plus_of(Fc_amp), minus_of(Fc_amp)
        Phi_p, Phi_m = plus_of(Fc_ph), minus_of(Fc_ph)

        def mirror_centric(plus, minus):
            # For centrics, the absent mate equals the present one.
            p = np.where(
                centric & ~np.isfinite(plus) & np.isfinite(minus), minus, plus
            )
            m = np.where(
                centric & ~np.isfinite(minus) & np.isfinite(plus), plus, minus
            )
            return p, m

        Fobs_p_out, Fobs_m_out = mirror_centric(Fobs_p, Fobs_m)
        Fmod_p_out, Fmod_m_out = mirror_centric(Fmod_p, Fmod_m)
        Phi_p_out, Phi_m_out = mirror_centric(Phi_p, Phi_m)

        # Merged display quantities: mean observed amplitude over present mates,
        # and the ASU representative structure factor (the + member, else the
        # conjugate of the - member).
        with np.errstate(invalid="ignore"):
            Fobs_disp = np.nanmean(np.vstack([Fobs_p, Fobs_m]), axis=0)
        fc_disp = np.full(M, np.nan, dtype=complex)
        fc_disp[has_plus] = fc[pi][has_plus]
        only_minus = has_minus & ~has_plus
        fc_disp[only_minus] = np.conj(fc[mi])[only_minus]

        Fc_disp_amp = np.abs(fc_disp)
        ph_disp = np.angle(fc_disp, deg=True)

        # Map coefficients (same convention as the legacy per-row path).
        two_mfo = np.abs(2.0 * Fobs_disp - Fc_disp_amp)
        mfo_complex = Fobs_disp * np.exp(1j * np.deg2rad(ph_disp)) - fc_disp
        delf = np.abs(mfo_complex)
        delph = np.angle(mfo_complex, deg=True)

        # Anomalous difference map coefficients. The anomalous-difference Fourier
        # uses |ΔF_ano| with phase (phi_model - 90deg); peaks then fall on the
        # anomalous scatterers (verified against the Zn site of thermolysin:
        # phi-90 gives +2.6 sigma, phi+90 gives a -2.6 sigma hole). ANOM is stored
        # signed, so the (-) member maps to phi-270 (= the +180deg / negative-
        # amplitude equivalent of phi-90).
        anom = Fobs_p_out - Fobs_m_out
        panom = np.where(anom < 0.0, ph_disp - 270.0, ph_disp - 90.0)

        uniq_np = uniq.numpy()
        data = {
            "H": uniq_np[:, 0],
            "K": uniq_np[:, 1],
            "L": uniq_np[:, 2],
            "F-obs": Fobs_disp,
            "F-obs(+)": Fobs_p_out,
            "F-obs(-)": Fobs_m_out,
            "F-model": Fc_disp_amp,
            "PH-model": ph_disp,
            "F-model(+)": Fmod_p_out,
            "PHIF-model(+)": Phi_p_out,
            "F-model(-)": Fmod_m_out,
            "PHIF-model(-)": Phi_m_out,
            "FWT": two_mfo,
            "PHWT": ph_disp,
            "DELFWT": delf,
            "PHDELWT": delph,
            "ANOM": anom,
            "PANOM": panom,
        }
        if Fsig is not None:
            data["SIGF-obs(+)"], data["SIGF-obs(-)"] = mirror_centric(
                plus_of(Fsig), minus_of(Fsig)
            )
        if rfree is not None:
            rf = np.zeros(M, dtype=int)
            rf[has_minus] = rfree[mi][has_minus]
            rf[has_plus] = rfree[pi][has_plus]  # both mates share a flag
            data["R-free-flags"] = rf

        # The display-map / merged columns must be FFT-safe (no NaN); the
        # anomalous (+/-) columns may legitimately carry NaN where a mate is
        # absent (incomplete anomalous data), matching phenix output.
        for key in ("F-obs", "F-model", "PH-model", "FWT", "PHWT", "DELFWT", "PHDELWT"):
            data[key] = np.nan_to_num(data[key], nan=0.0)

        return pd.DataFrame(data)

    def write_mtz(
        self,
        fname: str,
        fcalc: Optional[torch.Tensor] = None,
        model_ft: Optional["ModelFT"] = None,
        anomalous: bool = False,
    ) -> None:
        """
        Write reflection data to MTZ file with optional map coefficients.

        Parameters
        ----------
        fname : str
            Output MTZ filename.
        fcalc : torch.Tensor, optional
            Complex calculated structure factors of shape (N,).
            If provided, computes phases and map coefficients.
        model_ft : ModelFT, optional
            ModelFT object to compute fcalc if not provided.
        anomalous : bool, optional
            If True, write a phenix-style anomalous MTZ on the canonical ASU:
            display maps (FWT/PHWT, DELFWT/PHDELWT) and merged F-obs/F-model
            with Friedel mates merged by mean amplitude, plus unstacked
            F-obs(+/-), SIGF-obs(+/-), F-model(+/-), PHIF-model(+/-) and
            ANOM/PANOM columns. No negative-ASU indices are emitted. Default
            False (legacy per-row layout).

        Notes
        -----
        The MTZ file will contain canonical column names:
            - FP, SIGFP: Observed amplitudes and uncertainties
            - I, SIGI: Observed intensities and uncertainties (if available)
            - FreeR_flag: R-free test set flags
            - FWT, PHWT: 2mFo-DFc map coefficients (if fcalc provided)
            - DELFWT, PHDELWT: mFo-DFc map coefficients (if fcalc provided)

        Map coefficients are computed as:
            - 2mFo-DFc: 2*Fo - Fc (filled to resolution limit)
            - mFo-DFc: Fo - Fc

        Examples
        --------
        Write MTZ with map coefficients::

            data = ReflectionData().load_mtz('observed.mtz')
            model = Model().load_pdb('model.pdb')
            model_ft = ModelFT(model, data.cell, data.spacegroup)
            fcalc = model_ft.forward(data.hkl)
            data.write_mtz('output.mtz', fcalc=fcalc)
        """
        from torchref.io.mtz import write

        if anomalous:
            # Signed HKL preserves the anomalous/Bijvoet difference (see
            # hkl_for_sf) so the two mates get distinct F-model.
            if fcalc is None and model_ft is not None:
                fcalc = model_ft.forward(self.hkl_for_sf())
            df = self._build_anomalous_dataframe(fcalc)
            write(df, self.cell.data, self.spacegroup, fname)
            if self.verbose > 0:
                print(f"✓ Wrote phenix-style anomalous MTZ: {fname}")
                print(f"  ASU reflections: {len(df)}")
                print(f"  Columns: {', '.join(df.columns)}")
            return

        # Convert data to numpy for DataFrame creation
        hkl_np = self.hkl.detach().cpu().numpy()

        # Create DataFrame with HKL indices
        data_dict = {
            "H": hkl_np[:, 0],
            "K": hkl_np[:, 1],
            "L": hkl_np[:, 2],
        }

        # Add observed amplitudes (canonical names: FP, SIGFP)
        if self.F is not None:
            data_dict["F-obs"] = self.F.detach().cpu().numpy()
            if self.F_sigma is not None:
                data_dict["SIGF-obs"] = self.F_sigma.detach().cpu().numpy()

        # Add observed intensities (canonical names: I, SIGI)
        if self.I is not None:
            data_dict["I-obs"] = self.I.detach().cpu().numpy()
            if self.I_sigma is not None:
                data_dict["SIGI-obs"] = self.I_sigma.detach().cpu().numpy()

        # Add R-free flags (canonical name: FreeR_flag)
        if self.rfree_flags is not None:
            data_dict["R-free-flags"] = (
                self.rfree_flags.detach().cpu().numpy().astype(int)
            )

        # Compute fcalc if model_ft is provided but fcalc is not
        if fcalc is None and model_ft is not None:
            fcalc = model_ft.forward(self.hkl)
        mask = self.masks().detach().cpu().numpy()
        # Add map coefficients if fcalc is provided
        if fcalc is not None:
            # Ensure fcalc is complex
            if not torch.is_complex(fcalc):
                raise ValueError("fcalc must be a complex tensor")

            # Convert to numpy
            fcalc_np = fcalc.detach().cpu().numpy()
            F_obs = self.F.detach().cpu().numpy()

            # Compute phases in degrees
            phases = np.angle(fcalc_np, deg=True)
            F_calc_amp = np.abs(fcalc_np)

            # Compute map coefficients
            # 2mFo-DFc: Use observed amplitudes with calculated phases
            # When 2*Fobs - Fcalc < 0, flip phase by 180° and use absolute amplitude
            two_mfo_dfc_raw = 2.0 * F_obs - F_calc_amp
            two_mfo_dfc_amp = np.abs(two_mfo_dfc_raw)
            two_mfo_dfc_phase = phases.copy()

            # mFo-DFc: Difference map
            mfo_dfc_complex = F_obs * np.exp(1j * np.deg2rad(phases)) - fcalc_np
            mfo_dfc_complex[~mask] = 0.0  # Zero out reflections outside mask
            mfo_dfc_amp = np.abs(mfo_dfc_complex)
            mfo_dfc_phase = np.angle(mfo_dfc_complex, deg=True)

            # Add 2mFo-DFc map coefficients (standard Coot names: FWT, PHWT)
            data_dict["FWT"] = two_mfo_dfc_amp
            data_dict["PHWT"] = two_mfo_dfc_phase

            # Add mFo-DFc map coefficients (standard Coot names: DELFWT, PHDELWT)
            data_dict["DELFWT"] = mfo_dfc_amp
            data_dict["PHDELWT"] = mfo_dfc_phase

            data_dict["F-model"] = F_calc_amp
            data_dict["PH-model"] = phases

            if self.verbose > 0:
                print("Added map coefficients:")
                print("  2mFo-DFc: FWT, PHWT")
                print("  mFo-DFc: DELFWT, PHDELWT")
                print(
                    f"  Resolution range: {self.resolution.min().item():.2f} - {self.resolution.max().item():.2f} Å"
                )

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        # Write MTZ file
        write(df, self.cell.data, self.spacegroup, fname)

        if self.verbose > 0:
            print(f"✓ Wrote MTZ file: {fname}")
            print(f"  Reflections: {len(df)}")
            print(f"  Columns: {', '.join(df.columns)}")

    @property
    def centric(self):
        """
        Get boolean mask for centric reflections (full size, unfiltered).

        Calculates it if not already present. Returns unfiltered centric flags
        matching the full HKL array size, consistent with how forward() returns
        full-size arrays when using MaskedTensors.

        Returns
        -------
        torch.Tensor or None
            Boolean tensor of shape (N,) where N is total reflections.
            True indicates centric reflection, False indicates acentric.
        """
        if self.hkl is None:
            return None

        # Check if we already have it cached (could be stored in a buffer if we want persistence)
        if not hasattr(self, "_centric_flags") or self._centric_flags is None:
            from torchref.base.french_wilson import is_centric_from_hkl

            # Ensure we have spacegroup
            sg = self.spacegroup if self.spacegroup else "P1"

            self._centric_flags = is_centric_from_hkl(self.hkl, sg)

        # Return full-size centric flags (no filtering)
        return self._centric_flags

    def calc_patterson(
        self,
        grid_size: Optional[Tuple[int, int, int]] = None,
        grid_sampling: Optional[float] = 1,
    ) -> torch.Tensor:
        """
        Calculate Patterson map of the dataset.

        The Patterson function P(u,v,w) = Σ|F(hkl)|² exp(-2πi(hu+kv+lw))
        is computed via inverse FFT of F². Data is expanded to P1 symmetry
        using only observed reflections (no filling of missing data).

        Parameters
        ----------
        grid_size : tuple of int, optional
            Grid dimensions (Nx, Ny, Nz). If None, automatically determined
            from unit cell and resolution.
        grid_sampling : float, optional
            Sampling interval for the grid. Default is 1.
            This sets the grid so that we sample twice as much as normal for a given resolution

        Returns
        -------
        torch.Tensor
            Real-valued Patterson map of shape (Nx, Ny, Nz).
            Origin is at grid position [0, 0, 0].
        """
        from torchref.base.fourier import find_grid_size
        from torchref.base.reciprocal import place_on_grid

        # Expand to P1 symmetry (don't fill missing reflections - use only observed data)
        data = self.expand_to_p1()

        max_res = data.resolution.min() * grid_sampling

        if grid_size is None:
            grid_size = find_grid_size(data.cell, max_res)

        # Use data_indexed to get only valid (observed) reflections
        hkl, F, _, _ = data.data_indexed()

        F_2 = F**2

        # Place F² on reciprocal grid (don't enforce Hermitian since we have P1 expansion)
        grid = place_on_grid(hkl, F_2, grid_size, enforce_hermitian=False)

        patterson = torch.fft.ifftn(grid, dim=(0, 1, 2), norm="forward").real

        return patterson

    def possible_hkl(self) -> torch.Tensor:
        """
        Generate all possible HKL indices within the resolution limit.

        Returns
        -------
        torch.Tensor
            Tensor of shape (M, 3) containing all possible Miller indices
            within the resolution limit defined by self.resolution.
        """
        from torchref.base.reciprocal import generate_possible_hkl

        if self.cell is None or self.resolution is None:
            raise ValueError(
                "Cell and resolution must be defined to generate possible HKL"
            )

        max_res = self.resolution.min().item()
        possible_hkl = generate_possible_hkl(self.cell.data, max_res, device=self.device)

        return possible_hkl

    def remap(
        self,
        new_hkl: torch.Tensor,
        index_mapping: torch.Tensor,
        phase_shifts: Optional[torch.Tensor] = None,
        spacegroup=None,
        op_name: str = "remap",
    ) -> "ReflectionData":
        """
        Create new ReflectionData with remapped HKL set and data.

        This is the core method for index-based transformations of reflection
        data. It handles remapping all tensor fields based on an index mapping,
        with support for missing reflections (indicated by -1 in index_mapping).

        Parameters
        ----------
        new_hkl : torch.Tensor, shape (M, 3)
            New Miller indices.
        index_mapping : torch.Tensor, shape (M,), dtype int64
            Maps new indices to original: ``new[i] = old[index_mapping[i]]``
            Values of -1 indicate missing reflections (filled with defaults).
        phase_shifts : torch.Tensor, optional, shape (M,)
            Phase offsets to apply (e.g., from symmetry translations).
        spacegroup : str, int, gemmi.SpaceGroup, or None
            New spacegroup. If None, keeps original.
        op_name : str
            Operation name for provenance tracking.

        Returns
        -------
        ReflectionData
            New object with remapped data. Missing reflections get:
            - 0.0 for F, I, phase, fom
            - 1.0 for F_sigma, I_sigma (conservative uncertainty)
            - True for masks['missing']
        """
        from torchref.symmetry.spacegroup import SpaceGroup

        # Helper function for remapping tensors with missing handling
        def _remap_tensor(tensor, fill_value):
            if tensor is None:
                return None
            valid_mask = index_mapping >= 0
            result = torch.full(
                (len(new_hkl),) + tensor.shape[1:],
                fill_value,
                dtype=tensor.dtype,
                device=self.device,
            )
            result[valid_mask] = tensor[index_mapping[valid_mask]]
            return result

        # Create new ReflectionData
        remapped = ReflectionData(verbose=self.verbose, device=self.device)

        # Set new HKL
        remapped.hkl = new_hkl.to(device=self.device)

        # Remap amplitude and intensity fields
        remapped.F = _remap_tensor(self.F, fill_value=0.0)
        remapped.F_sigma = _remap_tensor(self.F_sigma, fill_value=1.0)
        remapped.I = _remap_tensor(self.I, fill_value=0.0)
        remapped.I_sigma = _remap_tensor(self.I_sigma, fill_value=1.0)
        remapped.fom = _remap_tensor(self.fom, fill_value=0.0)

        # Carry forward prior mask if available
        prior_mask = self.masks()
        if prior_mask is not None:
            remapped.masks["prior_flagged"] = _remap_tensor(
                prior_mask.to(dtype=dtypes.int), fill_value=0
            ).to(torch.bool)

        # Handle rfree_flags (True = include in Rfree for missing)
        if self.rfree_flags is not None:
            remapped.rfree_flags = _remap_tensor(
                self.rfree_flags.to(dtype=dtypes.int), fill_value=1
            ).to(torch.bool)

        # Handle phases with optional shifts
        if self.phase is not None:
            remapped.phase = _remap_tensor(self.phase, fill_value=0.0)
            if phase_shifts is not None:
                remapped.phase = remapped.phase + phase_shifts.to(device=self.device)
        elif phase_shifts is not None:
            # Store phase shifts even if no original phases
            remapped._expansion_phase_shifts = phase_shifts.to(device=self.device)

        # Clone cell
        remapped.cell = self.cell.clone() if self.cell is not None else None

        # Set spacegroup
        if spacegroup is not None:
            remapped.spacegroup = SpaceGroup(spacegroup)
        else:
            remapped.spacegroup = self.spacegroup

        # Recalculate resolution
        if remapped.cell is not None and remapped.hkl is not None:
            remapped._calculate_resolution()

        # Invalidate dependent fields
        remapped.bin_indices = None

        # Copy metadata sources
        remapped.amplitude_source = self.amplitude_source
        remapped.intensity_source = self.intensity_source
        remapped.phase_source = self.phase_source
        remapped.rfree_source = self.rfree_source

        # Track provenance
        remapped.source = self
        remapped.last_op = op_name

        # Add missing mask
        missing_mask = index_mapping < 0
        if missing_mask.any():
            remapped.masks["missing"] = ~missing_mask.to(device=self.device)

        return remapped

    def fill(self, d_min: Optional[float] = None) -> "ReflectionData":
        """
        Fill missing reflections within resolution limit.

        Generates all possible reflections for the current spacegroup within
        the resolution limit, identifies which are missing, and creates a
        complete dataset. Missing reflections are filled with default values.

        Parameters
        ----------
        d_min : float, optional
            High resolution limit in Angstroms. If None, uses the minimum
            resolution from the current dataset.

        Returns
        -------
        ReflectionData
            New ReflectionData with complete set of reflections.
            Missing reflections have:
            - F, I, phase, fom = 0.0
            - F_sigma, I_sigma = 1.0
            - masks['missing'] = True

        Examples
        --------
        Fill to completeness::

            data = ReflectionData().load_mtz('data.mtz')
            data_filled = data.fill(d_min=2.0)
            print(f"Original: {len(data)}, Filled: {len(data_filled)}")
        """
        from torchref.symmetry.reciprocal_symmetry import complete_hkl

        if self.hkl is None:
            raise ValueError("ReflectionData has no Miller indices loaded")
        if self.cell is None:
            raise ValueError("ReflectionData has no unit cell defined")

        # Use current resolution limit if not specified
        if d_min is None:
            if self.resolution is None:
                raise ValueError("Resolution not available - specify d_min")
            d_min = self.resolution.min().item()

        # Get complete HKL set with index mapping
        filled_hkl, indices, missing = complete_hkl(
            self.hkl, self.cell.data, self.spacegroup or "P1", d_min, device=self.device
        )

        # Use remap to create the new dataset
        remapped = self.remap(
            new_hkl=filled_hkl,
            index_mapping=indices,
            spacegroup=self.spacegroup,  # Keep same spacegroup
            op_name=f"fill(d_min={d_min:.2f})",
        )

        return remapped

    def expand_to_p1(
        self, include_friedel: bool = True, remove_absences: bool = True
    ) -> "ReflectionData":
        """
        Expand reflection data from asymmetric unit to P1.

        Applies all symmetry operations from the current space group to generate
        all symmetry-equivalent reflections. Returns a NEW ReflectionData object
        with expanded reflections; does not modify self.

        Parameters
        ----------
        include_friedel : bool, default True
            Include Friedel mates (-h, -k, -l). For normal (non-anomalous)
            scattering, Friedel pairs have identical amplitudes.
        remove_absences : bool, default True
            Remove systematically absent reflections from output.

        Returns
        -------
        ReflectionData
            New ReflectionData object with:
            - All symmetry-equivalent reflections
            - spacegroup set to P1
            - Expanded tensor fields (F, F_sigma, I, I_sigma, phase, fom, rfree_flags)
            - Recalculated resolution values
            - Provenance tracking via source and last_op attributes

        Notes
        -----
        The expansion handles tensor fields as follows:

        - hkl: Symmetry operations applied, Friedel mates added, duplicates removed
        - F, F_sigma, I, I_sigma, fom, rfree_flags: Indexed from original
        - phase: Indexed + phase shift from translation component
        - resolution: Recalculated from expanded hkl + cell
        - bin_indices: Cleared (invalidated by expansion)

        Examples
        --------
        Expand to P1 for calculations::

            # Load data in original space group
            data = ReflectionData().load_mtz('data.mtz')
            print(f"Original: {len(data)} reflections, {data.spacegroup}")

            # Expand to P1
            data_p1 = data.expand_to_p1()
            print(f"Expanded: {len(data_p1)} reflections, {data_p1.spacegroup}")

            # Without Friedel mates (for anomalous data)
            data_p1_anom = data.expand_to_p1(include_friedel=False)
        """
        from torchref.symmetry.reciprocal_symmetry import expand_hkl

        if self.hkl is None:
            raise ValueError("ReflectionData has no Miller indices loaded")

        # Get expanded HKL set with index mapping and phase shifts
        hkl_p1, indices, phase_shifts = expand_hkl(
            self.hkl,
            self.spacegroup or "P1",
            include_friedel=include_friedel,
            remove_absences=remove_absences,
            device=self.device,
        )

        # Use remap to create the new dataset
        return self.remap(
            new_hkl=hkl_p1,
            index_mapping=indices,
            phase_shifts=phase_shifts,
            spacegroup="P1",
            op_name=f"expand_to_p1(include_friedel={include_friedel})",
        )

    def reduce_to_spacegroup(
        self, spacegroup, include_friedel: bool = True, aggregation: str = "mean"
    ) -> "ReflectionData":
        """
        Reduce P1 reflection data to asymmetric unit of a target spacegroup.

        This is the inverse of expand_to_p1(). Takes reflection data in P1 and
        merges symmetry-equivalent reflections into single ASU reflections using
        the specified aggregation function.

        Parameters
        ----------
        spacegroup : str, int, or gemmi.SpaceGroup
            Target space group specification.
        include_friedel : bool, default True
            If True, also merge Friedel mates when reducing.
        aggregation : str, default 'mean'
            Aggregation function for merging equivalent reflections:
            - 'mean': Average values (default, good for amplitudes)
            - 'sum': Sum values
            - 'first': Take first valid value (no averaging)

        Returns
        -------
        ReflectionData
            New ReflectionData with merged reflections in the target spacegroup.

        Notes
        -----
        Field handling during reduction:

        - F, I: Aggregated using specified function
        - F_sigma, I_sigma: Propagated as sqrt(sum(sigma^2) / n) for 'mean'
        - phase: Aggregated via complex averaging (handles phase wrapping)
        - fom: Weighted by amplitude during phase averaging
        - rfree_flags: OR operation (True if any equivalent is True)

        Examples
        --------
        Reduce symmetry-expanded data::

            # Expand to P1 for calculations, then reduce back
            data_p1 = data.expand_to_p1()
            # ... modify F_p1 ...
            data_merged = data_p1.reduce_to_spacegroup('P21')

            # Reduce with sum instead of mean
            data_summed = data_p1.reduce_to_spacegroup('P21', aggregation='sum')
        """
        from torchref.symmetry.reciprocal_symmetry import reduce_hkl
        from torchref.symmetry.spacegroup import SpaceGroup

        if self.hkl is None:
            raise ValueError("ReflectionData has no Miller indices loaded")

        # Get reduction mapping
        hkl_asu, reduction_indices, phase_shifts = reduce_hkl(
            self.hkl, spacegroup, include_friedel=include_friedel, device=self.device
        )

        n_asu = len(hkl_asu)
        n_equiv = reduction_indices.shape[1]
        valid_mask = reduction_indices >= 0  # (n_asu, n_equiv)
        count_valid = valid_mask.sum(dim=1).clamp(min=1).float()  # (n_asu,)

        # Helper function for aggregating 1D tensors
        def _aggregate_tensor(tensor, agg_func="mean", fill_value=0.0):
            if tensor is None:
                return None

            # Gather values: (n_asu, n_equiv)
            # Use clamp(min=0) to avoid indexing errors, then mask invalid
            gathered = tensor[reduction_indices.clamp(min=0)]
            gathered = torch.where(valid_mask, gathered, torch.zeros_like(gathered))

            if agg_func == "mean":
                return gathered.sum(dim=1) / count_valid
            elif agg_func == "sum":
                return gathered.sum(dim=1)
            elif agg_func == "first":
                # Take first valid value
                first_valid_idx = valid_mask.to(dtype=dtypes.int).argmax(dim=1)
                return gathered[
                    torch.arange(n_asu, device=self.device), first_valid_idx
                ]
            else:
                raise ValueError(f"Unknown aggregation: {agg_func}")

        def _aggregate_sigma(tensor, agg_func="mean"):
            """Propagate uncertainty correctly for averaging."""
            if tensor is None:
                return None

            # Gather values
            gathered = tensor[reduction_indices.clamp(min=0)]
            gathered = torch.where(valid_mask, gathered, torch.zeros_like(gathered))

            if agg_func == "mean":
                # For averaging: sigma_mean = sqrt(sum(sigma^2)) / n
                variance_sum = (gathered**2).sum(dim=1)
                return torch.sqrt(variance_sum) / count_valid
            elif agg_func == "sum":
                # For summing: sigma_sum = sqrt(sum(sigma^2))
                variance_sum = (gathered**2).sum(dim=1)
                return torch.sqrt(variance_sum)
            elif agg_func == "first":
                first_valid_idx = valid_mask.to(dtype=dtypes.int).argmax(dim=1)
                return gathered[
                    torch.arange(n_asu, device=self.device), first_valid_idx
                ]
            else:
                raise ValueError(f"Unknown aggregation: {agg_func}")

        # Create new ReflectionData
        reduced = ReflectionData(verbose=self.verbose, device=self.device)

        # Set HKL
        reduced.hkl = hkl_asu.to(device=self.device)

        # Aggregate amplitude and intensity fields
        reduced.F = _aggregate_tensor(self.F, aggregation)
        reduced.F_sigma = _aggregate_sigma(self.F_sigma, aggregation)
        reduced.I = _aggregate_tensor(self.I, aggregation)
        reduced.I_sigma = _aggregate_sigma(self.I_sigma, aggregation)

        # Handle phases via complex averaging
        if self.phase is not None:
            # Gather phases and apply phase shifts for proper averaging
            phases_gathered = self.phase[reduction_indices.clamp(min=0)]
            phases_gathered = phases_gathered + phase_shifts
            phases_gathered = torch.where(
                valid_mask, phases_gathered, torch.zeros_like(phases_gathered)
            )

            # Get weights (amplitudes or FOM)
            if self.fom is not None:
                weights = self.fom[reduction_indices.clamp(min=0)]
            elif self.F is not None:
                weights = self.F[reduction_indices.clamp(min=0)]
            else:
                weights = torch.ones_like(phases_gathered)
            weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

            # Complex averaging: mean of F*exp(i*phi) then extract angle
            complex_sf = weights * torch.exp(1j * phases_gathered)
            complex_mean = complex_sf.sum(dim=1) / count_valid
            reduced.phase = torch.angle(complex_mean).float()

            # FOM as magnitude of normalized mean complex vector
            if self.fom is not None:
                norm_weights = weights / weights.sum(dim=1, keepdim=True).clamp(
                    min=1e-10
                )
                unit_vectors = torch.exp(1j * phases_gathered)
                mean_vector = (norm_weights * unit_vectors).sum(dim=1)
                reduced.fom = torch.abs(mean_vector).float()
        else:
            reduced.phase = None
            reduced.fom = (
                _aggregate_tensor(self.fom, aggregation)
                if self.fom is not None
                else None
            )

        # Handle rfree_flags: OR operation (free if any equivalent is free)
        if self.rfree_flags is not None:
            rfree_gathered = self.rfree_flags[reduction_indices.clamp(min=0)].to(
                dtypes.int
            )
            rfree_gathered = torch.where(
                valid_mask,
                rfree_gathered,
                torch.ones_like(rfree_gathered),  # Default to work set
            )
            # 0 = free, non-zero = work. Take min to get free if any is free.
            reduced.rfree_flags = rfree_gathered.min(dim=1).values != 0

        # Clone cell
        reduced.cell = self.cell.clone() if self.cell is not None else None

        # Set spacegroup
        reduced.spacegroup = SpaceGroup(spacegroup)

        # Recalculate resolution
        if reduced.cell is not None and reduced.hkl is not None:
            reduced._calculate_resolution()

        # Invalidate dependent fields
        reduced.bin_indices = None

        # Copy metadata sources
        reduced.amplitude_source = self.amplitude_source
        reduced.intensity_source = self.intensity_source
        reduced.phase_source = self.phase_source
        reduced.rfree_source = self.rfree_source

        # Track provenance
        reduced.source = self
        reduced.last_op = (
            f"reduce_to_spacegroup({spacegroup}, aggregation={aggregation})"
        )

        return reduced

    def canonicalize(self, include_friedel: bool = True) -> "ReflectionData":
        """Return new ReflectionData with HKL in standard CCP4 ASU form.

        Remaps all Miller indices to the canonical CCP4 asymmetric unit
        representative using ``gemmi.ReciprocalAsu``, adjusts phases
        accordingly, and sorts reflections lexicographically by (h, k, l).

        Parameters
        ----------
        include_friedel : bool, default True
            Whether Friedel mates are considered equivalent.

        Returns
        -------
        ReflectionData
            New object with canonicalized, sorted Miller indices.
        """
        from torchref.symmetry.reciprocal_symmetry import canonicalize_hkl

        if self.hkl is None:
            raise ValueError("ReflectionData has no Miller indices loaded")

        canonical_hkl, phase_shifts, friedel_flags, sort_indices = canonicalize_hkl(
            self.hkl, self.spacegroup or "P1", include_friedel, device=self.device
        )

        # Reorder all fields using __select__
        result = self.__select__(sort_indices, op=f"canonicalize(include_friedel={include_friedel})")

        # Overwrite HKL with canonical form (already sorted)
        result.hkl = canonical_hkl

        # Fix phases: phi_new = where(friedel, -phi_old, phi_old) + phase_shift
        if result.phase is not None:
            result.phase = torch.where(friedel_flags, -result.phase, result.phase) + phase_shifts

        # Record anomalous bookkeeping for the canonical result (the stale values
        # carried over by __select__ are recomputed here). See hkl_for_sf.
        result.friedel_flags = friedel_flags
        result.hkl_anomalous = torch.where(
            friedel_flags.unsqueeze(-1), -canonical_hkl, canonical_hkl
        )

        # Recalculate resolution from canonical HKL + cell
        if result.cell is not None:
            result._calculate_resolution()

        # Invalidate bin_indices
        result.bin_indices = None

        return result

    # ========== E-VALUE AND ANISOTROPY CORRECTION METHODS ==========

    def get_scattering_vectors(self) -> torch.Tensor:
        """
        Get scattering vectors (s-vectors) from hkl and cell.

        The s-vector for a reflection hkl is defined as:
            s = B* @ hkl
        where B* is the reciprocal basis matrix.

        Returns
        -------
        s_vectors : torch.Tensor
            Reciprocal space vectors in Angstroms^-1, shape (N, 3).

        Raises
        ------
        ValueError
            If hkl or cell is not available.
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded")
        if self.cell is None:
            raise ValueError("No unit cell defined")

        return math_torch.get_scattering_vectors(self.hkl, self.cell.data)

    def get_radial_shells(
        self,
        n_shells: int = 20,
        d_min: Optional[float] = None,
        d_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create uniform radial shells in 1/d space for normalization.

        This creates shells with uniform spacing in reciprocal space (1/d),
        different from get_bins() which creates equal-count bins.

        Parameters
        ----------
        n_shells : int
            Number of radial shells. Default is 20.
        d_min : float, optional
            High resolution limit in Angstroms. If None, uses dataset minimum.
        d_max : float, optional
            Low resolution limit in Angstroms. If None, uses dataset maximum.

        Returns
        -------
        shell_edges : torch.Tensor
            Shell boundaries in Angstroms^-1, shape (n_shells+1,).
        shell_centers : torch.Tensor
            Shell centers in Angstroms^-1, shape (n_shells,).
        shell_indices : torch.Tensor
            Shell index for each reflection, shape (N,). Values -1 for out-of-range.
        """
        from torchref.base.normalization import (
            assign_to_shells,
            compute_radial_shells,
        )

        if self.resolution is None:
            self._calculate_resolution()

        # Get resolution limits
        if d_min is None:
            d_min = self.get_max_res()
        if d_max is None:
            d_max = self.get_min_res()

        # Compute shells
        shell_edges, shell_centers = compute_radial_shells(
            d_min, d_max, n_shells, device=self.device
        )

        # Get s-vectors and magnitudes
        s_vectors = self.get_scattering_vectors()
        s_mag = torch.linalg.norm(s_vectors, dim=1)

        # Assign to shells
        shell_indices = assign_to_shells(s_mag, shell_edges)

        # Cache shell indices
        self.radial_shell_indices = shell_indices

        return shell_edges, shell_centers, shell_indices

    def fit_anisotropy(
        self,
        n_shells: int = 20,
        d_min: Optional[float] = None,
        d_max: Optional[float] = None,
        n_iterations: int = 100,
        verbose: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Fit anisotropy correction parameters to minimize CV within shells.

        Optimizes U parameters so that corrected F² values have minimal
        coefficient of variation within each resolution shell.

        Parameters
        ----------
        n_shells : int
            Number of resolution shells for variance calculation.
        d_min : float, optional
            High resolution limit in Angstroms. If None, uses dataset minimum.
        d_max : float, optional
            Low resolution limit in Angstroms. If None, uses dataset maximum.
        n_iterations : int
            Number of optimization iterations.
        verbose : bool, optional
            Print progress. If None, uses self.verbose.

        Returns
        -------
        U : torch.Tensor
            Fitted anisotropy parameters [u11, u22, u33, u12, u13, u23], shape (6,).
            Also stored in self.U_aniso.

        Raises
        ------
        ValueError
            If no amplitude data is available.
        """
        from torchref.base import fit_anisotropy_correction

        if self.F is None:
            raise ValueError("No amplitude data loaded")

        if verbose is None:
            verbose = self.verbose > 0

        # Get F² values
        F_squared = self.F**2

        # Get s-vectors
        s_vectors = self.get_scattering_vectors()

        # Get resolution limits
        if d_min is None:
            d_min = self.get_max_res()
        if d_max is None:
            d_max = self.get_min_res()

        # Fit anisotropy
        U, final_cv = fit_anisotropy_correction(
            F_squared,
            s_vectors,
            n_shells=n_shells,
            d_min=d_min,
            d_max=d_max,
            n_iterations=n_iterations,
            verbose=verbose,
        )

        # Store result
        self.U_aniso = U

        return U
    
    def setup_anisotropy(
        self,
        U_aniso: Optional[torch.Tensor] = None,
    ) -> None:
        
        """
        Setup anisotropy correction parameters.
        Parameters
        ----------
        U_aniso : torch.Tensor, optional
            Anisotropic parameters [u11, u22, u33, u12, u13, u23], shape (6,).
            If None, uses Initializes as zeros.
        """

        if U_aniso is None:
            U_aniso = torch.zeros(6, device=self.device, dtype=dtypes.float, requires_grad=False)
        else:
            U_aniso = U_aniso.to(device=self.device, dtype=dtypes.float, requires_grad=False)
        self.U_aniso = U_aniso

        return self

    def apply_anisotropy_correction(
        self,
        U_aniso: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply anisotropy correction to F² values.

        Parameters
        ----------
        U_aniso : torch.Tensor, optional
            Anisotropic parameters [u11, u22, u33, u12, u13, u23], shape (6,).
            If None, uses self.U_aniso (must have called fit_anisotropy first).

        Returns
        -------
        F_corrected: torch.Tensor
            Anisotropy-corrected F values, shape (N,).
        sigma_F_corrected: torch.Tensor
            Uncertainties of corrected F values, shape (N,).
        Raises
        ------
        ValueError
            If no U parameters available and none provided.
        """
        from torchref.base import apply_anisotropy_correction

        if U_aniso is None:
            U_aniso = self.U_aniso
        if U_aniso is None:
            raise ValueError(
                "No anisotropy parameters available. "
                "Call fit_anisotropy() first or provide U_aniso."
            )

        if self.F is None:
            raise ValueError("No amplitude data loaded")

        # Get s-vectors
        s_vectors = self.get_scattering_vectors()
        # Use raw tensors directly to preserve gradient flow
        # (MaskedTensor doesn't support autograd operations)
        F = self.F
        sigma = self.F_sigma
        # Apply correction
        F_corrected = apply_anisotropy_correction(F, s_vectors, U_aniso)
        sigma_F_corrected = apply_anisotropy_correction(
            sigma, s_vectors, U_aniso
        ) if sigma is not None else None

        return F_corrected, sigma_F_corrected

    def compute_e_values(
        self,
        n_shells: int = 20,
        d_min: Optional[float] = None,
        d_max: Optional[float] = None,
        apply_anisotropy: bool = True,
        fit_anisotropy: bool = True,
        verbose: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Compute E-values with optional anisotropy correction.

        E-values are normalized structure factors where <E²> = 1 within each
        resolution shell. Anisotropy correction can be applied first to account
        for directional variation in diffraction.

        Parameters
        ----------
        n_shells : int
            Number of resolution shells for normalization.
        d_min : float, optional
            High resolution limit in Angstroms. If None, uses dataset minimum.
        d_max : float, optional
            Low resolution limit in Angstroms. If None, uses dataset maximum.
        apply_anisotropy : bool
            If True, apply anisotropy correction before E-value calculation.
        fit_anisotropy : bool
            If True and apply_anisotropy is True, fit anisotropy parameters.
            If False and apply_anisotropy is True, uses existing self.U_aniso.
        verbose : bool, optional
            Print progress. If None, uses self.verbose.

        Returns
        -------
        E : torch.Tensor
            E-values, shape (N,). Also stored in self.E.
            self.E_squared is also populated with E² values.

        Raises
        ------
        ValueError
            If no amplitude data is available.

        Examples
        --------
        Compute E-values with automatic anisotropy correction::

            data = ReflectionData().load_mtz('data.mtz')
            E = data.compute_e_values(n_shells=30)
            print(f"E-values: mean={E.mean():.3f}, std={E.std():.3f}")

        Compute E-values without anisotropy correction::

            E = data.compute_e_values(apply_anisotropy=False)
        """
        from torchref.base import F_squared_to_E_values

        if self.F is None:
            raise ValueError("No amplitude data loaded")

        if verbose is None:
            verbose = self.verbose > 0

        # Get resolution limits
        if d_min is None:
            d_min = self.get_max_res()
        if d_max is None:
            d_max = self.get_min_res()

        # Get F² values (possibly with anisotropy correction)
        if apply_anisotropy:
            if fit_anisotropy:
                self.fit_anisotropy(
                    n_shells=n_shells, d_min=d_min, d_max=d_max, verbose=verbose
                )
            F_squared = self.apply_anisotropy_correction()[0] ** 2 
        else:
            F_squared = self.F**2

        # Get s-vectors
        s_vectors = self.get_scattering_vectors()

        # Compute E-values
        E, E_squared, shell_idx = F_squared_to_E_values(
            F_squared, s_vectors, n_shells=n_shells, d_min=d_min, d_max=d_max
        )

        # Store results
        self.E = E
        self.E_squared = E_squared
        self.radial_shell_indices = shell_idx

        if verbose:
            print(f"E-value statistics:")
            print(f"  E range: [{E.min():.3f}, {E.max():.3f}]")
            print(f"  E mean: {E.mean():.3f}, std: {E.std():.3f}")
            print(f"  E² mean: {E_squared.mean():.3f} (should be ~1.0)")
        return E

    def setup_scale(self, scale: Optional[float] = None) -> float:
        """
        Set overall scale factor, parametrized in log space. 

        Parameters
        ----------
        scale : float, optional
            If provided, sets the scale factor directly.
            If None, computes scale to make mean F equal to 1.0.

        Returns
        -------
        float
            The scale factor applied.
        """
        if scale is None:
            self.log_scale = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=dtypes.float)
        else:
            self.log_scale = torch.log(torch.tensor(scale, device=self.device, requires_grad=False, dtype=dtypes.float))
        return self

    def get_corrected_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get scaled amplitude and uncertainty tensors.

        Applies the current scale factor (from self.log_scale) to F and F_sigma.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Scaled F and F_sigma tensors. If log_scale is None, returns original.
        """

        if not hasattr(self, "log_scale") or self.log_scale is None:
            raise ValueError("Scale not set up. Call setup_scale() first.")
        if not hasattr(self, "U_aniso") or self.U_aniso is None:
            raise ValueError("Anisotropy not set up. Call setup_anisotropy() first.")
        F_corrected, F_sigma_corrected = self.apply_anisotropy_correction()
        scale_factor = torch.exp(self.log_scale)
        F_scaled = F_corrected * scale_factor
        F_sigma_scaled = F_sigma_corrected * scale_factor

        return F_scaled, F_sigma_scaled
    
    def parameters(self) -> List[Parameter]:
        """
        Get list of learnable parameters for optimization.

        Returns
        -------
        iter[Parameter]
            iter of torch Parameters to be optimized. Includes log_scale and U_aniso if set.
        """
        params = []
        if self.log_scale is not None:
            params.append(self.log_scale)
        if self.U_aniso is not None:
            params.append(self.U_aniso)
        return params
