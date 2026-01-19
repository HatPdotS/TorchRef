"""
MTZ file format reading and writing.

This module provides functions for reading and writing MTZ files containing
crystallographic reflection data (structure factor amplitudes, intensities,
R-free flags, etc.).

Space groups are returned as gemmi.SpaceGroup objects for consistency
throughout torchref.

Functions
---------
read
    Read an MTZ file and return a reader object.
write
    Write reflection data to an MTZ file.

Classes
-------
MTZReader
    Reader class for MTZ files.

Examples
--------
::

    from torchref.io import mtz
    
    # Reading
    reader = mtz.read('data.mtz', verbose=1)
    data_dict, cell, spacegroup = reader()
    print(spacegroup.short_name())  # gemmi.SpaceGroup object
    
    # Writing
    mtz.write(df, cell, spacegroup, 'output.mtz')
"""

from typing import Optional, Tuple, Union

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import torch


class MTZReader:
    """
    Reader for MTZ files containing crystallographic structure factor data.

    This class reads MTZ files using reciprocalspaceship and extracts:
    - Miller indices (h, k, l)
    - Structure factor amplitudes or intensities
    - Associated uncertainties (sigma values)
    - R-free test set flags

    Attributes
    ----------
    verbose : int
        Verbosity level for logging (0=silent, 1=normal, 2=debug).
    data : dict
        Dictionary containing extracted data arrays.
    cell : np.ndarray
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : gemmi.SpaceGroup
        Space group object.

    Examples
    --------
    ::

        reader = mtz.read('data.mtz', verbose=1)
        data_dict, cell, spacegroup = reader()
        print(f"Found {len(data_dict['HKL'])} reflections in {spacegroup.short_name()}")
    """

    AMPLITUDE_PRIORITY = [
        "F-obs",
        "FOBS",
        "FP",
        "F",
        "F-obs-filtered",
        "FOBS-filtered",
        "F(+)",
        "FPLUS",
        "FMEAN",
        "F-pk",
        "F_pk",
        "FO",
        "FODD",
        "F-model",
        "FC",
        "FCALC",
    ]

    INTENSITY_PRIORITY = [
        "I-obs",
        "IOBS",
        "I",
        "IMEAN",
        "I-obs-filtered",
        "IOBS-filtered",
        "I(+)",
        "IPLUS",
        "IP",
        "I-pk",
        "I_pk",
        "IHLI",
        "I_full",
        "IOBS_full",
        "IO",
    ]

    RFREE_FLAG_NAMES = [
        "R-free-flags",
        "RFREE",
        "FreeR_flag",
        "FREE",
        "R-free",
        "Rfree",
        "FREER",
        "FREE_FLAG",
        "test",
        "TEST",
        "free",
        "Free",
    ]

    def __init__(self, verbose: int = 0):
        """
        Initialize MTZ reader.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level (0=silent, 1=normal, 2=debug). Default is 0.
        """
        self.verbose = verbose
        self.data = None
        self.cell = None
        self.spacegroup = None
        self.mtz_data = None

    def read(self, filepath: str) -> "MTZReader":
        """
        Read an MTZ file and extract reflection data.

        Parameters
        ----------
        filepath : str
            Path to the MTZ file.

        Returns
        -------
        MTZReader
            Self, for method chaining.
        """
        self.data = dict()
        if self.verbose > 1:
            print(f"Reading MTZ file: {filepath}")

        self.mtz_data = rs.read_mtz(filepath)
        self.cell = np.array(
            [
                self.mtz_data.cell.a,
                self.mtz_data.cell.b,
                self.mtz_data.cell.c,
                self.mtz_data.cell.alpha,
                self.mtz_data.cell.beta,
                self.mtz_data.cell.gamma,
            ]
        )
        hkl = self.mtz_data.reset_index()[["H", "K", "L"]].to_numpy().astype(np.int32)
        self.data["HKL"] = hkl
        # Store as gemmi.SpaceGroup object
        self.spacegroup = self.mtz_data.spacegroup

        self._extract_amplitudes_and_intensities()
        self._extract_rfree_flags()

        return self

    def __call__(self) -> Tuple[dict, np.ndarray, gemmi.SpaceGroup]:
        """
        Return extracted data in a standardized format.

        Returns
        -------
        data : dict
            Dictionary with extracted data arrays.
        cell : np.ndarray
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : gemmi.SpaceGroup
            Space group object.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")
        return self.data, self.cell, self.spacegroup

    def _extract_amplitudes_and_intensities(self) -> None:
        """Extract amplitude and intensity data with priority ordering."""
        available_cols = set(self.mtz_data.columns)

        # Find intensity column
        intensity_col = None
        for col in self.INTENSITY_PRIORITY:
            if col in available_cols:
                dtype = str(self.mtz_data.dtypes[col])
                if "Intensity" in dtype or "J" in dtype:
                    intensity_col = col
                    break

        # Find amplitude column
        amplitude_col = None
        for col in self.AMPLITUDE_PRIORITY:
            if col in available_cols:
                dtype = str(self.mtz_data.dtypes[col])
                if "SFAmplitude" in dtype or "F" in dtype:
                    amplitude_col = col
                    break

        # Extract data
        if intensity_col:
            self.data["I"] = self.mtz_data[intensity_col].to_numpy().astype(np.float32)
            self.data["I_col"] = intensity_col
            sigma_col = self._find_sigma_column(intensity_col, is_intensity=True)
            if sigma_col:
                self.data["SIGI"] = (
                    self.mtz_data[sigma_col].to_numpy().astype(np.float32)
                )
                self.data["SIGI_col"] = sigma_col

        if amplitude_col:
            self.data["F"] = self.mtz_data[amplitude_col].to_numpy().astype(np.float32)
            self.data["F_col"] = amplitude_col
            sigma_col = self._find_sigma_column(amplitude_col, is_intensity=False)
            if sigma_col:
                self.data["SIGF"] = (
                    self.mtz_data[sigma_col].to_numpy().astype(np.float32)
                )
                self.data["SIGF_col"] = sigma_col

    def _extract_rfree_flags(self) -> None:
        """Extract R-free flags from the dataset."""
        available_cols = set(self.mtz_data.columns)

        for col in self.RFREE_FLAG_NAMES:
            if col in available_cols:
                dtype = str(self.mtz_data.dtypes[col])
                if "int" in dtype.lower() or "flag" in dtype.lower() or "I" in dtype:
                    try:
                        flags = self.mtz_data[col].to_numpy()

                        if flags.dtype == object or not np.issubdtype(
                            flags.dtype, np.integer
                        ):
                            flags = pd.to_numeric(flags, errors="coerce")
                            flags = np.nan_to_num(flags, nan=-1).astype(np.int32)
                        else:
                            flags = flags.astype(np.int32)

                        rfree_flags = np.array(flags, dtype=np.int32)
                        n_free = (rfree_flags == 0).sum()
                        free_pct = (
                            100.0 * n_free / len(rfree_flags)
                            if len(rfree_flags) > 0
                            else 0
                        )

                        # Flip convention if needed
                        if free_pct > 50.0:
                            flipped = np.zeros_like(rfree_flags)
                            flipped[rfree_flags == 0] = 1
                            flipped[rfree_flags > 0] = 0
                            flipped[rfree_flags < 0] = -1
                            rfree_flags = flipped

                            if self.verbose > 0:
                                n_free = (rfree_flags == 0).sum()
                                free_pct = 100.0 * n_free / len(rfree_flags)
                                print(f"   After flip: free={n_free} ({free_pct:.1f}%)")

                        self.data["R-free-flags"] = rfree_flags.astype(bool)
                        self.data["R-free-source"] = col
                        return

                    except Exception as e:
                        if self.verbose > 0:
                            print(
                                f"Warning: Could not load R-free flags from {col}: {e}"
                            )

    def _find_sigma_column(self, data_col: str, is_intensity: bool) -> Optional[str]:
        """Find the sigma column for a data column."""
        available_cols = set(self.mtz_data.columns)
        sigma_variants = [
            f"SIG{data_col}",
            f"SIGM{data_col}",
            f"{data_col}_sigma",
            f"{data_col}-sigma",
        ]

        if is_intensity:
            sigma_variants.extend(
                [
                    data_col.replace("I", "SIGI", 1),
                    data_col.replace("I-", "SIGI-"),
                    "SIGI",
                    "SIGIMEAN",
                    "SIGI-obs",
                    "SIGIOBS",
                ]
            )
        else:
            sigma_variants.extend(
                [
                    data_col.replace("F", "SIGF", 1),
                    data_col.replace("F-", "SIGF-"),
                    "SIGF",
                    "SIGFOBS",
                    "SIGF-obs",
                    "SIGFP",
                ]
            )

        for sigma_col in sigma_variants:
            if sigma_col in available_cols:
                dtype = str(self.mtz_data.dtypes[sigma_col])
                if "Stddev" in dtype or "Sigma" in dtype or "SIG" in sigma_col.upper():
                    return sigma_col

        return None


def read(filepath: str, verbose: int = 0) -> MTZReader:
    """
    Read an MTZ file.

    Parameters
    ----------
    filepath : str
        Path to the MTZ file.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    MTZReader
        Reader object with data loaded.
    """
    return MTZReader(verbose=verbose).read(filepath)


def write(
    df: pd.DataFrame,
    cell: Union[list, np.ndarray, torch.Tensor],
    spacegroup: Union[str, gemmi.SpaceGroup],
    filepath: str,
) -> int:
    """
    Write a DataFrame to an MTZ file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing reflection data. Expected columns include
        H, K, L (Miller indices) and data columns like Fobs, I-obs, etc.
    cell : list, numpy.ndarray, or torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] in A and degrees.
    spacegroup : str or gemmi.SpaceGroup
        Space group symbol or gemmi SpaceGroup object.
    filepath : str
        Output MTZ filename.

    Returns
    -------
    int
        Returns 1 on success.
    """
    import gemmi

    mtz_rs = rs.DataSet(df)
    if "H" in mtz_rs.columns and "K" in mtz_rs.columns and "L" in mtz_rs.columns:
        mtz_rs = mtz_rs.set_index("H", "K", "L")

    if torch.is_tensor(cell):
        cell = cell.detach().cpu().numpy().tolist()
    elif isinstance(cell, np.ndarray):
        cell = cell.tolist()

    # Handle spacegroup
    if isinstance(spacegroup, gemmi.SpaceGroup):
        pass
    elif isinstance(spacegroup, str):
        if spacegroup.startswith("<gemmi.SpaceGroup"):
            import re

            match = re.search(r'SpaceGroup\("([^"]+)"\)', spacegroup)
            if match:
                spacegroup = gemmi.SpaceGroup(match.group(1))
            else:
                raise ValueError(f"Could not parse spacegroup string: {spacegroup}")
        else:
            spacegroup = gemmi.SpaceGroup(spacegroup)
    else:
        raise ValueError(
            f"Spacegroup must be str or gemmi.SpaceGroup, got {type(spacegroup)}"
        )

    # Assign MTZ data types
    structure_factor_cols = ["Fobs", "2FOFCWT", "FOFCWT", "F-model"]
    intensity_cols = ["I-obs"]
    sigma_cols = ["SIGF-obs", "SIGI-obs"]
    phase_cols = ["PH2FOFCWT", "PHFOFCWT", "PH-model"]
    flags = ["R-free-flags"]

    for col in structure_factor_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype("F")

    for col in intensity_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype("J")

    for col in sigma_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype("Q")

    for col in phase_cols:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype("P")

    for col in flags:
        if col in mtz_rs.columns:
            mtz_rs[col] = mtz_rs[col].astype("I")

    mtz_rs = mtz_rs.infer_mtz_dtypes()
    mtz_rs.cell = gemmi.UnitCell(*cell)
    mtz_rs.spacegroup = spacegroup
    mtz_rs.write_mtz(filepath)

    return 1


# Legacy alias for backwards compatibility during transition
MTZ = MTZReader
