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

    def __init__(
        self,
        verbose: int = 0,
        column_names: Optional[dict] = None,
        anomalous: Optional[bool] = None,
    ):
        """
        Initialize MTZ reader.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level (0=silent, 1=normal, 2=debug). Default is 0.
        column_names : dict, optional
            Explicit column name mapping to override automatic detection.
            Supported keys: ``"F"``, ``"SIGF"``, ``"I"``, ``"SIGI"``.
            Example: ``{"F": "DFo", "SIGF": "sig_DFo"}``.
        anomalous : bool, optional
            Controls anomalous (Bijvoet) handling. If None (default), anomalous
            ``F(+)/F(-)`` (or ``I(+)/I(-)``) columns are auto-detected and stacked
            into explicit Friedel pairs when present (preferring anomalous data).
            True forces that behavior (warns if no anomalous columns exist); False
            forces a merged load (anomalous columns, if any, are not stacked).
        """
        self.verbose = verbose
        self.column_names = column_names or {}
        self.anomalous = anomalous
        self.data = None
        self.cell = None
        self.spacegroup = None
        self.mtz_data = None
        # True when the loaded data are Friedel-merged (one row per ASU reflection);
        # set False by _maybe_stack_anomalous when F(+)/F(-) (or I(+)/I(-)) columns
        # are detected and stacked into explicit signed-HKL Bijvoet pairs.
        self.friedel_merged = True

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
        # Auto-detect anomalous F(+)/F(-) (or I(+)/I(-)) columns and stack them into
        # explicit signed-HKL Bijvoet pairs (one row per Friedel mate). Sets
        # self.friedel_merged accordingly. No-op for ordinary merged data.
        self._maybe_stack_anomalous()
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
        # Store as string (the caller wraps in SpaceGroup as needed)
        self.spacegroup = self.mtz_data.spacegroup.hm

        self._extract_amplitudes_and_intensities()
        self._extract_rfree_flags()

        # Carry the merge state out to the caller (ReflectionData.load reads it).
        self.data["friedel_merged"] = self.friedel_merged

        return self

    def _maybe_stack_anomalous(self) -> None:
        """Stack anomalous F(+)/F(-) (or I(+)/I(-)) columns into Bijvoet pairs.

        When the MTZ carries two-column anomalous data and is merged, this converts
        it to one-column form with separate rows for the Friedel-plus and
        Friedel-minus members (the minus member carries the negated Miller index),
        using :meth:`reciprocalspaceship.DataSet.stack_anomalous`. Centric
        reflections are not split (handled by reciprocalspaceship). The chosen base
        columns are pinned in ``column_names`` so the downstream priority search
        does not pick a coexisting merged column (e.g. ``FMEAN``) instead of the
        stacked one. On any failure the original merged data are kept.
        """
        if self.anomalous is False:
            # Caller forced a merged load. If the file carries only anomalous
            # columns (no coexisting merged column), average the pairs into merged
            # base columns so extraction can read them.
            self._merge_anomalous_columns()
            return

        cols = list(self.mtz_data.columns)
        plus_cols = [c for c in cols if c.endswith("(+)")]
        minus_cols = [c for c in cols if c.endswith("(-)")]
        if not plus_cols or not minus_cols:
            if self.anomalous is True and self.verbose > 0:
                print(
                    "   anomalous=True requested but no F(+)/F(-) columns found; "
                    "loading merged."
                )
            return  # no anomalous columns
        if not getattr(self.mtz_data, "merged", True):
            return  # stack_anomalous requires merged data

        # Match plus/minus columns by their base name (suffix is exactly 3 chars).
        plus_map = {c[:-3]: c for c in plus_cols}
        minus_map = {c[:-3]: c for c in minus_cols}
        bases = [b for b in plus_map if b in minus_map]
        if not bases:
            return

        plus_labels = [plus_map[b] for b in bases]
        minus_labels = [minus_map[b] for b in bases]

        # Drop columns that would block stacking: coexisting merged columns whose
        # name equals a base label (stack_anomalous raises on the collision), and
        # any unmatched anomalous singletons that would otherwise ride along.
        selected = set(plus_labels) | set(minus_labels)
        to_drop = [b for b in bases if b in cols]
        to_drop += [c for c in plus_cols + minus_cols if c not in selected]
        try:
            ds = self.mtz_data
            if to_drop:
                ds = ds.drop(columns=list(dict.fromkeys(to_drop)))
            stacked = ds.stack_anomalous(
                plus_labels=plus_labels, minus_labels=minus_labels
            )
        except Exception as e:  # keep merged data on any failure
            if self.verbose > 0:
                print(f"   Anomalous stacking skipped ({e}); using merged data.")
            return

        self.mtz_data = stacked
        self.friedel_merged = False

        # Pin the stacked data column so extraction uses it (and not a coexisting
        # merged column via the priority search). Prefer intensities so French-Wilson
        # runs per Bijvoet member. The matching sigma is auto-discovered by
        # _extract_amplitudes_and_intensities. Respect any user-provided names.
        intensity_bases = [b for b in bases if "Intensity" in str(stacked.dtypes[b])]
        amplitude_bases = [b for b in bases if "SFAmplitude" in str(stacked.dtypes[b])]
        if intensity_bases and "I" not in self.column_names:
            self.column_names["I"] = intensity_bases[0]
        elif amplitude_bases and "F" not in self.column_names:
            self.column_names["F"] = amplitude_bases[0]

        if self.verbose > 0:
            print(
                f"   Detected anomalous data; stacked Bijvoet pairs "
                f"({len(self.mtz_data)} reflections, friedel_merged=False)."
            )

    def _merge_anomalous_columns(self) -> None:
        """Average F(+)/F(-) (and sigmas) into merged base columns (anomalous=False).

        Only used when a merged load is forced. For each matched anomalous pair
        whose stripped base name is not already a column, a merged column is
        created: amplitudes/intensities by mean, sigmas in quadrature
        (sqrt((s+^2 + s-^2)/4)), with the de-stacked (standard) MTZ dtype. The
        anomalous (+)/(-) columns are then dropped. No-op when there is no
        anomalous pair or a merged column already exists.
        """
        cols = list(self.mtz_data.columns)
        plus_map = {c[:-3]: c for c in cols if c.endswith("(+)")}
        minus_map = {c[:-3]: c for c in cols if c.endswith("(-)")}
        bases = [b for b in plus_map if b in minus_map]
        if not bases:
            return

        ds = self.mtz_data
        dropped = []
        for b in bases:
            pcol, mcol = plus_map[b], minus_map[b]
            dropped += [pcol, mcol]
            if b in cols:
                continue  # a merged column with this name already exists
            p = ds[pcol].to_numpy(dtype="float32")
            m = ds[mcol].to_numpy(dtype="float32")
            mtztype = getattr(ds.dtypes[pcol], "mtztype", "")
            with np.errstate(invalid="ignore"):
                if mtztype in ("L", "M"):  # stddev -> quadrature combination
                    both = np.isfinite(p) & np.isfinite(m)
                    merged = np.where(np.isfinite(p), p, m)
                    merged[both] = np.sqrt((p[both] ** 2 + m[both] ** 2) / 4.0)
                else:  # amplitude / intensity -> mean over present mates
                    merged = np.nanmean(np.vstack([p, m]), axis=0)
            target_dtype = ds[pcol].from_friedel_dtype().dtype
            ds[b] = merged.astype("float32")
            ds[b] = ds[b].astype(target_dtype)

        self.mtz_data = ds.drop(columns=list(dict.fromkeys(dropped)))
        if self.verbose > 0:
            print("   anomalous=False: averaged F(+)/F(-) into merged columns.")

    def __call__(self) -> Tuple[dict, np.ndarray, str]:
        """
        Return extracted data in a standardized format.

        Returns
        -------
        data : dict
            Dictionary with extracted data arrays.
        cell : np.ndarray
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : str
            Space group name string.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")
        return self.data, self.cell, self.spacegroup

    def _extract_amplitudes_and_intensities(self) -> None:
        """Extract amplitude and intensity data with priority ordering.

        If ``column_names`` were provided at init, those columns are used
        directly instead of the priority-based search.
        """
        available_cols = set(self.mtz_data.columns)

        # --- Explicit column names override priority search ---
        if "I" in self.column_names:
            intensity_col = self.column_names["I"]
            if intensity_col not in available_cols:
                raise ValueError(
                    f"Intensity column '{intensity_col}' not found in MTZ. "
                    f"Available: {sorted(available_cols)}"
                )
        else:
            intensity_col = None
            for col in self.INTENSITY_PRIORITY:
                if col in available_cols:
                    dtype = str(self.mtz_data.dtypes[col])
                    if "Intensity" in dtype or "J" in dtype:
                        intensity_col = col
                        break

        if "F" in self.column_names:
            amplitude_col = self.column_names["F"]
            if amplitude_col not in available_cols:
                raise ValueError(
                    f"Amplitude column '{amplitude_col}' not found in MTZ. "
                    f"Available: {sorted(available_cols)}"
                )
        else:
            amplitude_col = None
            for col in self.AMPLITUDE_PRIORITY:
                if col in available_cols:
                    dtype = str(self.mtz_data.dtypes[col])
                    if "SFAmplitude" in dtype or "F" in dtype:
                        amplitude_col = col
                        break

        # Extract intensity data
        if intensity_col:
            self.data["I"] = self.mtz_data[intensity_col].to_numpy().astype(np.float32)
            self.data["I_col"] = intensity_col
            if "SIGI" in self.column_names:
                scol = self.column_names["SIGI"]
                if scol in available_cols:
                    self.data["SIGI"] = (
                        self.mtz_data[scol].to_numpy().astype(np.float32)
                    )
                    self.data["SIGI_col"] = scol
            else:
                sigma_col = self._find_sigma_column(intensity_col, is_intensity=True)
                if sigma_col:
                    self.data["SIGI"] = (
                        self.mtz_data[sigma_col].to_numpy().astype(np.float32)
                    )
                    self.data["SIGI_col"] = sigma_col

        # Extract amplitude data
        if amplitude_col:
            self.data["F"] = self.mtz_data[amplitude_col].to_numpy().astype(np.float32)
            self.data["F_col"] = amplitude_col
            if "SIGF" in self.column_names:
                scol = self.column_names["SIGF"]
                if scol in available_cols:
                    self.data["SIGF"] = (
                        self.mtz_data[scol].to_numpy().astype(np.float32)
                    )
                    self.data["SIGF_col"] = scol
            else:
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
        H, K, L (Miller indices) and data columns like F_obs, I_obs, etc.
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

    if torch.is_tensor(cell):
        cell = cell.detach().cpu().numpy().tolist()
    elif isinstance(cell, np.ndarray):
        cell = cell.tolist()

    cell = gemmi.UnitCell(*cell)

    # Handle spacegroup — normalize to gemmi.SpaceGroup for reciprocalspaceship
    from torchref.symmetry import SpaceGroup as TorchRefSpaceGroup

    if isinstance(spacegroup, TorchRefSpaceGroup):
        spacegroup = spacegroup._gemmi
    elif isinstance(spacegroup, gemmi.SpaceGroup):
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
            f"Spacegroup must be str, gemmi.SpaceGroup, or torchref SpaceGroup, got {type(spacegroup)}"
        )
    mtz_rs = rs.DataSet(df, cell=cell, spacegroup=spacegroup)

    # Assign MTZ data types
    structure_factor_cols = [
        "F-obs",
        "Fobs",
        "FP",
        "2FOFCWT",
        "FOFCWT",
        "F-model",
        "FWT",
        "DELFWT",
    ]
    intensity_cols = ["I-obs", "I"]
    sigma_cols = ["SIGF-obs", "SIGI-obs", "SIGFP", "SIGI"]
    phase_cols = [
        "PH2FOFCWT",
        "PHFOFCWT",
        "PH-model",
        "PHWT",
        "PHDELWT",
        "PHIF-model(+)",
        "PHIF-model(-)",
        "PANOM",
    ]
    flags = ["R-free-flags", "FreeR_flag", "FREE"]

    if "H" in mtz_rs.columns and "K" in mtz_rs.columns and "L" in mtz_rs.columns:
        mtz_rs["H"] = mtz_rs["H"].astype("H")
        mtz_rs["K"] = mtz_rs["K"].astype("H")
        mtz_rs["L"] = mtz_rs["L"].astype("H")
        mtz_rs = mtz_rs.set_index("H", "K", "L")

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
    mtz_rs.write_mtz(filepath)

    return 1


# Legacy alias for backwards compatibility during transition
MTZ = MTZReader
