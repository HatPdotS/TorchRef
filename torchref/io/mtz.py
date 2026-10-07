"""
MTZ reading and writing: amplitudes, intensities, sigmas and R-free flags.

``read`` returns a reader object, not the data -- call it for the tuple::

    data_dict, cell, spacegroup = mtz.read('data.mtz')()
    mtz.write(df, cell, spacegroup, 'output.mtz')
    mtz.write_reflections(data, 'output.mtz', fcalc=fcalc)   # a ReflectionData

The space group comes back as an H-M symbol **string** (``"P 21 21 21"``), not
a SpaceGroup object; callers wrap it themselves.
"""

import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import torch

from torchref.base.fourier.coefficients import map_coefficients
from torchref.config import get_int_dtype

if TYPE_CHECKING:
    from torchref.io.datasets.reflection_data import ReflectionData


class MTZReader:
    """
    Reader for MTZ files: Miller indices, amplitudes/intensities, sigmas, flags.

    Column selection is by priority list (``AMPLITUDE_PRIORITY`` etc.) unless
    ``column_names`` pins it. Populated by :meth:`read`; call the instance for
    ``(data, cell, spacegroup)``.

    Attributes
    ----------
    data : dict
        Extracted arrays; see :meth:`__call__` for the keys.
    cell : np.ndarray
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str
        H-M symbol string, e.g. ``"P 21 21 21"``.
    friedel_merged : bool
        False once anomalous columns have been stacked into Bijvoet pairs.
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

    # Optional third-class flag for ensemble refinement / hyperparameter tuning.
    # Distinct from R-free: validation reflections are held out for tuning
    # regularization strength while R-free stays untouched for final reporting.
    VALIDATION_FLAG_NAMES = [
        "Validation_flag",
        "Validation-flags",
        "VAL_FLAG",
        "VALID",
        "Validation",
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
            Example: ``{"F": "dFo", "SIGF": "sig_dFo"}``. A pinned column is
            read as what its MTZ type says it is: an ``"F"`` pin naming an
            intensity (type J or K) is an ``"I"`` pin, and an ``"I"`` pin
            naming an amplitude (F or G) an ``"F"`` pin, with the sigma pin
            following. Pinning one kind of data turns off the search for the
            other, so ``{"F": "FP"}`` loads amplitudes even when the file also
            has intensities, and French-Wilson does not run.
        anomalous : bool, optional
            None (default) stacks ``F(+)/F(-)`` (or ``I(+)/I(-)``) into explicit
            Friedel pairs when such columns exist; True forces that (a notice at
            verbose > 0 if none exist); False forces a merged load, averaging pairs.
        """
        self.verbose = verbose
        # A copy: pins are re-keyed by MTZ type when a file is read.
        self.column_names = dict(column_names or {})
        # The data column anomalous stacking chose, consulted after the pins.
        self._stacked_columns = {}
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
        self._key_pins_by_type()
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
        self.spacegroup = self.mtz_data.spacegroup.hm

        self._extract_amplitudes_and_intensities()
        self._extract_rfree_flags()
        self._extract_validation_flags()

        # Carry the merge state out to the caller (ReflectionData.load reads it).
        self.data["friedel_merged"] = self.friedel_merged

        return self

    def _key_pins_by_type(self) -> None:
        """Re-key a pinned data column (and its sigma) by the column's MTZ type.

        Intensities are types J and K, amplitudes F and G; a pin under the
        other family's key moves to the right one unless that key is pinned
        too. Pins naming a column the file lacks are left for extraction to
        report.
        """
        pins = self.column_names
        for key, sigma, other, other_sigma, types in (
            ("F", "SIGF", "I", "SIGI", "JK"),
            ("I", "SIGI", "F", "SIGF", "FG"),
        ):
            column = pins.get(key)
            if column not in self.mtz_data.columns or other in pins:
                continue
            if getattr(self.mtz_data.dtypes[column], "mtztype", "") in types:
                pins[other] = pins.pop(key)
                if sigma in pins:
                    pins[other_sigma] = pins.pop(sigma)

    def _maybe_stack_anomalous(self) -> None:
        """Stack anomalous F(+)/F(-) (or I(+)/I(-)) columns into Bijvoet pairs.

        One row per Friedel mate, the minus member carrying the negated Miller
        index (centrics are not split). Records the chosen base column so the
        priority search cannot pick a coexisting merged column such as
        ``FMEAN`` instead; a pin in ``column_names`` still wins. On any failure
        the merged data stand.
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

        # Choose the stacked data column so extraction uses it (and not a coexisting
        # merged column via the priority search). Prefer intensities so French-Wilson
        # runs per Bijvoet member. The matching sigma is auto-discovered by
        # _extract_amplitudes_and_intensities. A pin of either kind wins.
        intensity_bases = [b for b in bases if "Intensity" in str(stacked.dtypes[b])]
        amplitude_bases = [b for b in bases if "SFAmplitude" in str(stacked.dtypes[b])]
        if intensity_bases:
            self._stacked_columns["I"] = intensity_bases[0]
        elif amplitude_bases:
            self._stacked_columns["F"] = amplitude_bases[0]

        if self.verbose > 0:
            print(
                f"   Detected anomalous data; stacked Bijvoet pairs "
                f"({len(self.mtz_data)} reflections, friedel_merged=False)."
            )

    def _merge_anomalous_columns(self) -> None:
        """Average F(+)/F(-) (and sigmas) into merged base columns (anomalous=False).

        Amplitudes/intensities by mean, sigmas in quadrature
        ``sqrt((s+^2 + s-^2)/4)``; the (+)/(-) columns are then dropped. No-op
        when there is no pair or the merged column already exists.
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
            Dictionary with extracted data arrays. Keys present depend on the
            file, and may include: ``"HKL"`` (int32 Miller indices); ``"F"`` /
            ``"SIGF"`` and/or ``"I"`` / ``"SIGI"`` (float32 data, with
            ``"*_col"`` provenance keys recording the source column names);
            ``"R-free-flags"`` (int32: ``1`` = work, ``0`` = free, ``-1`` =
            excluded, as :func:`~torchref.io.rfree.read_free_set` reads the
            column -- CCP4 ``0..K`` with ``0`` free, or a binary column whose
            majority value is work) and ``"R-free-source"``;
            ``"Validation-flags"`` (a **bool** mask) and ``"Validation-source"``;
            and ``"friedel_merged"`` (bool) indicating the Bijvoet state of the
            returned data (False when anomalous F(+)/F(-) pairs were stacked).
        cell : np.ndarray
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : str
            Space group as an H-M symbol string (e.g. ``"P 21 21 21"``).
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read() first.")
        return self.data, self.cell, self.spacegroup

    def _extract_amplitudes_and_intensities(self) -> None:
        """Extract amplitude and intensity data with priority ordering.

        If ``column_names`` were provided at init, those columns are used
        directly instead of the priority-based search. The ``"F"`` / ``"I"``
        keys override the amplitude / intensity column, and the ``"SIGF"`` /
        ``"SIGI"`` keys override their associated sigma columns (otherwise the
        sigma column is auto-discovered). A pin of one kind turns off the
        search for the other. Without a pin of its kind, the column anomalous
        stacking chose replaces the search.
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
        elif "F" in self.column_names:
            intensity_col = None
        elif "I" in self._stacked_columns:
            intensity_col = self._stacked_columns["I"]
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
        elif "I" in self.column_names:
            amplitude_col = None
        elif "F" in self._stacked_columns:
            amplitude_col = self._stacked_columns["F"]
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
        """Extract R-free flags from the first integer ``RFREE_FLAG_NAMES`` column.

        The column is interpreted by :func:`~torchref.io.rfree.read_free_set`,
        the rule SF-mmCIF flags are read with too.
        """
        # rfree imports this module for its flag names.
        from torchref.io.rfree import read_free_set

        available_cols = set(self.mtz_data.columns)

        for col in self.RFREE_FLAG_NAMES:
            if col in available_cols:
                dtype = str(self.mtz_data.dtypes[col])
                if "int" in dtype.lower() or "flag" in dtype.lower() or "I" in dtype:
                    try:
                        free_set = read_free_set(self.mtz_data, col)
                        rfree_flags = np.where(free_set["free"], 0, 1).astype(np.int32)
                        rfree_flags[free_set["excluded"]] = -1
                        if self.verbose > 0:
                            print(
                                f"   R-free flags from '{col}': "
                                f"{free_set['convention']}, "
                                f"free={int(free_set['free'].sum())}"
                            )

                        # keep int: -1 (excluded) is masked by ReflectionData.load
                        self.data["R-free-flags"] = rfree_flags
                        self.data["R-free-source"] = col
                        return

                    except Exception as e:
                        if self.verbose > 0:
                            print(
                                f"Warning: Could not load R-free flags from {col}: {e}"
                            )

    def _extract_validation_flags(self) -> None:
        """Extract the optional third-class validation flags (1 = validation).

        Held out for tuning, distinct from the R-free test set. First matching
        column in ``VALIDATION_FLAG_NAMES`` wins; failures are only warned about.
        """
        available_cols = set(self.mtz_data.columns)
        for col in self.VALIDATION_FLAG_NAMES:
            if col in available_cols:
                try:
                    flags = self.mtz_data[col].to_numpy()
                    if not np.issubdtype(flags.dtype, np.integer):
                        flags = pd.to_numeric(flags, errors="coerce")
                        flags = np.nan_to_num(flags, nan=0).astype(np.int32)
                    else:
                        flags = flags.astype(np.int32)
                    self.data["Validation-flags"] = (flags == 1).astype(bool)
                    self.data["Validation-source"] = col
                    if self.verbose > 0:
                        n_val = int((flags == 1).sum())
                        print(
                            f"   Loaded validation flags from '{col}': "
                            f"{n_val} reflections ({100.0 * n_val / max(len(flags), 1):.1f}%)"
                        )
                    return
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not load validation flags from {col}: {e}")

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
        Reader object; call it for ``(data, cell, spacegroup)``. This wrapper
        forwards only ``verbose`` -- for ``column_names`` or ``anomalous``,
        build an :class:`MTZReader` yourself.
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
        H, K, L (Miller indices) and recognized data columns such as
        ``F-obs``, ``I-obs``, ``SIGF-obs``, ``SIGI-obs`` (the names used to
        assign MTZ data types; see the column lists in the implementation).
    cell : list, numpy.ndarray, or torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] in A and degrees.
    spacegroup : str or gemmi.SpaceGroup
        Space group symbol or gemmi SpaceGroup object.
    filepath : str
        Output MTZ filename.

    Returns
    -------
    int
        Always returns 1 (failures raise rather than return a sentinel).
    """
    if torch.is_tensor(cell):
        cell = cell.detach().cpu().numpy().tolist()
    elif isinstance(cell, np.ndarray):
        cell = cell.tolist()

    cell = gemmi.UnitCell(*cell)

    # reciprocalspaceship needs a gemmi.SpaceGroup.
    from torchref.symmetry import SpaceGroup as TorchRefSpaceGroup

    if isinstance(spacegroup, TorchRefSpaceGroup):
        spacegroup = spacegroup._gemmi
    elif isinstance(spacegroup, gemmi.SpaceGroup):
        pass
    elif isinstance(spacegroup, str):
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
    flags = [
        "R-free-flags",
        "FreeR_flag",
        "FREE",
        "Validation_flag",
        "Validation-flags",
        "VAL_FLAG",
    ]

    if "H" in mtz_rs.columns and "K" in mtz_rs.columns and "L" in mtz_rs.columns:
        mtz_rs["H"] = mtz_rs["H"].astype("H")
        mtz_rs["K"] = mtz_rs["K"].astype("H")
        mtz_rs["L"] = mtz_rs["L"].astype("H")
        mtz_rs = mtz_rs.set_index(["H", "K", "L"])

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


def _np(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    return None if t is None else t.detach().cpu().numpy()


def _rfree_column(data: "ReflectionData") -> np.ndarray:
    """The R-free flags as written, per row: 1 = work, 0 = free, -1 = excluded.

    Excluded rows are those whose input flag was negative or missing
    (``masks["flagged_initial"]`` False); written as free, they would join the
    test set on the next read.
    """
    flags = (_np(data.rfree_flags) != 0).astype(int)
    flagged_initial = data.masks.get("flagged_initial")
    if flagged_initial is not None:
        flags[~_np(flagged_initial)] = -1
    return flags


def _flag_columns(data: "ReflectionData") -> dict:
    """The flag columns both layouts write, per row, keyed by label.

    R-free-flags as :func:`_rfree_column` gives it and, for a non-empty
    validation set, Validation_flag (1 = validation): a separate column, so
    external tools keep reading R-free-flags.
    """
    if data.rfree_flags is None:
        return {}
    columns = {"R-free-flags": _rfree_column(data)}
    if data.validation_flags is not None and bool(data.validation_flags.any()):
        columns["Validation_flag"] = (_np(data.validation_flags) != 0).astype(int)
    return columns


def _amplitude_phase(coeff: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """``|c|`` and ``arg(c)`` in degrees: a negative coefficient becomes a 180° flip."""
    return _np(coeff.abs()), _np(torch.rad2deg(torch.angle(coeff)))


def reflection_table(
    data: "ReflectionData",
    fcalc: Optional[torch.Tensor] = None,
    anomalous: bool = False,
) -> pd.DataFrame:
    """The DataFrame :func:`write_reflections` writes, before MTZ typing.

    Parameters
    ----------
    data : ReflectionData
        Canonicalized dataset.
    fcalc : torch.Tensor, optional
        Complex structure factors row-aligned with ``data.hkl``, in the
        canonical-ASU convention (``data.structure_factors``) and on the scale
        of ``data.F``. Adds the model and map-coefficient columns.
    anomalous : bool, optional
        Phenix-style anomalous layout: one row per unique reflection, Bijvoet
        mates merged by mean amplitude for the display columns and unstacked
        into ``(+)/(-)`` columns, plus ANOM/PANOM. Otherwise one row per
        dataset row.

    Returns
    -------
    pandas.DataFrame
        Columns named as they are written: :func:`write` assigns MTZ types
        and renames nothing (``F-obs``, ``SIGF-obs``, ``I-obs``,
        ``R-free-flags``, ...).
    """
    if fcalc is not None and not torch.is_complex(fcalc):
        raise ValueError("fcalc must be a complex tensor")
    if anomalous:
        return _anomalous_table(data, fcalc)
    return _merged_table(data, fcalc)


def _merged_table(data, fcalc):
    hkl = _np(data.hkl)
    table = {"H": hkl[:, 0], "K": hkl[:, 1], "L": hkl[:, 2]}
    if data.F is not None:
        table["F-obs"] = _np(data.F)
        if data.F_sigma is not None:
            table["SIGF-obs"] = _np(data.F_sigma)
    if data.I is not None:
        table["I-obs"] = _np(data.I)
        if data.I_sigma is not None:
            table["SIGI-obs"] = _np(data.I_sigma)
    table.update(_flag_columns(data))
    if fcalc is not None:
        valid = data.masks().to(device=fcalc.device, dtype=torch.bool)
        two_fo_fc, fo_fc = map_coefficients(data.F, fcalc, observed=valid)
        table["FWT"], table["PHWT"] = _amplitude_phase(two_fo_fc)
        table["DELFWT"], table["PHDELWT"] = _amplitude_phase(fo_fc)
        table["F-model"], table["PH-model"] = _amplitude_phase(fcalc)
        # Past the resolution cut F_calc is unrefined, or aliased beyond the FFT
        # grid's Nyquist limit, so the model columns are missing there too.
        excluded = ~_np(valid)
        table["F-model"][excluded] = np.nan
        table["PH-model"][excluded] = np.nan
    return pd.DataFrame(table)


def _anomalous_table(data, fcalc):
    if data.friedel_flags is None:
        raise ValueError(
            "anomalous output requires canonicalized data with friedel_flags; "
            "load via load_mtz so Friedel bookkeeping is populated."
        )
    hkl = data.hkl.detach().cpu()
    n = hkl.shape[0]
    flag = data.friedel_flags.detach().cpu()
    inverse, m = data.asu_group_indices()
    inverse = inverse.cpu()
    uniq = hkl[data._group_representative_rows(inverse, m)]

    # A mate counts as present only if it is a real, positive observation:
    # stacked input carries a NaN row for every absent mate, which French-Wilson
    # maps to F=0, and pairing that phantom with its observed mate would write
    # the whole amplitude as the Bijvoet difference.
    F_cpu = data.F.detach().cpu()
    observed = torch.isfinite(F_cpu) & (F_cpu > 0)
    if data.F_sigma is not None:
        observed = observed & torch.isfinite(data.F_sigma.detach().cpu())
    arange = torch.arange(n, dtype=get_int_dtype())
    plus_idx = torch.full((m,), -1, dtype=get_int_dtype())
    minus_idx = torch.full((m,), -1, dtype=get_int_dtype())
    plus_sel, minus_sel = (~flag) & observed, flag & observed
    plus_idx[inverse[plus_sel]] = arange[plus_sel]
    minus_idx[inverse[minus_sel]] = arange[minus_sel]
    has_plus, has_minus = (plus_idx >= 0).numpy(), (minus_idx >= 0).numpy()
    pi, mi = plus_idx.clamp(min=0).numpy(), minus_idx.clamp(min=0).numpy()

    # Centrics obey Friedel's law, F(+) = F(-).
    centric = np.zeros(m, dtype=bool)
    if data.centric is not None:
        cen = _np(data.centric)
        centric[has_plus] = cen[pi][has_plus]
        centric[has_minus] = cen[mi][has_minus]

    def plus_of(src):
        out = np.full(m, np.nan, dtype=np.float64)
        out[has_plus] = src[pi][has_plus]
        return out

    def minus_of(src):
        out = np.full(m, np.nan, dtype=np.float64)
        out[has_minus] = src[mi][has_minus]
        return out

    def mirror_centric(plus, minus):
        p = np.where(centric & ~np.isfinite(plus) & np.isfinite(minus), minus, plus)
        q = np.where(centric & ~np.isfinite(minus) & np.isfinite(plus), plus, minus)
        return p, q

    F = _np(data.F)
    Fobs_p, Fobs_m = plus_of(F), minus_of(F)
    Fobs_p_out, Fobs_m_out = mirror_centric(Fobs_p, Fobs_m)
    # Mean over present mates; NaN where neither was measured.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Fobs_disp = np.nanmean(np.vstack([Fobs_p, Fobs_m]), axis=0)
    # Observed columns carry every measured mate; model and map columns only the
    # mates masks() keeps, merged over those alone so a rejected mate cannot leak
    # into the map amplitude.
    valid = _np(data.masks()).astype(bool)
    valid_p, valid_m = has_plus & valid[pi], has_minus & valid[mi]
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Fobs_map = np.nanmean(
            np.vstack(
                [np.where(valid_p, Fobs_p, np.nan), np.where(valid_m, Fobs_m, np.nan)]
            ),
            axis=0,
        )
    usable = valid_p | valid_m

    uniq_np = uniq.numpy()
    table = {
        "H": uniq_np[:, 0],
        "K": uniq_np[:, 1],
        "L": uniq_np[:, 2],
        "F-obs": np.nan_to_num(Fobs_disp, nan=0.0),
        "F-obs(+)": Fobs_p_out,
        "F-obs(-)": Fobs_m_out,
    }

    if fcalc is not None:
        fc = _np(fcalc)
        # The (+)/(-) phase columns describe each mate at its own index, the
        # signed convention; conjugate_friedel is its own inverse.
        Fc_ph = np.angle(_np(data.conjugate_friedel(fcalc)), deg=True)
        fc_amp = np.abs(fc)
        Fmod_p_out, Fmod_m_out = mirror_centric(
            np.where(valid_p, plus_of(fc_amp), np.nan),
            np.where(valid_m, minus_of(fc_amp), np.nan),
        )
        Phi_p_out, Phi_m_out = mirror_centric(
            np.where(valid_p, plus_of(Fc_ph), np.nan),
            np.where(valid_m, minus_of(Fc_ph), np.nan),
        )
        # Representative model value per reflection: the valid (+) row, else the
        # valid (-); missing where neither mate is valid.
        fc_disp = np.full(m, complex(np.nan, np.nan), dtype=complex)
        fc_disp[valid_p] = fc[pi][valid_p]
        only_minus = valid_m & ~valid_p
        fc_disp[only_minus] = fc[mi][only_minus]
        fc_disp_t = torch.from_numpy(fc_disp).to(fcalc.dtype)
        two_fo_fc, fo_fc = map_coefficients(
            torch.from_numpy(np.nan_to_num(Fobs_map, nan=0.0)),
            fc_disp_t,
            observed=torch.from_numpy(usable),
        )
        ph_disp = np.angle(fc_disp, deg=True)
        # Anomalous difference Fourier, phenix convention: ANOM = |F(+) - F(-)|
        # with the sign carried by a 180° flip in PANOM, so ANOM exp(i PANOM)
        # is (F(+) - F(-)) exp(i (phi - 90°)). Centric differences are exactly
        # zero, so any measured value is noise; they are omitted, as in phenix.
        anom = np.where(valid_p & valid_m, Fobs_p - Fobs_m, np.nan)
        panom = np.where(anom < 0.0, ph_disp - 270.0, ph_disp - 90.0)
        anom = np.abs(anom)
        anom[centric] = np.nan
        panom[centric] = np.nan
        table["F-model"], table["PH-model"] = _amplitude_phase(fc_disp_t)
        table["F-model(+)"], table["PHIF-model(+)"] = Fmod_p_out, Phi_p_out
        table["F-model(-)"], table["PHIF-model(-)"] = Fmod_m_out, Phi_m_out
        table["FWT"], table["PHWT"] = _amplitude_phase(two_fo_fc)
        table["DELFWT"], table["PHDELWT"] = _amplitude_phase(fo_fc)
        table["ANOM"], table["PANOM"] = anom, panom

    if data.F_sigma is not None:
        sig = _np(data.F_sigma)
        table["SIGF-obs(+)"], table["SIGF-obs(-)"] = mirror_centric(
            plus_of(sig), minus_of(sig)
        )
    for label, flags in _flag_columns(data).items():
        table[label] = np.full(m, -1, dtype=int)
        table[label][inverse.numpy()] = flags  # both mates share a flag
    return pd.DataFrame(table)


def write_reflections(
    data: "ReflectionData",
    filepath: str,
    fcalc: Optional[torch.Tensor] = None,
    anomalous: Optional[bool] = None,
    verbose: int = 0,
) -> None:
    """Write a :class:`~torchref.io.datasets.reflection_data.ReflectionData` to MTZ.

    Labels on disk are the :func:`reflection_table` column names. Merged: F-obs,
    SIGF-obs, I-obs, SIGI-obs; anomalous: F-obs, F-obs(+)/(-), SIGF-obs(+)/(-) and
    no intensities; both: R-free-flags (1 = work, 0 = free, -1 = excluded by the
    input's flags) and Validation_flag. With ``fcalc`` both add FWT/PHWT (2Fo-Fc),
    DELFWT/PHDELWT (Fo-Fc) and F-model/PH-model, the unweighted m = 1, D = 1
    coefficients of :func:`~torchref.base.fourier.map_coefficients`, not 2mFo-DFc;
    anomalous also F-model(+)/(-), PHIF-model(+)/(-) and ANOM/PANOM.
    R-free-flags is Phenix's label with the CCP4 free value 0, so tell Phenix
    the test-flag value rather than letting it assume 1.

    Map and model columns are missing (not filled) for every reflection
    ``data.masks()`` excludes -- beyond the resolution cut or rejected as an
    outlier -- so a map from them stops at the refinement resolution. Observed
    columns are written for every reflection.

    Parameters
    ----------
    data : ReflectionData
        Dataset to write.
    filepath : str
        Output path.
    fcalc : torch.Tensor, optional
        See :func:`reflection_table`.
    anomalous : bool, optional
        See :func:`reflection_table`. Default: anomalous exactly when the data
        hold Bijvoet pairs (``friedel_merged`` False).
    verbose : int, optional
        Print a summary when > 0.
    """
    if anomalous is None:
        anomalous = not data.friedel_merged
    df = reflection_table(data, fcalc, anomalous=anomalous)
    write(df, data.cell.data, data.spacegroup, filepath)
    if verbose > 0:
        layout = "anomalous (phenix-style)" if anomalous else "merged"
        print(f"✓ Wrote {layout} MTZ: {filepath}")
        print(f"  Reflections: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")
