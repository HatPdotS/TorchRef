"""
Legacy format readers for crystallographic data files.

This module provides readers for traditional crystallographic file formats
(MTZ, PDB) using specialized libraries (reciprocalspaceship, pandas).

Classes
-------
MTZ
    Reader for MTZ files containing structure factor data.
PDB
    Reader for PDB files containing atomic coordinate data.

Functions
---------
find_header_length_pdb_file
    Find the number of header lines in a PDB file.
load_pdb_as_pd
    Load a PDB file into a pandas DataFrame.
read_crystallographic_info
    Extract unit cell and space group from PDB file.
"""

import pandas as pd
import numpy as np
import reciprocalspaceship as rs
from typing import Optional, Tuple, List


class MTZ:
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
    spacegroup : str
        Space group symbol.

    Examples
    --------
    >>> mtz = MTZ(verbose=1).read('data.mtz')
    >>> data_dict, cell, spacegroup = mtz()
    >>> print(f"Found {len(data_dict['HKL'])} reflections")
    """
    AMPLITUDE_PRIORITY = [
        'F-obs', 'FOBS', 'FP', 'F',  # Direct observations
        'F-obs-filtered', 'FOBS-filtered',  # Filtered observations
        'F(+)', 'FPLUS',  # Anomalous pairs
        'FMEAN', 'F-pk', 'F_pk',  # Mean or peak values
        'FO', 'FODD',  # Other amplitude variants
        'F-model', 'FC', 'FCALC',  # Calculated (lowest priority)
    ]
    
    INTENSITY_PRIORITY = [
        'I-obs', 'IOBS', 'I', 'IMEAN',  # Direct observations
        'I-obs-filtered', 'IOBS-filtered',  # Filtered observations
        'I(+)', 'IPLUS', 'IP',  # Anomalous pairs
        'I-pk', 'I_pk',  # Peak values
        'IHLI', 'I_full', 'IOBS_full', 'IO',  # Other intensity variants
    ]

    RFREE_FLAG_NAMES = [
        'R-free-flags', 'RFREE', 'FreeR_flag', 'FREE',  # Common names
        'R-free', 'Rfree', 'FREER', 'FREE_FLAG',  # Variants
        'test', 'TEST', 'free', 'Free',  # Generic names
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
    
    def read(self, filepath: str) -> 'MTZ':
        """
        Read an MTZ file and extract reflection data.

        Parameters
        ----------
        filepath : str
            Path to the MTZ file.

        Returns
        -------
        MTZ
            Self, for method chaining.
        """
        self.data = dict()
        if self.verbose > 1:
            print(f"Reading MTZ file: {filepath}")
        self.mtz_data = rs.read_mtz(filepath)
        self.cell = np.array([self.mtz_data.cell.a, self.mtz_data.cell.b, self.mtz_data.cell.c,
                             self.mtz_data.cell.alpha, self.mtz_data.cell.beta, self.mtz_data.cell.gamma])
        hkl = self.mtz_data.reset_index()[['H', 'K', 'L']].to_numpy().astype(np.int32)
        self.data['HKL'] = hkl
        self.spacegroup = self.mtz_data.spacegroup.short_name()
        self._extract_amplitudes_and_intensities()
        self._extract_rfree_flags()
        return self
    
    def __call__(self) -> Tuple[dict, np.ndarray, str]:
        """
        Return extracted data in a standardized format.

        Returns
        -------
        data : dict
            Dictionary with extracted data arrays:
            - 'HKL': Miller indices, shape (N, 3)
            - 'F', 'SIGF': Amplitudes and sigmas (if available)
            - 'I', 'SIGI': Intensities and sigmas (if available)
            - 'R-free-flags': R-free test set flags (if available)
        cell : np.ndarray
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : str
            Space group symbol.

        Raises
        ------
        ValueError
            If data has not been read yet.
        """
        try:
            return self.data, self.cell, self.spacegroup
        except Exception as e:
            print("Error returning MTZ data, try reading data first:")
            raise e
    
    def _extract_amplitudes_and_intensities(self) -> None:
        """
        Extract amplitude and intensity data with priority ordering.

        Prioritizes based on column priority lists, with intensities preferred
        over amplitudes when both are present at similar priority levels.
        Automatically identifies associated sigma columns.

        Sets
        ----
        self.data['I'] : np.ndarray
            Intensity values (if available).
        self.data['SIGI'] : np.ndarray
            Intensity uncertainties (if available).
        self.data['F'] : np.ndarray
            Amplitude values (if available).
        self.data['SIGF'] : np.ndarray
            Amplitude uncertainties (if available).
        """
        available_cols = set(self.mtz_data.columns)
        
        # Find the highest priority intensity column
        intensity_col = None
        intensity_sigma_col = None

        intensity_priority_idx = None
        for idx, col in enumerate(self.INTENSITY_PRIORITY):
            if col in available_cols:
                dtype = str(self.mtz_data.dtypes[col])
                # Check if this is actually intensity data using reciprocalspaceship dtype
                if 'Intensity' in dtype or 'J' in dtype:
                    intensity_col = col
                    intensity_priority_idx = idx
                    break
        
        # Find the highest priority amplitude column
        amplitude_col = None
        amplitude_sigma_col = None
        amplitude_priority_idx = None
        for idx, col in enumerate(self.AMPLITUDE_PRIORITY):
            if col in available_cols:
                dtype = str(self.mtz_data.dtypes[col])
                # Check if this is amplitude data using reciprocalspaceship dtype
                if 'SFAmplitude' in dtype or 'F' in dtype:
                    amplitude_col = col
                    amplitude_priority_idx = idx
                    break

        if intensity_col:
            intensity_sigma_col = self._find_sigma_column(self.mtz_data, intensity_col, is_intensity=True)
        if amplitude_col:
            amplitude_sigma_col = self._find_sigma_column(self.mtz_data, amplitude_col, is_intensity=False)

        # put all the selected data cleanly into self.data
        if intensity_col:
            self.data['I'] = self.mtz_data[intensity_col].to_numpy().astype(np.float32)
            self.data['I_col'] = intensity_col
        if intensity_sigma_col:
            self.data['SIGI'] = self.mtz_data[intensity_sigma_col].to_numpy().astype(np.float32)
            self.data['SIGI_col'] = intensity_sigma_col
        if amplitude_col:
            self.data['F'] = self.mtz_data[amplitude_col].to_numpy().astype(np.float32)
            self.data['F_col'] = amplitude_col
        if amplitude_sigma_col:
            self.data['SIGF'] = self.mtz_data[amplitude_sigma_col].to_numpy().astype(np.float32)
            self.data['SIGF_col'] = amplitude_sigma_col
        
    def _extract_rfree_flags(self) -> None:
        """
        Extract R-free flags from the dataset.

        R-free flags use the convention:
        - 0 = test set (free reflections, not used in refinement)
        - 1+ = work set (used in refinement)

        Some programs may use inverted conventions, which are automatically
        detected and corrected.

        Sets
        ----
        self.data['R-free-flags'] : np.ndarray
            Boolean array where True indicates work set reflections.
        self.data['R-free-source'] : str
            Name of the source column.
        """

        dataset = self.mtz_data
        available_cols = set(dataset.columns)
        
        for col in self.RFREE_FLAG_NAMES:
            if col in available_cols:
                # Get the data type from reciprocalspaceship
                dtype = str(dataset.dtypes[col])
                
                # Check if this looks like flag data (usually integer types)
                if 'int' in dtype.lower() or 'flag' in dtype.lower() or 'I' in dtype:
                    try:
                        flags = dataset[col].to_numpy()
                        
                        # Handle NaN values or object types
                        if flags.dtype == object or not np.issubdtype(flags.dtype, np.integer):
                            # Try to convert to integer, replacing NaN with -1
                            flags = pd.to_numeric(flags, errors='coerce')
                            flags = np.nan_to_num(flags, nan=-1).astype(np.int32)
                        else:
                            flags = flags.astype(np.int32)
                        
                        rfree_flags = np.array(flags, dtype=np.int32)
                        rfree_source = col
                        
                        # Get unique flag values
                        unique_flags = np.unique(rfree_flags).tolist()
                        n_free = (rfree_flags == 0).sum().item()
                        n_work = (rfree_flags != 0).sum().item()
                        free_pct = 100.0 * n_free / len(rfree_flags) if len(rfree_flags) > 0 else 0
                        
                        # Check if convention is flipped (more "free" than "work")
                        # Standard convention: 0=free (test set, ~5-10%), other=work (~90-95%)
                        # If free > 50%, the convention is likely inverted
                        if free_pct > 50.0:
                            # Flip the convention: 0 becomes 1, non-zero becomes 0
                            # But preserve -1 for missing/NA values
                            flipped = np.zeros_like(rfree_flags)
                            flipped[rfree_flags == 0] = 1  # Old free (0) becomes work (1)
                            flipped[rfree_flags > 0] = 0   # Old work (>0) becomes free (0)
                            flipped[rfree_flags < 0] = -1  # Preserve NA markers
                            
                            rfree_flags = flipped
                            # Recalculate statistics
                            n_free = (rfree_flags == 0).sum().item()
                            n_work = (rfree_flags != 0).sum().item()
                            free_pct = 100.0 * n_free / len(rfree_flags) if len(rfree_flags) > 0 else 0
                            unique_flags = np.unique(rfree_flags).tolist()
                            
                            
                            if self.verbose > 0: print(f"   After flip: free={n_free} ({free_pct:.1f}%), work={n_work} ({100-free_pct:.1f}%)")
                        if self.verbose > 1:
                            print(f"Found R-free flags: {col}")
                            print(f"  Unique flag values: {unique_flags}")
                            print(f"  Convention: 0=test(free), other=work")
                            print(f"  Test set: {n_free} reflections ({free_pct:.1f}%)")
                            print(f"  Work set: {n_work} reflections ({100-free_pct:.1f}%)")
                        
                        # Warn if free set is still unusually large or small after flipping
                        if free_pct > 10 or free_pct < 1:
                            print(f"  ⚠️  WARNING: Free set percentage ({free_pct:.1f}%) is unusual (typical: 2-5%)")
                            print(f"     This may indicate incomplete flags or non-standard partitioning.")
                        
                        self.data['R-free-flags'] = rfree_flags.astype(bool)
                        self.data['R-free-source'] = rfree_source
                        
                        return self.data
                    except Exception as e:
                        print(f"Warning: Could not load R-free flags from {col}: {e}")
                        return None, None
        return None, None
        
    def _find_sigma_column(self, dataset: rs.DataSet, data_col: str, 
                           is_intensity: bool) -> Optional[str]:
        """
        Find the sigma (uncertainty) column corresponding to a data column.

        Parameters
        ----------
        dataset : rs.DataSet
            Reciprocalspaceship dataset.
        data_col : str
            Name of the data column (e.g., 'IOBS', 'F-obs').
        is_intensity : bool
            Whether the data column is intensity (True) or amplitude (False).

        Returns
        -------
        str or None
            Name of sigma column if found, None otherwise.
        """
        available_cols = set(dataset.columns)
        
        # Build list of possible sigma column names
        sigma_variants = []
        
        # Common patterns
        sigma_variants.extend([
            f'SIG{data_col}',           # SIGIOBS
            f'SIGM{data_col}',          # SIGMIOBS (gemmi style)
            f'{data_col}_sigma',        # IOBS_sigma
            f'{data_col}-sigma',        # IOBS-sigma
        ])
        
        # Pattern replacements
        if is_intensity:
            # For intensities: I-obs → SIGI-obs, IOBS → SIGIOBS, etc.
            sigma_variants.extend([
                data_col.replace('I', 'SIGI', 1),
                data_col.replace('I-', 'SIGI-'),
                data_col.replace('IOBS', 'SIGIOBS'),
                data_col.replace('IMEAN', 'SIGIMEAN'),
            ])
            # Generic intensity sigmas
            sigma_variants.extend(['SIGI', 'SIGIMEAN', 'SIGI-obs', 'SIGIOBS'])
        else:
            # For amplitudes: F-obs → SIGF-obs, FOBS → SIGFOBS, etc.
            sigma_variants.extend([
                data_col.replace('F', 'SIGF', 1),
                data_col.replace('F-', 'SIGF-'),
                data_col.replace('FOBS', 'SIGFOBS'),
                data_col.replace('FP', 'SIGFP'),
            ])
            # Generic amplitude sigmas
            sigma_variants.extend(['SIGF', 'SIGFOBS', 'SIGF-obs', 'SIGFP'])
        
        # Check each variant in order
        for sigma_col in sigma_variants:
            if sigma_col in available_cols:
                # Verify it's actually a standard deviation dtype
                dtype = str(dataset.dtypes[sigma_col])
                if 'Stddev' in dtype or 'Sigma' in dtype or 'SIG' in sigma_col.upper():
                    return sigma_col
        
        return None

def find_header_length_pdb_file(file: str, max_header_length: int = 100000) -> int:
    """
    Find the number of header lines in a PDB file.

    Scans the file line by line until an ATOM or HETATM record is found.

    Parameters
    ----------
    file : str
        Path to the PDB file.
    max_header_length : int, optional
        Maximum number of header lines to scan. Default is 100000.

    Returns
    -------
    int
        Number of header lines before the first ATOM/HETATM record.

    Raises
    ------
    ValueError
        If header length exceeds max_header_length.
    """
    skipheader = 0
    with open(file,'r') as f:
        for line in f:
            if 'ATOM' in line[0:4] or 'HETATM' in line[0:6]:
                break
            skipheader+=1
            if skipheader > max_header_length:
                raise ValueError('Header length is too long, check file')
    return skipheader

def load_pdb_as_pd(file: str, skipheader: int = 0, skipfooter: int = 1) -> pd.DataFrame:
    """
    Load a PDB file into a pandas DataFrame.

    Parses ATOM, HETATM, and ANISOU records from a PDB file and returns
    a structured DataFrame with all atomic properties.

    Parameters
    ----------
    file : str
        Path to the PDB file.
    skipheader : int, optional
        Number of header lines to skip. If 0, automatically detected.
    skipfooter : int, optional
        Number of footer lines to skip. Default is 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ATOM, serial, name, altloc, resname, chainid,
        resseq, icode, x, y, z, occupancy, tempfactor, element, charge,
        anisou_flag, u11, u22, u33, u12, u13, u23, index.
        DataFrame attributes include 'cell', 'spacegroup', and 'z'.
    """
    if skipheader == 0:
        skipheader = find_header_length_pdb_file(file)
    colspecs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
                    (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78),
                    (78, 80)]
    names = ['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 'resseq','icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 'element', 'charge']

    # colspecs_dtype = {'ATOM':str, 'serial':np.float32, 'name':str, 'altloc':str, 'resname':str, 'chainid':str, 'resseq':np.float32,'icode':str, 'x':np.float32, 'y':np.float32, 'z':np.float32, 'occupancy':np.float32, 'tempfactor':np.float32, 'element':str, 'charge':str}
    # colspecs_dtype = {'ATOM':str, 'serial':str, 'name':str, 'altloc':str, 'resname':str, 'chainid':str, 'resseq':str,'icode':str, 'x':str, 'y':str, 'z':str, 'occupancy':str, 'tempfactor':str, 'element':str, 'charge':str}

    pdb = pd.read_fwf(file, names=names, colspecs=colspecs, skiprows=skipheader, skipfooter=skipfooter, keep_default_na=False, na_values=[''])
    pdb['anisou_flag'] = False
    names = [
        "ATOM", "serial", "name", 
        "altloc", "resname", "chainid", "resseq", 
        "u11", "u22", "u33", "u12", "u13", "u23", "element"
    ]
    colspecs = [
        (0, 6),   # Record name
        (6, 11),  # Atom serial number
        (12, 16), # Atom name
        (16, 17), # Alternate location indicator
        (17, 20), # Residue name
        (21, 22), # Chain identifier
        (22, 26), # Residue sequence number
        (29, 35), # U(1,1)
        (36, 42), # U(2,2)
        (43, 49), # U(3,3)
        (50, 56), # U(1,2)
        (57, 63), # U(1,3)
        (63, 70), # U(2,3)
        (76, 78), # Element symbol
    ]
    anisou = pd.read_fwf(file, names=names, colspecs=colspecs, skiprows=skipheader, skipfooter=skipfooter, keep_default_na=False, na_values=[''])
    anisou = anisou.loc[anisou['ATOM']=="ANISOU"]
    pdb = pdb.loc[(pdb['ATOM']=="ATOM") | (pdb['ATOM']=="HETATM")]
    anisou.drop(columns=['ATOM'],inplace=True)
    pdb = pdb.merge(anisou, on=["serial", "name", "altloc", "resname", "chainid", "resseq", "element"], how="left")
    pdb.loc[pdb['u11'].notnull(),'anisou_flag'] = True
    pdb[['u11','u22','u33','u12','u13','u23']] = pdb[['u11','u22','u33','u12','u13','u23']].astype(float) / 1e4
    pdb[['serial','resseq']] = pdb[['serial','resseq']].astype(int)
    pdb[['x','y','z','occupancy','tempfactor']] = pdb[['x','y','z','occupancy','tempfactor']].astype(float)
    pdb[['altloc','icode']] = pdb[['altloc','icode']].fillna('')
    pdb['charge'] = pdb['charge'].astype(str).str.strip('+').str.replace('1-', '-1').str.replace('2-', '-2').astype(float).fillna(0).astype(int)
    # Format element type: strip whitespace and capitalize only first letter
    pdb['element'] = pdb['element'].astype(str).str.strip().str.capitalize()
    pdb['index'] = np.arange(pdb.shape[0]).astype(int)
    try: cell,spacegroup,z = read_crystallographic_info(file) 
    except: cell,spacegroup,z = None,None,None
    pdb.attrs['cell'] = cell
    pdb.attrs['spacegroup'] = spacegroup
    pdb.attrs['z'] = z

    return pdb

def read_crystallographic_info(file: str) -> Tuple[Optional[List[float]], Optional[str], Optional[str]]:
    """
    Extract crystallographic information from a PDB file.

    Reads the CRYST1 record to obtain unit cell parameters and space group.

    Parameters
    ----------
    file : str
        Path to the PDB file.

    Returns
    -------
    cell : list of float or None
        Unit cell parameters [a, b, c, alpha, beta, gamma] in Å and degrees.
    spacegroup : str or None
        Space group symbol.
    z : str or None
        Number of molecules per unit cell.
    """
    with open(file,'r') as f:
        for line in f:
            if 'CRYST1' in line:
                a = line[6:15]
                b = line[15:24]
                c = line[24:33]
                alpha = line[33:40]
                beta = line[40:47]
                gamma = line[47:54]
                spacegroup = line[55:68].strip()
                z = line[68:].strip()
                cell = [float(a),float(b),float(c),float(alpha),float(beta),float(gamma)]
                return cell,spacegroup,z
    return None,None,None

class PDB:
    """
    Reader for PDB files containing atomic coordinate data.

    This class reads PDB files and extracts atomic coordinates, properties,
    and crystallographic metadata.

    Attributes
    ----------
    verbose : int
        Verbosity level for logging.
    dataframe : pd.DataFrame
        DataFrame containing atomic data.
    cell : list or None
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str or None
        Space group symbol.

    Examples
    --------
    >>> pdb = PDB(verbose=1).read('structure.pdb')
    >>> df, cell, spacegroup = pdb()
    >>> print(f"Loaded {len(df)} atoms")
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize PDB reader.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level (0=silent, 1=normal, 2=debug). Default is 0.
        """
        self.verbose = verbose
    
    def read(self, filepath: str) -> 'PDB':
        """
        Read a PDB file and extract atomic data.

        Parameters
        ----------
        filepath : str
            Path to the PDB file.

        Returns
        -------
        PDB
            Self, for method chaining.
        """
        self.dataframe = load_pdb_as_pd(filepath)
        self.cell, self.spacegroup, self.z = read_crystallographic_info(filepath)
        return self

    def __call__(self) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[str]]:
        """
        Return extracted data in a standardized format.

        Returns
        -------
        dataframe : pd.DataFrame
            DataFrame with atomic data including coordinates, B-factors,
            occupancies, and anisotropic U parameters.
        cell : np.ndarray or None
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        spacegroup : str or None
            Space group symbol.

        Raises
        ------
        ValueError
            If data has not been read yet.
        """
        try:
            return self.dataframe, self.cell, self.spacegroup
        except Exception as e:
            print("Error returning PDB data, try reading data first:")
            raise e

    
    

        


