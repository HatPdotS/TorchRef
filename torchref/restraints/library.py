"""
Monomer Library Manager for TorchRef.

Manages access to the CCP4 Monomer Library restraint dictionaries with a
priority-based resolution strategy. Standard amino acids and nucleotides
are bundled as package data; non-standard residues are downloaded on demand
from the MonomerLibrary GitHub repository and cached locally.

The monomer library provides ideal geometry parameters (bond lengths, angles,
torsions, planes, chirals) derived from the Cambridge Structural Database.

References
----------
Long, F., et al. (2017). AceDRG: a stereochemical description generator
    for ligands. Acta Cryst. D73, 112-122.

"""

import os
import warnings
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

from torchref import ROOT_TORCHREF

# Pinned commit for reproducibility
_MONOMER_LIB_COMMIT = "713a04911"
_MONOMER_LIB_RAW_URL = (
    f"https://raw.githubusercontent.com/MonomerLibrary/monomers/"
    f"{_MONOMER_LIB_COMMIT}"
)

# Bundled package data location
_BUNDLED_PATH = Path(__file__).parent.parent / "data" / "monomer_library"

# Legacy external monomer library path
_LEGACY_PATH = ROOT_TORCHREF / "external_monomer_library"

# User cache directory
_CACHE_DIR = Path.home() / ".cache" / "torchref" / "monomer_library"


class MonomerLibraryManager:
    """
    Manages access to the CCP4 Monomer Library with priority-based resolution.

    Resolution priority for individual CIF files:
    1. ``TORCHREF_MONOMER_LIB`` environment variable (local library install)
    2. Bundled package data (standard amino acids, nucleotides)
    3. User cache (``~/.cache/torchref/monomer_library/``)
    4. Legacy ``external_monomer_library/`` directory
    5. On-demand download from GitHub (cached for future use)

    Parameters
    ----------
    verbose : int, optional
        Verbosity level. 0 = silent, 1 = warnings, 2 = info. Default 1.
    """

    def __init__(self, verbose=1):
        self.verbose = verbose
        self._env_path = self._resolve_env_path()

    @staticmethod
    def _resolve_env_path():
        """Check for TORCHREF_MONOMER_LIB environment variable."""
        env_val = os.environ.get("TORCHREF_MONOMER_LIB")
        if env_val:
            p = Path(env_val)
            if p.is_dir():
                return p
            warnings.warn(
                f"TORCHREF_MONOMER_LIB points to non-existent directory: {env_val}"
            )
        return None

    def get_cif_file(self, resname):
        """
        Resolve the CIF file path for a given residue name.

        Parameters
        ----------
        resname : str
            Residue name (e.g., 'ALA', 'GLY', 'ATP').

        Returns
        -------
        Path or None
            Path to the CIF file, or None if not found anywhere.
        """
        first_char = resname[0].lower()
        relative = Path(first_char) / f"{resname}.cif"
        relative_upper = Path(first_char) / f"{resname.upper()}.cif"

        # 1. Environment variable override
        if self._env_path:
            for rel in (relative, relative_upper):
                p = self._env_path / rel
                if p.exists():
                    return p

        # 2. Bundled package data
        for rel in (relative, relative_upper):
            p = _BUNDLED_PATH / rel
            if p.exists():
                return p

        # 3. User cache
        for rel in (relative, relative_upper):
            p = _CACHE_DIR / rel
            if p.exists():
                return p

        # 4. Legacy external_monomer_library
        for rel in (relative, relative_upper):
            p = _LEGACY_PATH / rel
            if p.exists():
                return p

        # 5. On-demand download
        return self._download_cif(resname)

    def get_link_definitions_path(self):
        """
        Resolve the path to mon_lib_list.cif (inter-residue link definitions).

        Returns
        -------
        Path
            Path to mon_lib_list.cif.

        Raises
        ------
        FileNotFoundError
            If the file cannot be found or downloaded.
        """
        relative = Path("list") / "mon_lib_list.cif"

        # 1. Environment variable override
        if self._env_path:
            p = self._env_path / relative
            if p.exists():
                return p

        # 2. Bundled package data
        p = _BUNDLED_PATH / relative
        if p.exists():
            return p

        # 3. User cache
        p = _CACHE_DIR / relative
        if p.exists():
            return p

        # 4. Legacy external_monomer_library
        p = _LEGACY_PATH / relative
        if p.exists():
            return p

        # 5. Download
        return self._download_file(
            f"{_MONOMER_LIB_RAW_URL}/list/mon_lib_list.cif",
            _CACHE_DIR / relative,
        )

    def ensure_gemmi_base(self):
        """Return a directory usable as a gemmi monomer-library root.

        ``gemmi.read_monomer_lib`` needs ``ener_lib.cif`` at the root and
        ``list/mon_lib_list.cif``. If a complete local library is configured
        (``TORCHREF_MONOMER_LIB`` / legacy install) that already has both, return
        it. Otherwise stage both into the user cache (downloading from the pinned
        mirror) and return the cache directory. Per-residue component CIFs are
        resolved separately via :meth:`get_cif_file`, so this only provides the
        global energy/link files that the bundled per-residue data lacks.

        Returns
        -------
        Path
            A directory containing ``ener_lib.cif`` and ``list/mon_lib_list.cif``.
        """
        for base in (self._env_path, _LEGACY_PATH):
            if (
                base
                and (base / "ener_lib.cif").exists()
                and (base / "list" / "mon_lib_list.cif").exists()
            ):
                return base

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ener = _CACHE_DIR / "ener_lib.cif"
        if not ener.exists():
            self._download_file(f"{_MONOMER_LIB_RAW_URL}/ener_lib.cif", ener)
        link = _CACHE_DIR / "list" / "mon_lib_list.cif"
        if not link.exists():
            self._download_file(
                f"{_MONOMER_LIB_RAW_URL}/list/mon_lib_list.cif", link
            )
        return _CACHE_DIR

    @property
    def monomer_dir(self):
        """
        Return a directory path suitable for monomer library access.

        Prefers environment variable, then bundled data, then legacy path.
        This is provided for backward compatibility with code that expects
        a directory path rather than individual file resolution.

        Returns
        -------
        Path
            Path to the monomer library root directory.
        """
        if self._env_path:
            return self._env_path
        if _BUNDLED_PATH.exists():
            return _BUNDLED_PATH
        if _LEGACY_PATH.exists():
            return _LEGACY_PATH
        return _BUNDLED_PATH  # fallback to bundled even if not fully populated

    def _download_cif(self, resname):
        """
        Download a single CIF file from the MonomerLibrary GitHub repo.

        Parameters
        ----------
        resname : str
            Residue name.

        Returns
        -------
        Path or None
            Path to downloaded file, or None if download failed.
        """
        first_char = resname[0].lower()
        url = f"{_MONOMER_LIB_RAW_URL}/{first_char}/{resname}.cif"
        dest = _CACHE_DIR / first_char / f"{resname}.cif"
        result = self._download_file(url, dest, required=False)
        if result is None:
            # Try uppercase
            url_upper = f"{_MONOMER_LIB_RAW_URL}/{first_char}/{resname.upper()}.cif"
            dest_upper = _CACHE_DIR / first_char / f"{resname.upper()}.cif"
            result = self._download_file(url_upper, dest_upper, required=False)
        return result

    def _download_file(self, url, dest, required=True):
        """
        Download a file from a URL and save to dest.

        Parameters
        ----------
        url : str
            URL to download from.
        dest : Path
            Local destination path.
        required : bool
            If True, raise FileNotFoundError on failure.

        Returns
        -------
        Path or None
            Path to the downloaded file, or None if failed and not required.
        """
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if self.verbose >= 2:
                print(f"Downloading restraint dictionary: {url}")
            with urlopen(url, timeout=30) as response:
                data = response.read()
            if response.status != 200:
                raise URLError(f"HTTP {response.status}")
            dest.write_bytes(data)
            if self.verbose >= 2:
                print(f"  Cached to: {dest}")
            return dest
        except Exception as e:
            if required:
                raise FileNotFoundError(
                    f"Could not download monomer library file from {url}: {e}\n"
                    f"You can set TORCHREF_MONOMER_LIB to point to a local "
                    f"CCP4 monomer library installation."
                ) from e
            if self.verbose >= 1:
                warnings.warn(
                    f"Could not download restraint dictionary for residue "
                    f"from {url}: {e}"
                )
            return None


# Module-level singleton (lazily created)
_manager = None


def get_library_manager(verbose=1):
    """
    Get the global MonomerLibraryManager singleton.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (only used on first call).

    Returns
    -------
    MonomerLibraryManager
    """
    global _manager
    if _manager is None:
        _manager = MonomerLibraryManager(verbose=verbose)
    return _manager
