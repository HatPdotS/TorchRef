"""Monomer Library Manager: CCP4 restraint dictionaries with priority resolution.

Supplies the ideal geometry (bond lengths, angles, torsions, planes, chirals)
used to build restraints. Standard amino acids and nucleotides are bundled as
package data; anything else is fetched on demand from a pinned commit of the
MonomerLibrary GitHub repository and cached under ``~/.cache/torchref`` -- so
first use of a novel ligand needs network access unless
``TORCHREF_MONOMER_LIB`` points at a local install.

Reference: Long, F., et al. (2017). AceDRG. Acta Cryst. D73, 112-122.
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

    Note that this 5-step chain applies to per-residue CIF resolution via
    :meth:`get_cif_file`. The :meth:`ensure_gemmi_base` and :meth:`monomer_dir`
    accessors use reduced chains (see their own docstrings).

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
        """Path to mon_lib_list.cif (inter-residue link definitions).

        Same five-step chain as :meth:`get_cif_file`; raises ``FileNotFoundError``
        rather than returning None if even the download fails.
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
        """A directory holding ``ener_lib.cif`` and ``list/mon_lib_list.cif``.

        What ``gemmi.read_monomer_lib`` requires at its root. Returns a configured
        local library if it already has both, else stages them into the user cache
        (downloading if needed) and returns that. Provides only these global files
        -- per-residue CIFs still resolve through :meth:`get_cif_file`.
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
        """Monomer library root: env var, then bundled data, then legacy path.

        For callers that want a directory rather than per-file resolution. The
        bundled path is returned as a last resort even when absent or incomplete,
        so the result is **not** guaranteed usable -- check before relying on it.
        """
        if self._env_path:
            return self._env_path
        if _BUNDLED_PATH.exists():
            return _BUNDLED_PATH
        if _LEGACY_PATH.exists():
            return _LEGACY_PATH
        return _BUNDLED_PATH  # fallback to bundled even if not fully populated

    def _download_cif(self, resname):
        """Download ``resname``'s CIF into the cache; None if unavailable.

        Retries with an upper-cased filename before giving up.
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
        """Download ``url`` to ``dest`` (parents created); returns ``dest`` or None.

        With ``required=True`` a failure raises ``FileNotFoundError`` instead of
        returning None.
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
    """Get the global MonomerLibraryManager singleton.

    ``verbose`` takes effect only on the first call, which creates the instance.
    """
    global _manager
    if _manager is None:
        _manager = MonomerLibraryManager(verbose=verbose)
    return _manager
