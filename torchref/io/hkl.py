"""Reader for CrystFEL partialator reflection lists (`.hkl`).

CrystFEL partialator output format::

    CrystFEL reflection list version 2.0
    Symmetry: 2/m_uab
       h    k    l          I    phase   sigma(I)   nmeas
       0    0    8   15631.77        -    2997.88     499
       ...

The file carries intensities, sigmas, nmeas counts (and optional phase
strings), but **not** unit-cell or space-group metadata — those
typically live alongside in a ``.cell`` file. Cell and spacegroup must
therefore be supplied by the caller.

Usage mirrors :class:`torchref.io.mtz.MTZReader`::

    reader = HKLReader(verbose=1).read("td1.hkl",
                                       cell=[a,b,c,al,be,ga],
                                       spacegroup="P 1 21 1")
    data_dict, cell, spacegroup = reader()

The returned ``data_dict`` conforms to the intensity-path contract of
:meth:`ReflectionData.load` — ``{"HKL": (N,3) int, "I": (N,) float,
"SIGI": (N,) float, "I_col": "I (CrystFEL)"}`` — so French-Wilson kicks
in automatically and amplitudes are derived downstream.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np


class HKLReader:
    """Reader for CrystFEL partialator `.hkl` reflection lists."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.data: Optional[dict] = None
        self.cell: Optional[np.ndarray] = None
        self.spacegroup: Optional[str] = None
        self.nmeas: Optional[np.ndarray] = None

    def read(
        self,
        filepath: str,
        cell: Union[list, tuple, np.ndarray, Any],
        spacegroup: Union[str, Any],
    ) -> "HKLReader":
        """Parse a CrystFEL `.hkl` file.

        Parameters
        ----------
        filepath : str
            Path to the CrystFEL reflection list.
        cell : list | tuple | np.ndarray | torchref.symmetry.Cell | torch.Tensor
            Unit cell (a, b, c, alpha, beta, gamma). If a ``Cell`` object
            is passed, ``.data`` is extracted.
        spacegroup : str | gemmi.SpaceGroup | torchref.symmetry.SpaceGroup
            Space group. If a wrapper object is passed, its ``.hm`` or
            ``.short_name()`` is used.
        """
        if self.verbose > 1:
            print(f"Reading CrystFEL hkl file: {filepath}")

        # Normalize cell → (6,) np.ndarray
        if hasattr(cell, "data"):  # torchref.symmetry.Cell
            cell = cell.data
        if hasattr(cell, "detach"):  # torch.Tensor
            cell = cell.detach().cpu().numpy()
        cell_arr = np.asarray(cell, dtype=float).reshape(-1)
        if cell_arr.size != 6:
            raise ValueError(
                f"cell must have 6 entries (a, b, c, al, be, ga); got {cell_arr.size}"
            )
        self.cell = cell_arr

        # Normalize spacegroup → HM-name string
        if isinstance(spacegroup, str):
            self.spacegroup = spacegroup
        elif hasattr(spacegroup, "hm"):  # torchref.symmetry.SpaceGroup
            self.spacegroup = spacegroup.hm
        elif hasattr(spacegroup, "short_name"):  # gemmi.SpaceGroup
            self.spacegroup = spacegroup.short_name()
        else:
            raise ValueError(
                f"Cannot normalize spacegroup of type {type(spacegroup)}"
            )

        # Parse reflection rows
        h_list, k_list, l_list, I_list, sig_list, n_list = [], [], [], [], [], []
        in_header = True
        with open(filepath) as f:
            for line in f:
                if in_header:
                    if line.strip().startswith("h "):
                        in_header = False
                    continue
                s = line.split()
                if len(s) < 7 or not s[0].lstrip("-").isdigit():
                    # Trailing comment lines or blank lines
                    continue
                h_list.append(int(s[0]))
                k_list.append(int(s[1]))
                l_list.append(int(s[2]))
                I_list.append(float(s[3]))
                sig_list.append(float(s[5]))
                n_list.append(int(s[6]))

        if not h_list:
            raise ValueError(f"No reflections parsed from {filepath}")

        hkl = np.column_stack([h_list, k_list, l_list]).astype(np.int32)
        I = np.asarray(I_list, dtype=np.float64)
        sig = np.asarray(sig_list, dtype=np.float64)
        self.nmeas = np.asarray(n_list, dtype=np.int32)

        self.data = {
            "HKL": hkl,
            "I": I,
            "SIGI": sig,
            "I_col": "I (CrystFEL)",
        }

        if self.verbose > 0:
            print(
                f"Parsed {len(hkl)} reflections from {filepath} "
                f"(cell={cell_arr.tolist()}, spacegroup='{self.spacegroup}')"
            )
        return self

    def __call__(self) -> Tuple[dict, np.ndarray, str]:
        """Return ``(data_dict, cell, spacegroup)`` for ``ReflectionData.load()``."""
        if self.data is None:
            raise RuntimeError("Call .read(path, cell, spacegroup) first.")
        return self.data, self.cell, self.spacegroup


def read(
    filepath: str,
    cell: Union[list, tuple, np.ndarray, Any],
    spacegroup: Union[str, Any],
    verbose: int = 0,
) -> HKLReader:
    """Shortcut: ``HKLReader(verbose=verbose).read(filepath, cell, spacegroup)``."""
    return HKLReader(verbose=verbose).read(filepath, cell, spacegroup)
