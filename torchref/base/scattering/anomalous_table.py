"""Anomalous scattering factor lookup (f' and f'').

Wavelength-dependent f'/f'' from gemmi's Cromer-Liberman calculation: the
dispersive and absorptive components that matter near absorption edges. The
full factor is ``f(s, λ) = f0(s) + f'(λ) + i·f''(λ)``, with f0 the normal
(Thomson) term.
"""

import torch
import gemmi
from typing import Dict, List, Tuple

def wavelength_to_energy_ev(wavelength: float) -> float:
    """
    Convert X-ray wavelength (Å) to energy (eV) via the ``gemmi.hc`` constant.
    """
    return gemmi.hc / wavelength


def get_anomalous_correction(
    element: str,
    wavelength: float,
) -> Tuple[float, float]:
    """
    Get f' and f'' for one element at one wavelength (Cromer-Liberman, gemmi).

    Standard crystallographic values as in International Tables. The wavelength
    is converted to eV here because ``gemmi.cromer_liberman`` takes eV, not keV.

    Parameters
    ----------
    element : str
        Element symbol (e.g. 'Fe', 'Hg', 'Se').
    wavelength : float
        X-ray wavelength in Angstroms.

    Returns
    -------
    f_prime : float
        Real anomalous correction (electrons).
    f_double_prime : float
        Imaginary anomalous correction (electrons).
    """
    elem = gemmi.Element(element)
    z = elem.atomic_number
    energy_ev = wavelength_to_energy_ev(wavelength)
    f_prime, f_double_prime = gemmi.cromer_liberman(z, energy_ev)
    return f_prime, f_double_prime


def get_significant_elements(
    elements: List[str],
    wavelength: float,
    threshold: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Keep only the elements whose anomalous scattering is worth computing.

    Significant means ``|f'| > threshold`` or ``|f''| > threshold``, which drops
    the light atoms (H, C, N, O) that contribute nothing at usual wavelengths.

    Parameters
    ----------
    elements : list of str
        Unique element symbols.
    wavelength : float
        X-ray wavelength in Angstroms.
    threshold : float, optional
        Significance threshold in electrons. Default is 0.5.

    Returns
    -------
    dict
        ``{element: (f_prime, f_double_prime)}``, significant elements only.
    """
    significant = {}
    for elem in elements:
        f_prime, f_double_prime = get_anomalous_correction(elem, wavelength)
        if abs(f_prime) > threshold or abs(f_double_prime) > threshold:
            significant[elem] = (f_prime, f_double_prime)
    return significant


def get_anomalous_corrections_by_indices(
    element_list: List[str],
    significant_elements: Dict[str, Tuple[float, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the per-atom mask and compacted f'/f'' tensors for vectorized use.

    Parameters
    ----------
    element_list : list of str
        Element symbols for all atoms (length n_atoms).
    significant_elements : dict
        ``{element: (f_prime, f_double_prime)}`` from
        :func:`get_significant_elements`.
    device : torch.device
        Device for output tensors.
    dtype : torch.dtype
        Dtype for output tensors.

    Returns
    -------
    mask : torch.Tensor
        Bool, shape (n_atoms,); True where a correction applies.
    f_prime, f_double_prime : torch.Tensor
        Shape (n_significant,), *not* (n_atoms,) -- compacted in ``mask`` order,
        so they line up only with ``element_list[mask]``.
    """
    n_atoms = len(element_list)
    mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
    f_prime_list = []
    f_double_prime_list = []

    for i, elem in enumerate(element_list):
        if elem in significant_elements:
            mask[i] = True
            fp, fdp = significant_elements[elem]
            f_prime_list.append(fp)
            f_double_prime_list.append(fdp)

    f_prime = torch.tensor(f_prime_list, device=device, dtype=dtype)
    f_double_prime = torch.tensor(f_double_prime_list, device=device, dtype=dtype)

    return mask, f_prime, f_double_prime
