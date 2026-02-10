"""
Anomalous scattering factor lookup (f' and f'').

Uses gemmi for wavelength-dependent f'/f'' values via the Cromer-Liberman
calculation. These corrections account for the dispersive (f') and
absorptive (f'') components of X-ray scattering near atomic absorption edges.

The complete scattering factor is: f(s, lambda) = f0(s) + f'(lambda) + i*f''(lambda)

where f0 is the normal (Thomson) scattering factor and f'/f'' are the
wavelength-dependent anomalous corrections.
"""

import torch
import gemmi
from typing import Dict, List, Tuple

def wavelength_to_energy_ev(wavelength: float) -> float:
    """
    Convert X-ray wavelength in Angstroms to energy in eV.

    Uses the gemmi.hc constant (hc in eV*Angstrom).

    Parameters
    ----------
    wavelength : float
        Wavelength in Angstroms

    Returns
    -------
    float
        Energy in eV
    """
    return gemmi.hc / wavelength


def get_anomalous_correction(
    element: str,
    wavelength: float,
) -> Tuple[float, float]:
    """
    Get f' and f'' for a single element at given wavelength.

    Uses the Cromer-Liberman calculation via gemmi. Returns the standard
    crystallographic f' and f'' values as used in International Tables.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Fe', 'Hg', 'Se')
    wavelength : float
        X-ray wavelength in Angstroms

    Returns
    -------
    f_prime : float
        Real anomalous correction (electrons)
    f_double_prime : float
        Imaginary anomalous correction (electrons)

    Examples
    --------
    >>> f_prime, f_double_prime = get_anomalous_correction('Se', 0.9792)
    >>> print(f"Se at Se K-edge: f'={f_prime:.2f}, f''={f_double_prime:.2f}")

    Notes
    -----
    The gemmi.cromer_liberman function expects energy in eV, not keV.
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
    Find elements with significant anomalous scattering.

    An element is considered significant if |f'| > threshold OR |f''| > threshold.
    This filtering avoids unnecessary computation for light atoms (C, N, O, H)
    which have negligible anomalous contributions at typical wavelengths.

    Parameters
    ----------
    elements : list of str
        List of unique element symbols
    wavelength : float
        X-ray wavelength in Angstroms
    threshold : float, optional
        Significance threshold in electrons (default: 0.5)

    Returns
    -------
    dict
        {element: (f_prime, f_double_prime)} for significant elements only

    Examples
    --------
    >>> elements = ['C', 'N', 'O', 'S', 'Fe', 'Zn']
    >>> significant = get_significant_elements(elements, wavelength=1.0)
    >>> print(f"Significant anomalous scatterers: {list(significant.keys())}")
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
    Get anomalous corrections and mask for atoms needing correction.

    This function creates tensors suitable for vectorized computation
    of the anomalous correction to structure factors.

    Parameters
    ----------
    element_list : list of str
        Element symbols for all atoms (length n_atoms)
    significant_elements : dict
        {element: (f_prime, f_double_prime)} from get_significant_elements()
    device : torch.device
        Device for output tensors
    dtype : torch.dtype
        Data type for output tensors

    Returns
    -------
    mask : torch.Tensor
        Boolean mask of shape (n_atoms,) - True for atoms needing correction
    f_prime : torch.Tensor
        f' values for significant atoms only (n_significant,)
    f_double_prime : torch.Tensor
        f'' values for significant atoms only (n_significant,)

    Examples
    --------
    >>> elements = ['C', 'C', 'N', 'Fe', 'O', 'Fe']
    >>> significant = {'Fe': (-1.2, 3.1)}
    >>> mask, fp, fdp = get_anomalous_corrections_by_indices(
    ...     elements, significant, torch.device('cpu'), torch.float32
    ... )
    >>> print(f"Atoms needing correction: {mask.sum().item()}")  # 2 (two Fe atoms)
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
