"""
ITC92 atomic scattering factor calculations.

Functions for computing atomic scattering factors using the
International Tables for Crystallography Volume C (1992) parameterization.
"""

import numpy as np
import pandas as pd
import torch


def get_scattering_factors_unique(atoms, s):
    """
    Compute unique scattering factors for a set of atoms.

    Parameters
    ----------
    atoms : DataFrame-like
        Atoms with 'element' and 'charge' attributes.
    s : array-like
        Scattering vector magnitudes.

    Returns
    -------
    dict
        Dictionary mapping element symbols to scattering factors.
    """
    scattering_table = pd.read_feather(
        "/das/work/p17/p17490/Peter/manual_refinement/Scattering_table_as_Excel_corrected.feather"
    )
    PSE = list(scattering_table)
    ionized_equivalents = atoms.element
    ionized_nr_equivalents = torch.tensor(
        [PSE.index(ion) for ion in ionized_equivalents]
    )
    ionized_nr_equivalents -= atoms.charge.values
    ionized_element_equivalents = [PSE[ion] for ion in ionized_nr_equivalents]
    unique_atoms = np.unique(ionized_element_equivalents)
    scattering_angle = scattering_table.index.values
    idxs = np.digitize(s, scattering_angle)
    idxs_lower = idxs - 1
    s_lower = torch.tensor(scattering_angle[idxs_lower])
    s_higher = torch.tensor(scattering_angle[idxs])
    atom_dict = {}
    for unique_atom in unique_atoms:
        y_lower = torch.tensor(scattering_table.iloc[idxs_lower][unique_atom].values)
        y_higher = torch.tensor(scattering_table.iloc[idxs][unique_atom].values)
        scattering_factors = linear_interpolation(
            s, s_lower, s_higher, y_lower, y_higher
        )
        atom_dict[unique_atom] = scattering_factors.reshape(-1, 1)
    return atom_dict


def get_scattering_factors(scattering_dict, elements):
    """
    Get scattering factors from a pre-computed dictionary.

    Parameters
    ----------
    scattering_dict : dict
        Dictionary of scattering factors by element.
    elements : list
        List of element symbols.

    Returns
    -------
    torch.Tensor
        Concatenated scattering factors.
    """
    try:
        return torch.concatenate(
            [scattering_dict[element] for element in elements], axis=1
        )
    except KeyError as e:
        print("could not find scattering factor for all elements ")
        print("All loaded elements:", list(scattering_dict.keys()))
        print("Missing element:", e)
        raise e


def linear_interpolation(x, x0, x1, y0, y1):
    """
    Perform linear interpolation.

    Parameters
    ----------
    x : array-like
        Query points.
    x0 : array-like
        Lower bound x values.
    x1 : array-like
        Upper bound x values.
    y0 : array-like
        Lower bound y values.
    y1 : array-like
        Upper bound y values.

    Returns
    -------
    array-like
        Interpolated y values.
    """
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def get_scattering_itc92(df, s):
    """
    Get ITC92 scattering factors using gemmi.

    Parameters
    ----------
    df : DataFrame
        DataFrame with 'element' column.
    s : torch.Tensor
        Scattering vector magnitudes.

    Returns
    -------
    torch.Tensor
        Scattering factors for all atoms.
    """
    import gemmi

    all_atoms = df.element.values
    atoms = torch.unique(df.element)
    s_squared = ((s / 4) ** 2).reshape(-1, 1)
    elements = {}
    for element in atoms:
        SF = gemmi.Element(element).it92
        A = torch.tensor(SF.a).reshape(1, -1)
        B = torch.tensor(SF.b).reshape(1, -1)
        C = torch.tensor(SF.c).reshape(1, -1)
        f = torch.sum(A * np.exp(-B * s_squared), axis=1).reshape(-1, 1)
        f += C
        elements[element] = f
    return torch.concatenate([elements[element] for element in all_atoms], axis=1)


def calc_scattering_factors_paramtetrization(parametrization, s, atom_list):
    """
    Calculate scattering factors from ITC92 parametrization.

    Parameters
    ----------
    parametrization : dict
        Dictionary of (A, B, C) tuples by element.
    s : torch.Tensor
        Scattering vector magnitudes.
    atom_list : list
        List of atom symbols.

    Returns
    -------
    torch.Tensor
        Scattering factors.
    """
    scattering_factors = []
    for atom in atom_list:
        A, B, C = parametrization[atom]
        f = torch.sum(A * torch.exp(-B * s.reshape(-1, 1)), axis=1).reshape(-1, 1)
        f += C
        scattering_factors.append(f)
    return torch.concatenate(scattering_factors, axis=1)


def get_parameterization(df):
    """
    Get ITC92 parametrization for atoms in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame with 'element' and 'charge' columns.

    Returns
    -------
    dict
        Dictionary of parametrization by element.
    """
    charge_elements = []
    for i, df_group in df.groupby(["element", "charge"]):
        charge_elements.append(i)
    print("charge_elements", charge_elements)
    atoms_dict = {}
    for atom, charge in charge_elements:
        atoms_dict[str(atom)] = get_parametrization_atom(charge, atom)
    print("atoms_dict", atoms_dict)
    return atoms_dict


def get_parameterization_extended(df):
    """
    Extended parametrization function that handles all atoms in a DataFrame.

    Creates a dictionary mapping element symbols (and optionally charges) to
    their ITC92 parameters (A, B, C). This is optimized for FT-based calculations
    where we need fast access to parametrization without scattering vectors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'element' and 'charge' columns

    Returns
    -------
    dict : {element_str: (A, B, C)}
        A: torch.Tensor, shape (1, 4) - amplitude coefficients
        B: torch.Tensor, shape (1, 4) - width coefficients (Å²)
        C: torch.Tensor, shape (1,) - constant term
    """
    # Get unique element/charge combinations
    if "charge" in df.columns:
        charge_elements = list(df.groupby(["element", "charge"]).groups.keys())
    else:
        charge_elements = [(elem, 0) for elem in df["element"].unique()]

    atoms_dict = {}
    for atom, charge in charge_elements:
        key = str(atom) if charge == 0 else f"{atom}{charge:+d}"
        params = get_parametrization_atom(charge, atom)
        atoms_dict[key] = params

        # Also add without charge suffix for easy access
        if atom not in atoms_dict:
            atoms_dict[str(atom)] = params

    return atoms_dict


def get_parametrization_for_elements(elements, charges=None):
    """
    Get ITC92 parametrization for a list of elements.

    Useful for getting parametrization for specific atoms without a full DataFrame.

    Parameters
    ----------
    elements : list of str
        Element symbols (e.g., ['C', 'N', 'O'])
    charges : list of int, optional
        Charges for each element (default: all zeros)

    Returns
    -------
    dict : {element: (A, B, C)}
    """
    if charges is None:
        charges = [0] * len(elements)

    if len(charges) != len(elements):
        raise ValueError("Length of charges must match length of elements")

    atoms_dict = {}
    for elem, charge in zip(elements, charges):
        key = str(elem)
        atoms_dict[key] = get_parametrization_atom(charge, elem)

    return atoms_dict


def get_parametrization_atom(charge, atom):
    """
    Get ITC92 parametrization for a single atom.

    Parameters
    ----------
    charge : int
        Atomic charge.
    atom : str
        Element symbol.

    Returns
    -------
    list
        [A, B] tensors for the atom.
    """
    import gemmi

    try:
        SF = gemmi.IT92_get_exact(gemmi.Element(atom), charge)
        A = torch.tensor(SF.a, dtype=torch.float32)
        B = torch.tensor(SF.b, dtype=torch.float32)
        C = torch.tensor([SF.c], dtype=torch.float32)
        A = torch.cat([A, C]).reshape(1, -1)
        B = torch.cat([B, torch.tensor([0], dtype=torch.float32)]).reshape(1, -1)
        parametrization = [A, B]
        return parametrization
    except Exception as e:
        print(
            "Could not find scattering factor for",
            atom,
            charge,
            "Exception that was raised:",
            e,
        )
        if charge != 0:
            print("Try without charge")
            return get_parametrization_atom(0, atom)
        else:
            print(
                "could not find scattering factor for neutral atom either, setting to zero"
            )
            return [
                torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32),
                torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32),
            ]
