#!/usr/bin/env python
"""
Generate pre-computed ITC92 scattering factor table.

This script creates a .pt file containing scattering factor parameters
for all elements (Z=1 to 103) and available ions. The generated file
removes the need for runtime gemmi dependency for scattering parameter lookup.

Usage:
    python -m torchref.scripts.generate_scattering_table

Output:
    torchref/data/itc92_scattering_factors.pt
"""

import os
from pathlib import Path

import pandas as pd
import torch

# gemmi is only required for generation, not at runtime
import gemmi


def get_element_to_z_from_csv(csv_path: str) -> dict:
    """
    Build element-to-Z mapping from atomic_vdw_radii.csv.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with element data.

    Returns
    -------
    dict
        Mapping of element symbol to atomic number.
    """
    df = pd.read_csv(csv_path, comment="#")
    element_to_z = {}
    for _, row in df.iterrows():
        elem = row["element"].strip()
        z = int(row["Atomic_Number"])
        element_to_z[elem] = z
    return element_to_z


def get_itc92_params(element: str, charge: int = 0):
    """
    Get ITC92 scattering parameters for an element with optional charge.

    Parameters
    ----------
    element : str
        Element symbol (e.g., 'C', 'Fe').
    charge : int, optional
        Ionic charge. Default is 0.

    Returns
    -------
    tuple or None
        (A, B) tensors of shape (5,) each, or None if not available.
    """
    try:
        sf = gemmi.IT92_get_exact(gemmi.Element(element), charge)
        # IT92_get_exact returns None if entry not found
        if sf is None:
            return None
        # ITC92 has 4 Gaussians + constant: a1-a4, b1-b4, c
        # We store as 5 values: [a1, a2, a3, a4, c] and [b1, b2, b3, b4, 0]
        A = torch.tensor(list(sf.a) + [sf.c], dtype=torch.float32)
        B = torch.tensor(list(sf.b) + [0.0], dtype=torch.float32)
        return A, B
    except Exception:
        return None


def generate_scattering_table(output_path: str, csv_path: str, verbose: bool = True):
    """
    Generate the complete scattering factor table.

    Parameters
    ----------
    output_path : str
        Path to save the .pt file.
    csv_path : str
        Path to atomic_vdw_radii.csv for element-to-Z mapping.
    verbose : bool, optional
        Print progress information. Default is True.
    """
    if verbose:
        print("Generating ITC92 scattering factor table...")

    # Build element-to-Z mapping from CSV
    element_to_z = get_element_to_z_from_csv(csv_path)
    z_to_element = {v: k for k, v in element_to_z.items()}

    # Find max Z (should be 103 for Lr)
    max_z = max(element_to_z.values())
    if verbose:
        print(f"  Found {len(element_to_z)} elements (Z=1 to {max_z})")

    # Initialize tensors for neutral atoms: shape (max_z + 1, 5)
    # Index 0 is unused, indices 1-max_z correspond to elements
    A_neutral = torch.zeros(max_z + 1, 5, dtype=torch.float32)
    B_neutral = torch.zeros(max_z + 1, 5, dtype=torch.float32)

    # Track which elements we successfully got parameters for
    elements_found = []
    elements_missing = []

    # Get neutral atom parameters for all elements
    for elem, z in element_to_z.items():
        params = get_itc92_params(elem, charge=0)
        if params is not None:
            A_neutral[z] = params[0]
            B_neutral[z] = params[1]
            elements_found.append(elem)
        else:
            elements_missing.append(elem)

    if verbose:
        print(f"  Neutral atoms: {len(elements_found)} found, {len(elements_missing)} missing")
        if elements_missing:
            print(f"    Missing: {elements_missing}")

    # Build ions dictionary
    # Common charge states to check for each element
    # The actual availability depends on the ITC92 tables
    ions = {}
    ion_count = 0

    # Check a wide range of charge states for each element
    charge_range = range(-4, 9)  # -4 to +8 covers most cases

    for elem, z in element_to_z.items():
        for charge in charge_range:
            if charge == 0:
                continue  # Skip neutral, already handled

            params = get_itc92_params(elem, charge)
            if params is not None:
                # Key format: "Element+/-charge" e.g., "Fe2+", "O2-"
                if charge > 0:
                    key = f"{elem}{charge}+"
                else:
                    key = f"{elem}{abs(charge)}-"

                ions[key] = (params[0], params[1])
                ion_count += 1

    if verbose:
        print(f"  Ions: {ion_count} charge states found")
        if ion_count > 0:
            # Show some examples
            examples = list(ions.keys())[:10]
            print(f"    Examples: {examples}")

    # Build the complete table dictionary
    table = {
        "A": A_neutral,
        "B": B_neutral,
        "element_to_z": element_to_z,
        "z_to_element": z_to_element,
        "ions": ions,
        "metadata": {
            "source": "ITC92 via gemmi",
            "version": "1.0",
            "max_z": max_z,
            "n_neutral": len(elements_found),
            "n_ions": ion_count,
        },
    }

    # Save the table
    torch.save(table, output_path)

    if verbose:
        file_size = os.path.getsize(output_path) / 1024
        print(f"  Saved to: {output_path}")
        print(f"  File size: {file_size:.1f} KB")

    return table


def main():
    """Main entry point for the script."""
    # Determine paths
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent
    data_dir = package_dir / "data"

    csv_path = data_dir / "atomic_vdw_radii.csv"
    output_path = data_dir / "itc92_scattering_factors.pt"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    generate_scattering_table(str(output_path), str(csv_path), verbose=True)
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
