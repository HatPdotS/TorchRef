"""Monomer templates and the link modifications that patch them.

The monomer library describes each residue in its **free** form. Forming a peptide bond
applies the modifications the ``chem_link`` table names -- ``DEL-OXT`` to the residue
donating its C, ``DEL-HN1`` (``DEL-HNP`` for proline) to the residue donating its N --
which delete the restraints the link makes meaningless and overwrite the targets that
change. A residue therefore draws its restraints from a *patched* template, identified
by a key such as ``'ALA:DEL-HN1+DEL-OXT'``.

Chain termini are deliberately left unpatched: a real C-terminus keeps its ``OXT`` and
carboxylate geometry, a real N-terminus its ammonium.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from torchref.restraints.modifications import (
    apply_modifications,
    link_modifications,
    read_mod_definitions,
)


def resolve_template_keys(
    resnames: Sequence[str],
    peptide_pairs: Sequence[Tuple[int, int]],
    cif_dict: Dict,
    link_list,
    verbose: int = 0,
) -> Tuple[Dict, np.ndarray]:
    """Assign each residue the template it should draw restraints from.

    Parameters
    ----------
    resnames : sequence of str
        Residue name per residue index.
    peptide_pairs : sequence of tuple of int
        ``(donates C, donates N)`` residue index pairs.
    cif_dict : dict
        Restraint dictionary keyed by residue name.
    link_list : pandas.DataFrame or None
        Link-type definitions, as
        :func:`~torchref.restraints.restraints_helper.read_link_definitions` returns
        them. None disables patching.
    verbose : int, default 0
        Verbosity level.

    Returns
    -------
    comp_dict : dict
        ``cif_dict`` plus one entry per patched ``(residue type, modification set)`` in
        use. ``cif_dict`` itself is left keyed by residue name alone.
    template_key : numpy.ndarray
        Key per residue, shape ``(R,)``. Residues that are not patched carry their own
        residue name.
    """
    comp_dict = dict(cif_dict)
    keys = np.array([str(r) for r in resnames], dtype=object)

    if not cif_dict or link_list is None or len(peptide_pairs) == 0:
        return comp_dict, keys

    modifications = link_modifications(link_list)
    if "TRANS" not in modifications:
        return comp_dict, keys
    trans_mods = modifications["TRANS"]
    proline_mods = modifications.get("PTRANS", trans_mods)

    mods_by_residue: Dict[int, set] = {}
    for res_c, res_n in peptide_pairs:
        donor_mod, acceptor_mod = (
            proline_mods if str(resnames[res_n]) == "PRO" else trans_mods
        )
        for res_idx, mod_id in ((res_c, donor_mod), (res_n, acceptor_mod)):
            if mod_id is None:
                continue
            mods_by_residue.setdefault(res_idx, set()).add(mod_id)

    if not mods_by_residue:
        return comp_dict, keys

    mod_dict = read_mod_definitions()
    n_patched = 0
    for res_idx, mods in mods_by_residue.items():
        resname = str(resnames[res_idx])
        if resname not in comp_dict:
            continue
        variant = f"{resname}:{'+'.join(sorted(mods))}"
        if variant not in comp_dict:
            comp_dict[variant] = apply_modifications(
                comp_dict[resname], sorted(mods), mod_dict
            )
        keys[res_idx] = variant
        n_patched += 1

    if verbose > 1:
        print(
            f"Patched {n_patched} residues "
            f"({len(comp_dict) - len(cif_dict)} template variants)"
        )
    return comp_dict, keys


__all__ = ["resolve_template_keys"]
