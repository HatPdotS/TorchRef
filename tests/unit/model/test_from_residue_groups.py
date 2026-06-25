"""Regression test: from_residue_groups must not merge singletons into groups.

It labeled singleton atoms with arange ids (0..n-1) and multi-atom groups with
ids 0,1,2,... in the SAME number space, then compacted with torch.unique. A
group id equal to a surviving singleton's arange id silently merged them. The
fix starts group ids at n_atoms. See TORCHREF_AUDIT.md.

Layout below reproduces the documented collision: residues (by resseq) iterate
as [0,1], [2], [3,4,5], [6], [7,8], [9]; pre-fix the [7,8] group was assigned
id 2, colliding with singleton atom index 2.
"""

import pandas as pd
import pytest
import torch

from torchref.model.parameter_wrappers import OccupancyTensor


@pytest.mark.unit
def test_from_residue_groups_no_singleton_collision():
    df = pd.DataFrame(
        {
            "index": list(range(10)),
            "resname": ["ALA"] * 10,
            "resseq": [1, 1, 2, 3, 3, 3, 4, 5, 5, 6],
            "chainid": ["A"] * 10,
            "altloc": [""] * 10,
        }
    )
    init = torch.full((10,), 0.9)

    occ = OccupancyTensor.from_residue_groups(init, df)
    g = occ.expansion_mask.cpu().tolist()  # per-atom collapsed group id

    # The real sharing groups:
    assert g[0] == g[1]
    assert g[3] == g[4] == g[5]
    assert g[7] == g[8]
    # The regression: singleton atom 2 must NOT be merged into the [7,8] group.
    assert g[2] != g[7]
    # All six residues are distinct groups (one id per residue).
    assert len({g[0], g[2], g[3], g[6], g[7], g[9]}) == 6
    # Singletons are independent of every other atom.
    for singleton in (2, 6, 9):
        assert sum(1 for x in g if x == g[singleton]) == 1
