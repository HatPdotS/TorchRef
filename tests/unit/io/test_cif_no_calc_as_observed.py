"""Regression test: the CIF reader must not treat F_calc as observed F.

_refln.F_calc was in the F_obs column-search fallback, so a calc-only SF-CIF
silently loaded calculated amplitudes as observations (meaningless R-factors).
F_calc is removed from the fallback; a file with no measured F/I now raises a
clear error instead of masquerading calc as observed. See TORCHREF_AUDIT.md.
"""

import pytest

from torchref.io import cif

_CALC_ONLY = """data_test
loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_calc
1 0 0 100.0
2 0 0 90.0
0 1 0 80.0
1 1 0 70.0
1 1 1 60.0
"""

_CELL = """_cell.length_a      50.000
_cell.length_b      60.000
_cell.length_c      70.000
_cell.angle_alpha   90.000
_cell.angle_beta    90.000
_cell.angle_gamma   90.000
_symmetry.space_group_name_H-M   'P 21 21 21'
"""

_MEASURED = "data_test\n" + _CELL + """loop_
_refln.index_h
_refln.index_k
_refln.index_l
_refln.F_meas_au
_refln.F_meas_sigma_au
1 0 0 100.0 2.0
2 0 0 90.0 2.0
0 1 0 80.0 2.0
1 1 0 70.0 2.0
1 1 1 60.0 2.0
"""


@pytest.mark.unit
def test_calc_only_cif_does_not_load_calc_as_observed(tmp_path):
    p = tmp_path / "calc_only-sf.cif"
    p.write_text(_CALC_ONLY)
    # Pre-fix: F_calc was silently loaded as F_obs. Now: no observed data -> raise.
    with pytest.raises(ValueError, match="observed|F_calc|measured"):
        cif.ReflectionCIFReader(str(p))


@pytest.mark.unit
def test_measured_cif_still_loads(tmp_path):
    p = tmp_path / "measured-sf.cif"
    p.write_text(_MEASURED)
    reader = cif.ReflectionCIFReader(str(p))
    assert "F" in reader.data
    # The loaded F came from a measured column, not F_calc.
    assert "calc" not in str(reader.data.get("F_col", "")).lower()
