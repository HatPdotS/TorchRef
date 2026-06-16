"""Regression test: corefine_scaler default consistency.

_scaler_body_params() used getattr(self, "corefine_scaler", True) while the
constructor default is False. An instance built without __init__ (e.g.
create_from_state_dict) therefore silently co-refined the scaler. The getattr
fallback now matches the constructor default (False). See TORCHREF_AUDIT.md.
"""

import pytest


@pytest.mark.integration
def test_scaler_body_params_default_holds_scaler_fixed(pdb_dir, mtz_dir):
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    from torchref import LBFGSRefinement

    ref = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0)

    # Constructor default is False -> scaler held fixed during body steps.
    assert ref.corefine_scaler is False
    assert ref._scaler_body_params() == []

    # Missing attr (the create_from_state_dict path) must match the default,
    # not silently co-refine. This is the regression: fallback was True.
    del ref.corefine_scaler
    assert ref._scaler_body_params() == []

    # Explicit opt-in returns the scaler parameters.
    ref.corefine_scaler = True
    params = ref._scaler_body_params()
    assert isinstance(params, list) and len(params) > 0
