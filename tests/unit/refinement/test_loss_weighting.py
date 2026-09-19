"""Pin refinement's default group weights; LossState owns weight arithmetic."""

import pytest


class TestDefaultGroupWeights:
    """The validated default base group weights (single source of truth)."""

    @pytest.mark.unit
    def test_default_group_weights_values(self):
        """DEFAULT_GROUP_WEIGHTS is the AF-screen-tuned xray 1 / geom 0.2 / adp 0.02."""
        from torchref.refinement.base_refinement import DEFAULT_GROUP_WEIGHTS

        assert DEFAULT_GROUP_WEIGHTS == {
            "xray": 1.0,
            "geometry": 0.2,
            "geometry/ramachandran": 0.0,
            "adp": 0.02,
            # Sub-weight on the SIGD prior; 1.0 leaves it at the adp group weight
            # pending the R_free scan. Weights multiply down the path.
            "adp/sigd": 1.0,
            # Node load balancing, inert unless the ADPs are a node field. Above the
            # group weight because it bars a degenerate direction rather than
            # competing with the data.
            "adp/node_load": 10.0,
            # Magnitude prior on node values, off pending measurement.
            "adp/node_smoothness": 0.0,
        }
