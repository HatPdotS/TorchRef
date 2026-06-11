"""Integration test for the collection (multi-dataset) σ_A path.

Builds a minimal 2-dataset collection (dark + one timepoint) from 1DAW and
checks (a) CollectionScaler.get_beta() returns one shared finite beta on
the common HKL, caches, and resets; (b) CollectionMLTarget.forward() is
finite and its gradient reaches the model.
"""

import pytest
import torch


@pytest.mark.integration
class TestCollectionSigmaA:
    @pytest.fixture(scope="class")
    def collection(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "1DAW.pdb"
        mtz = mtz_dir / "1DAW.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("1DAW fixture not present")

        from torchref import LBFGSRefinement
        from torchref.io.datasets.collection import DatasetCollection
        from torchref.model.model_collection import ModelCollection
        from torchref.scaling.collection_scaler import CollectionScaler

        # Reuse LBFGSRefinement to get a properly-loaded model + reflection data.
        ref = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0)
        data = ref.reflection_data
        model = ref.model

        dc = DatasetCollection()
        dc.add_dataset("dark", data, set_as_reference=True)
        dc.add_dataset("t1", data)

        mc = ModelCollection([model], dark_key="dark", verbose=0)
        mc.add_dark()  # frozen [1.0]
        mc.add_timepoint("t1", [1.0])

        scaler = CollectionScaler(dc, mc, verbose=0)
        scaler.initialize()
        return dc, mc, scaler

    def test_shared_beta(self, collection):
        dc, mc, scaler = collection
        n = dc.hkl.shape[0]

        beta, eps = scaler.get_beta()
        assert beta.shape[0] == n
        assert torch.isfinite(beta).all()
        assert (beta > 0).all()
        # detached (constant in autograd)
        assert not beta.requires_grad

        # cached, and reset clears it
        assert scaler._beta_cache is not None
        scaler.reset_beta_cache()
        assert scaler._beta_cache is None

    def test_forward_and_gradient(self, collection):
        from torchref.refinement.targets import CollectionMLTarget

        dc, mc, scaler = collection
        target = CollectionMLTarget(dc, mc, scaler=scaler, verbose=0)

        loss = target.forward()
        assert torch.isfinite(loss)

        loss.backward()
        xyz = mc.base_models[0].xyz.refinable_params
        assert xyz.grad is not None and torch.isfinite(xyz.grad).all()

    def test_maintenance_resets_cache(self, collection):
        from torchref.refinement.targets import CollectionMLTarget

        dc, mc, scaler = collection
        target = CollectionMLTarget(dc, mc, scaler=scaler, verbose=0)
        scaler.get_beta()
        assert scaler._beta_cache is not None
        target.maintenance()
        assert scaler._beta_cache is None
