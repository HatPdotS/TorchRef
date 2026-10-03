"""Build fresh paired collections for difference-refinement integration tests."""

import pytest
import torch


@pytest.fixture
def difference_models(loaded_reflection_data, sample_structure_pair):
    """Return independent dark/light models and raw 1DAW dataset copies per test."""
    from torchref.cli._common import load_model
    from torchref.io import DatasetCollection
    from torchref.model import ModelCollection

    data = loaded_reflection_data
    assert data.I is not None
    models = [
        load_model(
            str(sample_structure_pair["model"]),
            max_res=2.05,
            device=data.device,
            verbose=0,
        )
        for _ in range(2)
    ]
    with torch.no_grad():
        models[1].xyz.refinable_params += 0.2
    dc = DatasetCollection(device=data.device, verbose=0)
    dc.add_dataset("dark", data, set_as_reference=True).add_dataset("light", data)
    mc = ModelCollection(models, dark_key="dark", verbose=0)
    mc.add_dark().add_timepoint("light", [0.78, 0.22])
    return dc, mc


@pytest.fixture
def difference_collection(difference_models):
    """Add a fresh initialized model-to-data scaler to the paired collection."""
    from torchref.scaling import CollectionScaler

    dc, mc = difference_models
    return dc, mc, CollectionScaler(dc, mc, verbose=0).initialize()
