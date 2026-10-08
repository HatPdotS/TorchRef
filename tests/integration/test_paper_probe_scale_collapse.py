"""``paper/probe_scale_collapse.py`` reads the scale of the scaler it probes."""

import importlib.util
import sys

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def probe(project_root, test_files_dir, tmp_path, monkeypatch):
    """The script as a module, reading 1DAW's placed AlphaFold model and data."""
    # The script puts the repository on sys.path when it loads.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "probe_scale_collapse", project_root / "paper" / "probe_scale_collapse.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "1DAW").mkdir()
    (tmp_path / "1DAW" / "1DAW.mtz").symlink_to(test_files_dir / "mtz" / "1DAW.mtz")
    monkeypatch.setattr(module, "PLACED", test_files_dir / "pdb")
    monkeypatch.setattr(module, "DATA", tmp_path)
    return module


def test_default_target_yields_a_per_reflection_scale(probe):
    """``one`` returns the fitted log scale and finite R-factors on the default target."""
    log_scale, r_work, r_free = probe.one("1DAW", probe.DEFAULT_SCALE_TARGET)

    k = log_scale.exp()
    assert k.numel() > 0 and bool((k > 0).all())
    assert 0 < r_work < 1 and 0 < r_free < 1
