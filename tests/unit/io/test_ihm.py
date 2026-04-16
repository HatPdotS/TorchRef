"""
Tests for IHM mmCIF reading, writing, and DataRouter integration.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from torchref.io.ihm_mapping import (
    IHMEnsembleMapping,
    IHMModelGroupInfo,
    IHMStateInfo,
)

# Path to test IHM file
TEST_IHM_FILE = Path(__file__).parent.parent.parent / "files" / "cif" / "test_ihm_ensemble.cif"


# ======================================================================
# IHMEnsembleMapping tests (no external dependencies)
# ======================================================================


class TestIHMEnsembleMapping:
    """Tests for the IHMEnsembleMapping dataclass."""

    def _make_mapping(self):
        """Create a minimal mapping for testing."""
        states = [
            IHMStateInfo(state_id=1, name="ground_state", model_num=1),
            IHMStateInfo(state_id=2, name="intermediate_1", model_num=2),
        ]
        groups = [
            IHMModelGroupInfo(
                group_id=1,
                name="dark",
                state_fractions={1: 1.0, 2: 0.0},
            ),
            IHMModelGroupInfo(
                group_id=2,
                name="1ps",
                state_fractions={1: 0.9, 2: 0.1},
            ),
            IHMModelGroupInfo(
                group_id=3,
                name="5ps",
                state_fractions={1: 0.7, 2: 0.3},
            ),
        ]
        return IHMEnsembleMapping(
            states=states,
            model_groups=groups,
            cell=[50.0, 60.0, 70.0, 90.0, 90.0, 90.0],
            spacegroup="P 21 21 21",
        )

    def test_get_state_ids(self):
        mapping = self._make_mapping()
        assert mapping.get_state_ids() == [1, 2]

    def test_get_timepoint_names(self):
        mapping = self._make_mapping()
        assert mapping.get_timepoint_names() == ["dark", "1ps", "5ps"]

    def test_get_fractions_for_group(self):
        mapping = self._make_mapping()
        fracs = mapping.get_fractions_for_group("1ps")
        assert len(fracs) == 2
        assert abs(fracs[0] - 0.9) < 1e-6
        assert abs(fracs[1] - 0.1) < 1e-6

    def test_get_fractions_for_group_missing(self):
        mapping = self._make_mapping()
        with pytest.raises(KeyError):
            mapping.get_fractions_for_group("nonexistent")

    def test_identify_dark_group(self):
        mapping = self._make_mapping()
        assert mapping.identify_dark_group() == "dark"

    def test_identify_dark_group_none(self):
        """When no group has a dominant fraction, returns None."""
        states = [
            IHMStateInfo(state_id=1, name="a", model_num=1),
            IHMStateInfo(state_id=2, name="b", model_num=2),
        ]
        groups = [
            IHMModelGroupInfo(
                group_id=1,
                name="mixed",
                state_fractions={1: 0.5, 2: 0.5},
            ),
        ]
        mapping = IHMEnsembleMapping(states=states, model_groups=groups)
        assert mapping.identify_dark_group() is None

    def test_get_state_by_id(self):
        mapping = self._make_mapping()
        state = mapping.get_state_by_id(2)
        assert state.name == "intermediate_1"

    def test_get_state_by_id_missing(self):
        mapping = self._make_mapping()
        with pytest.raises(KeyError):
            mapping.get_state_by_id(99)

    def test_get_group_by_name(self):
        mapping = self._make_mapping()
        group = mapping.get_group_by_name("5ps")
        assert group.group_id == 3

    def test_validate_ok(self):
        mapping = self._make_mapping()
        mapping.validate()  # Should not raise

    def test_validate_no_states(self):
        mapping = IHMEnsembleMapping(
            states=[],
            model_groups=[IHMModelGroupInfo(group_id=1, name="g", state_fractions={})],
        )
        with pytest.raises(ValueError, match="no states"):
            mapping.validate()

    def test_validate_bad_fractions(self):
        states = [IHMStateInfo(state_id=1, name="a", model_num=1)]
        groups = [
            IHMModelGroupInfo(
                group_id=1,
                name="g",
                state_fractions={1: 0.5},  # sums to 0.5
            ),
        ]
        mapping = IHMEnsembleMapping(states=states, model_groups=groups)
        with pytest.raises(ValueError, match="fractions sum to"):
            mapping.validate()

    def test_validate_invalid_state_reference(self):
        states = [IHMStateInfo(state_id=1, name="a", model_num=1)]
        groups = [
            IHMModelGroupInfo(
                group_id=1,
                name="g",
                state_fractions={1: 0.5, 99: 0.5},  # state 99 doesn't exist
            ),
        ]
        mapping = IHMEnsembleMapping(states=states, model_groups=groups)
        with pytest.raises(ValueError, match="state_id=99"):
            mapping.validate()

    def test_repr(self):
        mapping = self._make_mapping()
        r = repr(mapping)
        assert "ground_state" in r
        assert "dark" in r


# ======================================================================
# ModelCIFReader multi-model tests
# ======================================================================


class TestModelCIFReaderMultiModel:
    """Tests for pdbx_PDB_model_num extraction."""

    def test_get_atom_data_has_model_num(self):
        """Test that model_num column is extracted from multi-model CIF."""
        from torchref.io.cif_readers import ModelCIFReader

        reader = ModelCIFReader(str(TEST_IHM_FILE), verbose=0)
        df = reader.get_atom_data()
        assert "model_num" in df.columns
        assert set(df["model_num"].unique()) == {1, 2}

    def test_get_atom_data_by_model(self):
        """Test splitting atom data by model number."""
        from torchref.io.cif_readers import ModelCIFReader

        reader = ModelCIFReader(str(TEST_IHM_FILE), verbose=0)
        by_model = reader.get_atom_data_by_model()
        assert set(by_model.keys()) == {1, 2}
        # Each model should have 15 atoms
        assert len(by_model[1]) == 15
        assert len(by_model[2]) == 15

    def test_single_model_default(self):
        """Test that single-model files get model_num=1."""
        from torchref.io.cif_readers import ModelCIFReader

        # Use a standard single-model CIF
        single_model_cif = TEST_IHM_FILE.parent / "3E98.cif"
        if single_model_cif.exists():
            reader = ModelCIFReader(str(single_model_cif), verbose=0)
            by_model = reader.get_atom_data_by_model()
            assert 1 in by_model
            assert len(by_model) == 1


# ======================================================================
# DataRouter IHM detection tests
# ======================================================================


class TestDataRouterIHM:
    """Tests for IHM file detection in DataRouter."""

    def test_detect_ihm_ensemble(self):
        """Test that DataRouter identifies IHM files."""
        from torchref.io.data_router import DataRouter

        router = DataRouter(str(TEST_IHM_FILE), verbose=0)
        assert router.data_type == "ihm_ensemble"
        assert router.file_format == "cif"

    def test_regular_cif_not_ihm(self):
        """Test that regular CIF files are NOT detected as IHM."""
        from torchref.io.data_router import DataRouter

        regular_cif = TEST_IHM_FILE.parent / "3E98.cif"
        if regular_cif.exists():
            router = DataRouter(str(regular_cif), verbose=0)
            assert router.data_type == "structure"
            assert router.data_type != "ihm_ensemble"


# ======================================================================
# IHMReader static detection tests (no python-ihm needed)
# ======================================================================


class TestIHMReaderDetection:
    """Tests for IHMReader.is_ihm_file (gemmi-only)."""

    def test_is_ihm_file_positive(self):
        """Test that IHM file is correctly identified."""
        from torchref.io.ihm import IHMReader

        assert IHMReader.is_ihm_file(str(TEST_IHM_FILE))

    def test_is_ihm_file_negative(self):
        """Test that regular CIF file is NOT identified as IHM."""
        from torchref.io.ihm import IHMReader

        regular_cif = TEST_IHM_FILE.parent / "3E98.cif"
        if regular_cif.exists():
            assert not IHMReader.is_ihm_file(str(regular_cif))

    def test_is_ihm_file_nonexistent(self):
        """Test that nonexistent file returns False."""
        from torchref.io.ihm import IHMReader

        assert not IHMReader.is_ihm_file("/nonexistent/path.cif")


# ======================================================================
# IHMReader full tests (require python-ihm)
# ======================================================================

try:
    import ihm  # noqa: F401
    HAS_IHM = True
except ImportError:
    HAS_IHM = False


@pytest.mark.skipif(not HAS_IHM, reason="python-ihm not installed")
class TestIHMReader:
    """Tests for IHMReader (requires python-ihm)."""

    def test_read_mapping(self):
        """Test reading IHM mapping from test file."""
        from torchref.io.ihm import IHMReader

        reader = IHMReader(str(TEST_IHM_FILE), verbose=0)
        mapping = reader.read_mapping()

        assert len(mapping.states) >= 1
        assert mapping.cell is not None
        assert mapping.spacegroup is not None

    def test_read_atom_data(self):
        """Test reading per-state atom data."""
        from torchref.io.ihm import IHMReader

        reader = IHMReader(str(TEST_IHM_FILE), verbose=0)
        mapping = reader.read_mapping()
        atom_data = reader.read_atom_data(mapping)

        assert len(atom_data) == len(mapping.states)
        for state_id, df in atom_data.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 15  # our test file has 15 atoms per model

    def test_build_model_collection(self):
        """Test building ModelCollection from IHM file."""
        import torch

        from torchref.io.ihm import IHMReader

        reader = IHMReader(str(TEST_IHM_FILE), verbose=0)
        mapping = reader.read_mapping()
        mapping.atom_data_per_state = reader.read_atom_data(mapping)
        mc = reader.build_model_collection(
            mapping, max_res=3.0, device=torch.device("cpu")
        )

        # Should have base models matching number of states
        assert mc.n_base_models == len(mapping.states)

    def test_call_convenience(self):
        """Test the __call__ convenience method."""
        import torch

        from torchref.io.ihm import IHMReader

        reader = IHMReader(str(TEST_IHM_FILE), verbose=0)
        mc, mapping = reader(max_res=3.0, device=torch.device("cpu"))

        assert mc is not None
        assert mapping is not None
        assert mc.n_base_models >= 1

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        from torchref.io.ihm import IHMReader

        with pytest.raises(FileNotFoundError):
            IHMReader("/nonexistent/path.cif")


@pytest.mark.skipif(not HAS_IHM, reason="python-ihm not installed")
class TestIHMWriter:
    """Tests for IHMWriter (requires python-ihm)."""

    def test_write_from_mapping(self):
        """Test writing IHM file from a mapping."""
        import torch

        from torchref.io.ihm import IHMReader, IHMWriter

        # Read first
        reader = IHMReader(str(TEST_IHM_FILE), verbose=0)
        mc, mapping = reader(max_res=3.0, device=torch.device("cpu"))

        # Write
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            outpath = f.name

        try:
            writer = IHMWriter(mc, mapping=mapping, verbose=0)
            writer.write(outpath)

            # Verify file exists and has content
            assert os.path.exists(outpath)
            assert os.path.getsize(outpath) > 0

            # Verify it contains IHM categories
            with open(outpath) as f:
                content = f.read()
            assert "_atom_site." in content
            assert "pdbx_PDB_model_num" in content
        finally:
            os.unlink(outpath)

    def test_write_default_mapping(self):
        """Test writing IHM file without pre-existing mapping."""
        import torch

        from torchref.io.ihm import IHMReader, IHMWriter

        reader = IHMReader(str(TEST_IHM_FILE), verbose=0)
        mc, _ = reader(max_res=3.0, device=torch.device("cpu"))

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            outpath = f.name

        try:
            # Write without explicit mapping
            writer = IHMWriter(mc, verbose=0)
            writer.write(outpath)

            assert os.path.exists(outpath)
            assert os.path.getsize(outpath) > 0
        finally:
            os.unlink(outpath)


@pytest.mark.skipif(not HAS_IHM, reason="python-ihm not installed")
class TestModelCollectionIHM:
    """Tests for ModelCollection.from_ihm and write_ihm."""

    def test_from_ihm(self):
        """Test loading ModelCollection via class method."""
        import torch

        from torchref.model.model_collection import ModelCollection

        mc, mapping = ModelCollection.from_ihm(
            str(TEST_IHM_FILE),
            max_res=3.0,
            device=torch.device("cpu"),
            verbose=0,
        )
        assert mc.n_base_models >= 1
        assert isinstance(mapping, IHMEnsembleMapping)

    def test_write_ihm(self):
        """Test writing via ModelCollection method."""
        import torch

        from torchref.model.model_collection import ModelCollection

        mc, mapping = ModelCollection.from_ihm(
            str(TEST_IHM_FILE),
            max_res=3.0,
            device=torch.device("cpu"),
            verbose=0,
        )

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            outpath = f.name

        try:
            mc.write_ihm(outpath, mapping=mapping)
            assert os.path.exists(outpath)
            assert os.path.getsize(outpath) > 0
        finally:
            os.unlink(outpath)


@pytest.mark.skipif(not HAS_IHM, reason="python-ihm not installed")
class TestMixedModelIHM:
    """Tests for MixedModel.write_ihm."""

    def test_write_ihm_from_mixed_model(self):
        """Test writing IHM file directly from a MixedModel."""
        import torch

        from torchref.io.ihm import IHMReader
        from torchref.model.mixed_model import MixedModel

        # Load models from IHM file (via reader) to get ModelFT objects
        reader = IHMReader(str(TEST_IHM_FILE), verbose=0)
        mc, mapping = reader(max_res=3.0, device=torch.device("cpu"))

        # Create a MixedModel from the base models
        base_models = list(mc.base_models)
        mixed = MixedModel(base_models, initial_fractions=[0.8, 0.2])

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            outpath = f.name

        try:
            mixed.write_ihm(
                outpath,
                state_names=["ground_state", "intermediate"],
                group_name="time_point_1",
            )
            assert os.path.exists(outpath)
            assert os.path.getsize(outpath) > 0

            # Verify IHM content
            with open(outpath) as f:
                content = f.read()
            assert "_atom_site." in content
            assert "pdbx_PDB_model_num" in content
        finally:
            os.unlink(outpath)


# ======================================================================
# PDB multi-model write tests
# ======================================================================


class TestWriteMultiModel:
    """Tests for write_multi_model PDB function."""

    def test_write_multi_model(self):
        """Test writing a multi-model PDB file."""
        from torchref.io.pdb import write_multi_model

        # Create two simple DataFrames
        atoms = {
            "ATOM": ["ATOM", "ATOM"],
            "serial": [1, 2],
            "name": ["CA", "CB"],
            "altloc": ["", ""],
            "resname": ["ALA", "ALA"],
            "chainid": ["A", "A"],
            "resseq": [1, 1],
            "icode": ["", ""],
            "x": [10.0, 11.0],
            "y": [20.0, 21.0],
            "z": [30.0, 31.0],
            "occupancy": [1.0, 1.0],
            "tempfactor": [15.0, 18.0],
            "element": ["C", "C"],
            "charge": [0, 0],
            "anisou_flag": [False, False],
        }
        df1 = pd.DataFrame(atoms)
        df2 = pd.DataFrame(atoms)
        df2["x"] = [10.2, 11.2]  # Slightly shifted

        with tempfile.NamedTemporaryFile(
            suffix=".pdb", delete=False, mode="w"
        ) as f:
            outpath = f.name

        try:
            write_multi_model(
                [df1, df2],
                outpath,
                model_names=["ground_state", "intermediate"],
            )

            with open(outpath) as f:
                content = f.read()

            assert "ENDMDL" in content
            assert "ground_state" in content
            assert "intermediate" in content
            # Count MODEL records (lines starting with MODEL)
            model_lines = [l for l in content.split("\n") if l.startswith("MODEL")]
            assert len(model_lines) == 2
            assert content.count("ENDMDL") == 2
            assert "END" in content
        finally:
            os.unlink(outpath)

    def test_write_multi_model_empty(self):
        """Test that empty list produces no output."""
        from torchref.io.pdb import write_multi_model

        with tempfile.NamedTemporaryFile(
            suffix=".pdb", delete=False, mode="w"
        ) as f:
            outpath = f.name

        try:
            write_multi_model([], outpath)
            # File should either not exist or be empty
            if os.path.exists(outpath):
                assert os.path.getsize(outpath) == 0
        finally:
            if os.path.exists(outpath):
                os.unlink(outpath)
