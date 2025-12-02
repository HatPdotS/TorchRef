"""
Unit tests for torchref.utils.utils

Tests utility classes and functions.
"""

import pytest
import torch
import torch.nn as nn


class TestModuleReference:
    """Tests for ModuleReference wrapper class."""

    @pytest.mark.unit
    def test_module_reference_creation(self):
        """Test creating a ModuleReference."""
        from torchref.utils.utils import ModuleReference
        
        inner_module = nn.Linear(10, 5)
        ref = ModuleReference(inner_module)
        
        assert ref.module is inner_module

    @pytest.mark.unit
    def test_module_reference_not_registered(self):
        """ModuleReference should not register wrapped module as submodule."""
        from torchref.utils.utils import ModuleReference
        
        class ParentModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)  # This gets registered
                self._ref = ModuleReference(nn.Linear(5, 2))  # This should NOT
        
        parent = ParentModule()
        
        # Count registered submodules
        submodules = list(parent.modules())
        # Should be: parent, linear (2 total)
        # The wrapped module should NOT be counted
        assert len(submodules) == 2

    @pytest.mark.unit
    def test_module_reference_attribute_forwarding(self):
        """Test that attributes are forwarded to wrapped module."""
        from torchref.utils.utils import ModuleReference
        
        inner = nn.Linear(10, 5)
        ref = ModuleReference(inner)
        
        # Access attribute via reference
        assert ref.in_features == 10
        assert ref.out_features == 5

    @pytest.mark.unit
    def test_module_reference_callable(self):
        """Test that ModuleReference is callable."""
        from torchref.utils.utils import ModuleReference
        
        inner = nn.Linear(10, 5)
        ref = ModuleReference(inner)
        
        x = torch.randn(3, 10)
        output = ref(x)  # Call through reference
        
        assert output.shape == (3, 5)

    @pytest.mark.unit
    def test_module_reference_repr(self):
        """Test string representation."""
        from torchref.utils.utils import ModuleReference
        
        inner = nn.Linear(10, 5)
        ref = ModuleReference(inner)
        
        repr_str = repr(ref)
        assert "ModuleReference" in repr_str
        assert "Linear" in repr_str


class TestCIFReader:
    """Tests for CIFReader class - basic parsing tests without file I/O."""

    @pytest.mark.unit
    def test_cif_reader_initialization(self):
        """Test CIFReader can be initialized without file."""
        from torchref.utils.utils import CIFReader
        
        reader = CIFReader()
        
        assert reader.data == {}
        assert reader.filepath is None

    @pytest.mark.unit
    def test_cif_reader_tokenize_simple(self):
        """Test tokenization of simple CIF lines."""
        from torchref.utils.utils import CIFReader
        
        reader = CIFReader()
        
        # Test simple space-separated tokens
        tokens = reader._tokenize_line("A B C")
        assert tokens == ["A", "B", "C"]

    @pytest.mark.unit
    def test_cif_reader_tokenize_quoted(self):
        """Test tokenization handles quoted strings."""
        from torchref.utils.utils import CIFReader
        
        reader = CIFReader()
        
        # Test quoted string
        tokens = reader._tokenize_line("'hello world' test")
        assert len(tokens) == 2
        assert tokens[0] == "hello world"
        assert tokens[1] == "test"
