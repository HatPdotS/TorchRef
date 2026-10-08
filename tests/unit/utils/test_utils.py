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

    @pytest.mark.unit
    @pytest.mark.parametrize("how", ["copy", "deepcopy", "pickle"])
    def test_module_reference_copies_and_pickles(self, how):
        """``copy`` shares the referent; ``deepcopy`` and pickling copy it."""
        import copy
        import pickle

        from torchref.utils.utils import ModuleReference

        inner = nn.Linear(2, 2)
        clone = {
            "copy": copy.copy,
            "deepcopy": copy.deepcopy,
            "pickle": lambda r: pickle.loads(pickle.dumps(r)),
        }[how](ModuleReference(inner))

        assert isinstance(clone, ModuleReference)
        assert (clone.module is inner) == (how == "copy")
        torch.testing.assert_close(clone.weight, inner.weight)

    @pytest.mark.unit
    def test_module_reference_deepcopy_follows_the_copied_graph(self):
        """A deep-copied owner's reference points at the owner's copied child."""
        import copy

        from torchref.utils.utils import ModuleReference

        class Owner(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = nn.Linear(2, 2)
                self.ref = ModuleReference(self.child)

        owner = Owner()
        clone = copy.deepcopy(owner)

        assert clone.child is not owner.child
        assert clone.ref.module is clone.child

    @pytest.mark.unit
    def test_module_reference_does_not_forward_private_names(self):
        """Underscore lookups stop at the reference; public ones reach the referent."""
        from torchref.utils.utils import ModuleReference

        ref = ModuleReference(nn.Linear(2, 2))

        assert not hasattr(ref, "_apply")
        assert ref.in_features == 2


class TestSerialization:
    """One tensor-to-JSON rule, shared by convert_to_serializable and the encoder."""

    @pytest.mark.unit
    def test_empty_tensor_becomes_an_empty_list(self):
        import json

        from torchref.utils.serialization import convert_to_serializable

        assert convert_to_serializable({"empty": torch.zeros(0)}) == {"empty": []}
        assert json.dumps(torch.zeros(0)) == "[]"

    @pytest.mark.unit
    def test_one_element_tensor_stays_a_scalar(self):
        import json

        from torchref.utils.serialization import convert_to_serializable
        from torchref.utils.stats import stat

        assert convert_to_serializable(torch.zeros(1)) == 0.0
        assert json.dumps({"r": stat(torch.ones(1))}) == '{"r": 1.0}'
        assert json.dumps(torch.arange(3)) == "[0, 1, 2]"


@pytest.mark.unit
def test_tensordict_repr_is_balanced():
    from torchref.utils.utils import TensorDict

    assert repr(TensorDict({"a": torch.ones(1)})) == "TensorDict({a: tensor([1.])})"
