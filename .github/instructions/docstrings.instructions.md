---
applyTo: '**/*.py'
---

# Docstring Guidelines for torchref

This document outlines the docstring conventions for the torchref package. All docstrings should follow the **NumPy docstring style**, which is fully compatible with Sphinx and widely used in scientific Python projects.

## General Principles

1. **Every public function, method, and class must have a docstring**
2. **Docstrings should be concise but complete**
3. **Use imperative mood** for descriptions (e.g., "Calculate" not "Calculates")
4. **Include type hints** in function signatures AND document types in docstrings
5. **Provide examples** for complex functions

---

## Function Docstrings

### Basic Structure

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Short one-line summary of the function (imperative mood).

    Extended description providing more details about what the function
    does, when to use it, and any important notes. This section is
    optional for simple functions.

    Parameters
    ----------
    param1 : type1
        Description of param1. If the description is long, it can
        wrap to multiple lines with proper indentation.
    param2 : type2, optional
        Description of param2. Default is X.

    Returns
    -------
    return_type
        Description of what is returned.

    Raises
    ------
    ValueError
        When param1 is invalid.
    RuntimeError
        When computation fails.

    See Also
    --------
    related_function : Brief description of relationship.

    Notes
    -----
    Additional notes about implementation details, mathematical
    formulas, or references to papers.

    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0
    """
    pass
```

### Minimal Function Docstring

For simple, self-explanatory functions:

```python
def get_atom_count(model: Model) -> int:
    """
    Return the number of atoms in the model.

    Parameters
    ----------
    model : Model
        The crystallographic model.

    Returns
    -------
    int
        Number of atoms.
    """
    return model.xyz().shape[0]
```

---

## Class Docstrings

### Basic Structure

```python
class ClassName:
    """
    Short one-line summary of the class purpose.

    Extended description explaining what this class represents,
    its main functionality, and typical use cases.

    Parameters
    ----------
    param1 : type1
        Description of constructor parameter.
    param2 : type2, optional
        Description of optional parameter. Default is X.

    Attributes
    ----------
    attr1 : type
        Description of public attribute.
    attr2 : type
        Description of another attribute.

    Examples
    --------
    Basic usage example:

    >>> obj = ClassName(param1=value)
    >>> result = obj.method()

    See Also
    --------
    RelatedClass : Brief description.

    Notes
    -----
    Implementation notes, references, or important caveats.
    """

    def __init__(self, param1: type1, param2: type2 = default):
        """
        Initialize ClassName.

        Parameters
        ----------
        param1 : type1
            Description of param1.
        param2 : type2, optional
            Description of param2. Default is X.
        """
        pass
```

### PyTorch nn.Module Classes

For classes inheriting from `torch.nn.Module`:

```python
class ModelComponent(nn.Module):
    """
    Neural network component for X computation.

    This module implements Y algorithm for Z purpose in crystallographic
    refinement. It can be used standalone or as part of a larger pipeline.

    Parameters
    ----------
    model : Model
        The atomic model containing coordinates and B-factors.
    data : ReflectionData
        Reflection data with observed structure factors.
    n_bins : int, optional
        Number of resolution bins. Default is 20.
    device : torch.device, optional
        Computation device. Default is CPU.

    Attributes
    ----------
    parameters : torch.Tensor
        Learnable parameters of shape (N,).

    Examples
    --------
    >>> component = ModelComponent(model, data)
    >>> loss = component.forward()
    >>> loss.backward()

    Notes
    -----
    This implements the algorithm described in [1]_.

    References
    ----------
    .. [1] Author, "Paper Title", Journal, Year.
    """
    pass
```

---

## Specific Conventions for torchref

### Tensor Parameters

Always specify shape and dtype expectations:

```python
def compute_structure_factors(
    hkl: torch.Tensor,
    xyz: torch.Tensor,
    b_factors: torch.Tensor
) -> torch.Tensor:
    """
    Compute structure factors from atomic model.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (n_reflections, 3), dtype int32.
    xyz : torch.Tensor
        Atomic coordinates of shape (n_atoms, 3) in Ångströms, dtype float32.
    b_factors : torch.Tensor
        Isotropic B-factors of shape (n_atoms,) in Ų, dtype float32.

    Returns
    -------
    torch.Tensor
        Complex structure factors of shape (n_reflections,), dtype complex64.
    """
    pass
```

### Crystallographic Terminology

Use consistent terminology and provide units:

```python
def calculate_resolution(d_spacing: torch.Tensor) -> torch.Tensor:
    """
    Calculate resolution from d-spacing.

    Parameters
    ----------
    d_spacing : torch.Tensor
        d-spacing values in Ångströms (Å).

    Returns
    -------
    torch.Tensor
        Resolution in Ångströms (Å), where resolution = d_spacing.

    Notes
    -----
    Resolution and d-spacing are equivalent for X-ray crystallography.
    Lower values indicate higher resolution (more detail).
    """
    pass
```

### Unit Cell and Symmetry

```python
def apply_symmetry(
    xyz: torch.Tensor,
    cell: torch.Tensor,
    spacegroup: str
) -> torch.Tensor:
    """
    Apply crystallographic symmetry operations.

    Parameters
    ----------
    xyz : torch.Tensor
        Fractional coordinates of shape (n_atoms, 3).
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] where
        a, b, c are in Ångströms and angles are in degrees.
    spacegroup : str
        Space group symbol (e.g., 'P 21 21 21', 'C 2').

    Returns
    -------
    torch.Tensor
        Symmetry-expanded coordinates of shape (n_symops * n_atoms, 3).
    """
    pass
```

---

## Method Docstrings

### Regular Methods

```python
def method_name(self, param: type) -> return_type:
    """
    Short description of what the method does.

    Parameters
    ----------
    param : type
        Description of parameter.

    Returns
    -------
    return_type
        Description of return value.
    """
    pass
```

### Properties

```python
@property
def n_atoms(self) -> int:
    """
    int : Number of atoms in the model.

    This property returns the total count of atoms including
    all residues and chains.
    """
    return self._xyz.shape[0]
```

### Magic Methods

```python
def __repr__(self) -> str:
    """Return string representation of the object."""
    return f"Model(n_atoms={self.n_atoms}, spacegroup='{self.spacegroup}')"

def __len__(self) -> int:
    """Return the number of atoms."""
    return self.n_atoms
```

---

## Module-Level Docstrings

Each Python module should have a module docstring at the top:

```python
"""
Module for crystallographic refinement targets.

This module provides target functions (loss functions) for structure
refinement, including X-ray targets, geometry targets, and ADP targets.

Classes
-------
GaussianXrayTarget
    Gaussian negative log-likelihood for X-ray data.
BondTarget
    Bond length restraint target.
AngleTarget
    Bond angle restraint target.

Functions
---------
nll_xray
    Calculate X-ray negative log-likelihood.
rfactor
    Calculate crystallographic R-factor.

Examples
--------
>>> from torchref.refinement.targets import GaussianXrayTarget
>>> target = GaussianXrayTarget(refinement)
>>> loss = target.forward()

Notes
-----
All targets inherit from the base Target class and follow the
PyTorch nn.Module interface.
"""
```

---

## Common Sections Reference

| Section | When to Use |
|---------|-------------|
| `Parameters` | Always, if function has parameters |
| `Returns` | Always, if function returns something |
| `Raises` | When function can raise exceptions |
| `Examples` | For complex or commonly-used functions |
| `See Also` | To reference related functions/classes |
| `Notes` | For implementation details, math, references |
| `Warnings` | For important caveats or deprecations |
| `References` | For academic citations |

---

## Type Annotations

Always use type hints in function signatures. Common types for torchref:

```python
from typing import Optional, Tuple, List, Dict, Union
import torch
from torch import Tensor

# Common patterns:
def func(
    tensor: Tensor,                           # PyTorch tensor
    optional_param: Optional[float] = None,   # Optional with default
    shape: Tuple[int, ...] = (3, 3),         # Tuple of ints
    items: List[str] = None,                  # List of strings
    config: Dict[str, any] = None,           # Dictionary
    value: Union[int, float] = 1.0,          # Multiple types
) -> Tensor:
    pass
```

---

## Sphinx Integration

These docstrings are compatible with Sphinx using the `sphinx.ext.napoleon` extension. To build documentation:

```python
# conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
```

---

## Quick Reference Template

Copy this template for new functions:

```python
def function_name(param1: type1, param2: type2 = default) -> return_type:
    """
    One-line summary.

    Parameters
    ----------
    param1 : type1
        Description.
    param2 : type2, optional
        Description. Default is X.

    Returns
    -------
    return_type
        Description.
    """
    pass
```

Copy this template for new classes:

```python
class ClassName:
    """
    One-line summary.

    Parameters
    ----------
    param1 : type1
        Description.

    Attributes
    ----------
    attr1 : type
        Description.

    Examples
    --------
    >>> obj = ClassName(value)
    >>> obj.method()
    """

    def __init__(self, param1: type1):
        """Initialize ClassName."""
        pass
```
