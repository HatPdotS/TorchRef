Contributing
============

Contributions are very welcome. This started as a small personal project but has become quite complex.
If you have any idea for improvement or find a bug let me know via a github issue or fis it yourself and submit a pull request!


Development Setup
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/HatPdotS/TorchRef.git
      cd torchref

2. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

3. Use of Generative AI
    Please feel free to use it, most of the doc strings were written with its help and only corrected.
    I used it a lot for docstrings and refactoring.

Code Style
----------

We follow these conventions:

- **Python Style**: PEP 8 with 88 character line length (Black formatter)
- **Docstrings**: NumPy style (see below)
- **Type Hints**: Use type hints for all public functions

Docstring Format
----------------

All public functions, methods, and classes must have NumPy-style docstrings:

.. code-block:: python

   def compute_structure_factors(hkl, xyz, b_factors):
       """
       Compute structure factors for given reflections.

       Parameters
       ----------
       hkl : torch.Tensor
           Miller indices of shape (N, 3).
       xyz : torch.Tensor
           Atomic coordinates of shape (M, 3) in Ångströms.
       b_factors : torch.Tensor
           Isotropic B-factors of shape (M,) in Ų.

       Returns
       -------
       torch.Tensor
           Complex structure factors of shape (N,).

       Raises
       ------
       ValueError
           If tensor shapes are incompatible.

       Examples
       --------
       >>> hkl = torch.tensor([[1, 0, 0], [0, 1, 0]])
       >>> xyz = torch.tensor([[0.0, 0.0, 0.0]])
       >>> b = torch.tensor([20.0])
       >>> F = compute_structure_factors(hkl, xyz, b)
       """
       pass

Running Tests
-------------

Run the test suite:

.. code-block:: bash

   # All tests
   pytest tests/

   # With coverage
   pytest tests/ --cov=torchref

   # Specific test categories
   pytest tests/unit/           # Fast unit tests
   pytest tests/integration/    # Integration tests

Submitting Changes
------------------

1. Create a branch for your changes
2. Make your changes with appropriate tests
3. Ensure all tests pass
4. Submit a pull request

Please include:

- Clear description of the changes
- Any relevant issue numbers
- Tests for new functionality
