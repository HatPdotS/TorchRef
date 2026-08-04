Contributing
============

Contributions are very welcome. If you have an idea for an improvement or find a
bug, open a GitHub issue — or fix it yourself and send a pull request.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/HatPdotS/TorchRef.git
   cd TorchRef
   pip install -e ".[dev]"

Use of Generative AI
--------------------

Please feel free to use it, including for docstrings and refactoring. Review what
it produces against the code before submitting.

Code Style
----------

- **Python**: PEP 8, 88-character lines (Black)
- **Type hints**: on all public functions
- **Docstrings**: NumPy style on every public function, method, and class

Docstring Format
----------------

A docstring answers "how do I call this and what will it do to me?" for someone
who is not going to read the body. Document the contract — what it does,
parameters, returns, raises — and any trap a caller needs to avoid a silently
wrong result: dtype or device restrictions, in-place mutation, a cache that must
be invalidated afterwards. Keep design rationale to a clause, and leave benchmark
numbers and superseded approaches to the commit history.

.. code-block:: python

   def compute_structure_factors(hkl, xyz, b_factors):
       """Compute structure factors for the given reflections.

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
       """

Don't restate the signature in prose — types live in the annotations. Don't add
an ``Examples`` block that is entirely ``# doctest: +SKIP``: it costs lines and
tests nothing. The examples in :doc:`quickstart` run under
``sphinx.ext.doctest``, so put runnable examples there and they will be checked.

Running Tests
-------------

.. code-block:: bash

   pytest tests/                    # all tests
   pytest tests/ --cov=torchref     # with coverage
   pytest tests/unit/               # fast unit tests only

GPU, slow, and Amber tests are skipped unless enabled. See :doc:`user_guide/testing`.

Submitting Changes
------------------

1. Branch, change, add tests, confirm the suite passes.
2. Open a pull request with a clear description and any relevant issue numbers.
