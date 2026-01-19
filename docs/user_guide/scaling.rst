Scaling
=======

Structure factor scaling corrects for experimental effects and bulk solvent.

Overview
--------

The :class:`~torchref.scaling.scaler.Scaler` class handles:

- Overall scale factor
- Anisotropic scaling (6-parameter tensor)
- Bulk solvent correction

Basic Usage
-----------

.. code-block:: python

   from torchref.scaling.scaler import Scaler

   scaler = Scaler(model, reflection_data, verbose=1)

   scaler.initialize()
   
   scaler.refine_lbfgs() # refine scaling parameters

   # Apply scaling to calculated structure factors
   F_calc_scaled = scaler(F_calc)

Bin wise scaling
----------------

Data is binned by resolution shells and overall scale factors for each resolution bin are calculated.
This is enabled by default with 20 bins. 

.. code-block:: python

   scaler = Scaler(model, reflection_data, verbose=1,nbins=1) # sets nbins to 1, so now its just an overall scale factor
   scaler.calc_initial_scale()


Bulk Solvent Model
------------------

TorchRef uses the flat bulk solvent model:

.. math::

   F_{calc}^{total} = k \cdot F_{calc}^{model} + k_s \exp(-B_s s^2 / 4) \cdot F_{calc}^{solvent}

where:

- :math:`k` is the overall scale factor
- :math:`k_s` is the solvent scale
- :math:`B_s` is the solvent B-factor
- :math:`s = 1/d` is the reciprocal resolution

Anisotropic Scaling
-------------------

Anisotropic scaling corrects directional effects on the scattering of a crystal (mainly crystal shape). 

The anisotropic scale factor is given by:

.. math::

   k_{aniso}(\mathbf{h}) = \exp\left( -\mathbf{h}^T \mathbf{U} \mathbf{h} \right)

This scaling is enabled by calling the initialize method in :class:`~torchref.scaling.scaler.Scaler`.

