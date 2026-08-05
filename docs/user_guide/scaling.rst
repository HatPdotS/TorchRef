Scaling
=======

:class:`~torchref.scaling.scaler.Scaler` puts F_calc on the observed scale and
absorbs what the atomic model does not describe: an overall (per-resolution-bin)
scale, an anisotropic correction, and the bulk solvent contribution.

Basic Usage
-----------

.. code-block:: python

   from torchref import Scaler

   scaler = Scaler(model, reflection_data, verbose=1)

   scaler.initialize()      # initial bin scales + solvent + anisotropy
   scaler.refine_lbfgs()    # refine the scaling parameters

   F_calc_scaled = scaler(F_calc)

``initialize()`` is what brings the three terms into existence
(``calc_initial_scale`` → ``setup_solvent`` → ``setup_anisotropy_correction``).
Each factor defaults to 1 while its parameter is absent, so a freshly
constructed scaler is the identity, and one on which only
``calc_initial_scale()`` has run applies the overall bin scale alone.

Bin-wise Scaling
----------------

Reflections are binned by resolution shell and an overall scale is fitted per
bin. On by default with 20 bins:

.. code-block:: python

   scaler = Scaler(model, reflection_data, verbose=1, nbins=1)   # single global scale
   scaler.calc_initial_scale()

Bulk Solvent Model
------------------

The solvent contribution is mask-derived, not analytic: a solvent mask is built
from the model, smoothed, and Fourier transformed to give :math:`F_{solvent}`,
which is then Debye-Waller damped and scaled.

.. math::

   F_{calc}^{total} = k \cdot F_{calc}^{model}
                    + k_s \exp(-B_s s^2) \cdot F_{calc}^{solvent}

where :math:`k` is the overall (per-bin) scale, :math:`k_s` the solvent scale,
:math:`B_s` the solvent B-factor, and :math:`s = \sin\theta/\lambda` — the
*half*-length of the scattering vector (``ScalerBase._s_half_sq``), which is why
the exponent carries no factor of 4. This is the ordinary Debye-Waller
convention written the other way round: :math:`\exp(-B_s/4d^2)`, i.e.
:math:`\exp(-B_s s^2/4)` for :math:`s = 1/d`. A :math:`B_s` from another program
therefore transfers unchanged (the default is 46 Å²).

The refined parameters are ``log_k_solvent``, ``b_solvent`` and
``phase_offset``; the last blends the mask phases toward the protein phases and
is only active when ``optimize_phase`` is set.

Anisotropic Scaling
-------------------

Corrects direction-dependent effects, chiefly crystal shape:

.. math::

   k_{aniso}(\mathbf{h}) = \exp\left( -2\pi^2\,
       \mathbf{s}^T \mathbf{U} \mathbf{s} \right)

with :math:`\mathbf{s}` the **full** scattering vector
:math:`(h a^*, k b^*, l c^*)`, :math:`|\mathbf{s}| = 1/d` — ``ScalerBase.s``,
*not* the half-vector of the solvent term above, which would change the exponent
by a factor of 4 — and :math:`\mathbf{U}` the 6-parameter symmetric tensor
stored on the scaler. The
:math:`2\pi^2` is part of the definition, not a unit choice — dropping it makes
the fitted ``U`` disagree with an ADP-convention ``U`` by that factor.
