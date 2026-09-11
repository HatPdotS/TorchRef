Scaling
=======

:class:`~torchref.scaling.scaler.Scaler` puts F_calc on the observed scale and
absorbs what the atomic model does not describe: an overall isotropic scale, an
anisotropic correction, and the bulk solvent contribution.

Basic Usage
-----------

.. code-block:: python

   from torchref import Scaler

   scaler = Scaler(model, reflection_data, verbose=1)

   scaler.initialize()      # initial scale + solvent + anisotropy
   scaler.refine_lbfgs()    # refine the scaling parameters

   F_calc_scaled = scaler(F_calc)

``initialize()`` is what brings the three terms into existence
(``calc_initial_scale`` → ``setup_solvent`` → ``setup_anisotropy_correction``).
Each factor defaults to 1 while its parameter is absent, so a freshly
constructed scaler is the identity, and one on which only
``calc_initial_scale()`` has run applies the overall isotropic scale alone.

Isotropic Scaling
-----------------

The overall scale is a Chebyshev polynomial in :math:`s = \sin\theta/\lambda`,
evaluated per reflection:

.. math::

   k_{iso}(s) = \exp\left( \sum_{i} c_i\, T_i(u) \right),
   \qquad u \in [-1, 1]

with ``n_iso_coeff`` coefficients (default 6) held in ``scaler.c_iso``. Every
reflection contributes to every coefficient with a continuous weight, so there are
no bin boundaries and nothing changes discontinuously when a reflection moves
between shells. ``n_iso_coeff=1`` is a single global scale
(:math:`T_0 \equiv 1`); ``2`` spans scale-plus-overall-B.

.. code-block:: python

   scaler = Scaler(model, reflection_data, n_iso_coeff=1)   # single global scale
   scaler.calc_initial_scale()

Resolution bins survive only as the device that *seeds* the coefficients: the
closed-form per-bin :math:`|F_{obs}|/|F_{calc}|` ratio is projected onto the basis
by least squares, so the fit starts from the curve a binned model would have
started from. Nothing downstream is binned; ``nbins`` controls only that seed.

Use ``scaler.iso_log_scale()`` for the per-reflection log scale, and
``scaler.get_scale()`` for a single summary number.

Bulk Solvent Model
------------------

The solvent contribution is mask-derived, not analytic: a binary solvent mask is
built from the model and Fourier transformed to give :math:`F_{solvent}`, which is
then damped and scaled.

.. math::

   F_{calc}^{total} = k_{iso}(s)\, k_{aniso}(\mathbf{h}) \cdot F_{calc}^{model}
       + k_s \exp\!\left( -\ln 2 \left(\frac{s^2}{s^2_{1/2}}\right)^{\!n} \right)
         \cdot F_{calc}^{solvent}

where :math:`k_s` is the solvent scale, :math:`s^2_{1/2}` the point at which the
solvent term is halved, :math:`n` how sharply it switches off, and
:math:`s = \sin\theta/\lambda` — the *half*-length of the scattering vector
(``ScalerBase._s_half_sq``), which is why the exponent carries no factor of 4.

:math:`n = 1` reduces this exactly to a Debye-Waller factor
:math:`\exp(-B_s s^2)` with :math:`B_s = \ln 2 / s^2_{1/2}`, so a solvent B from
another program transfers unchanged. Larger :math:`n` gives a plateau followed by a
sharper cutoff, which is the shape a flat bulk-solvent prior actually has: it
describes the data well at low resolution and then stops being informative.

The refined parameters are ``log_k_solvent``, ``log_ss_half``, ``log_n_exp`` and
``phase_offset``; each falloff parameter is refined in log space so it stays
positive, and is clamped to ``SS_HALF_BOUNDS`` / ``N_EXP_BOUNDS``. The phase offset
blends the mask phases toward the protein phases and is only active when
``optimize_phase`` is set.

PDB ``REMARK 3`` and mmCIF carry a single solvent B, which this form does not have;
``SolventModel.b_solvent_equivalent`` back-fits one from the curve for deposition.

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
