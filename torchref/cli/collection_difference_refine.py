#!/usr/bin/env python3 -u

"""
Collection-based difference refinement: joint scaling with bulk solvent.

Uses ModelCollection / DatasetCollection / CollectionScaler so ONE set of scale
parameters (overall scale, anisotropy, bulk solvent k_sol/B_sol) is shared across the
dark and light datasets.

Writes refined dark and light models (PDB/CIF), a JSON summary, and a difference MTZ.
See :func:`write_results_mtz` for the columns and why each is where it is -- the table
used to live here, four hundred lines from the function that writes it, which is part of
how the output drifted from its own documentation.

Examples
--------
::

    torchref.difference-refine \
        -dm dark.pdb -lm light.pdb \
        -dsf dark.mtz -lsf light.mtz \
        --fraction 0.37 -o output/
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

from torchref.cli._common import (
    add_all_columns_arg,
    add_dmin_arg,
    add_dual_model_args,
    add_general_args,
    add_metadata_args,
    add_outdir_arg,
    add_output_format_args,
    add_weights_arg,
    build_dual_column_names,
    configure_unbuffered_output,
    load_model,
    load_reflection_data,
    parse_device_str,
    parse_weights,
    register_timing,
    validate_cif_files,
    validate_files,
)
from torchref.utils.serialization import convert_to_serializable

configure_unbuffered_output()

# ---------------------------------------------------------------------------
# Default target weights
# ---------------------------------------------------------------------------

DEFAULT_TARGET_WEIGHTS = {
    "xray/difference": 1.0,
    # The absolute channel. Zero by default: the difference refinement fixes the
    # dark model, so the overall level is already anchored and this term only adds
    # the systematic errors the difference cancels.
    "xray/ml": 0.0,
    # Registered only under --two-moment; harmless in the dict either way.
    "xray/two_moment": 1.0,
    # "geometry/bond": 1.0, # geometry restraint should never require tuning, so leave at 1.0
    # "geometry/angle": 1.0,
    # "geometry/torsion": 1.0,
    # "geometry/planarity": 1.0,
    # "geometry/chiral": 1.0,
    # "geometry/nonbonded": 1.0,
    # "geometry/ramachandran": 1.0,
    # "adp/simu": 1.0,
    # "adp/locality": 1.0,
    # "adp/sigd": 1.0,
    "similarity": 1.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_model_collection(pdb_dark, pdb_light, fractions, cif, d_min,
                           device, verbose, hydrogenate=False):
    """Load models and create a ModelCollection.

    Parameters
    ----------
    hydrogenate : bool
        If True, add explicit hydrogens to both models.  H atoms
        participate in geometry/VDW restraints (preventing clashes)
        but are excluded from structure factor calculations.
    """
    from torchref.model.model_collection import ModelCollection

    model_dark = load_model(
        pdb_dark, max_res=d_min, device=device, verbose=verbose, cif=cif,
    )
    model_light = load_model(
        pdb_light, max_res=d_min, device=device, verbose=verbose, cif=cif,
    )

    if hydrogenate:
        if verbose > 0:
            print("Adding hydrogens for VDW clash prevention...")
            sys.stdout.flush()
        model_dark = model_dark.hydrogenate(verbose=max(0, verbose - 1))
        model_light = model_light.hydrogenate(verbose=max(0, verbose - 1))
        model_dark.hydrogens_in_xray = False
        model_light.hydrogens_in_xray = False

    mc = ModelCollection([model_dark, model_light], dark_key="dark")
    mc.add_dark()
    mc.add_timepoint("light", fractions=fractions)
    return mc


def setup_dataset_collection(sf_dark, sf_light, d_min, device,
                             column_names_dark=None, column_names_light=None):
    """Load reflection data and create a DatasetCollection."""
    from torchref import DatasetCollection

    data_dark = load_reflection_data(
        sf_dark, device=device, column_names=column_names_dark,
    )
    data_light = load_reflection_data(
        sf_light, device=device, column_names=column_names_light,
    )
    if d_min is not None:
        data_dark.cut_res(highres=d_min)
        data_light.cut_res(highres=d_min)

    dc = DatasetCollection(device=device)
    dc.add_dataset("dark", data_dark)
    dc.add_dataset("light", data_light)
    dc.scale()
    return dc


def setup_scaler(dataset_collection, model_collection, device, verbose=1):
    """Create a CollectionScaler with per-component solvent models."""
    from torchref.scaling import CollectionScaler

    scaler = CollectionScaler(
        dataset_collection=dataset_collection,
        model_collection=model_collection,
        device=device,
        verbose=verbose,
    )
    scaler.initialize()
    scaler.screen_solvent_params_joint()
    scaler.refine_lbfgs_joint()
    return scaler


def setup_dark_only(pdb_dark, dc, cif, d_min, device, verbose, hydrogenate=False):
    """Load the dark model alone and scale it against the dark data.

    This is everything a weighted difference map needs. The amplitude is
    ``|Fo_light| - |Fo_dark|``, which :meth:`DatasetCollection.scale` has already put on
    one scale without reference to any model, and the phase comes from the dark state.
    So there is no mixed model, no occupancy fraction, and no joint model-to-data fit --
    the joint fit exists to share scale parameters between two models, and here there is
    only one.

    A fitted scaler rather than a bare ``angle(model(hkl))`` because bulk solvent
    contributes a phase at low resolution, which is exactly where difference density is
    largest.

    Returns
    -------
    tuple
        ``(model_dark, scaler)`` -- ready to hand to :func:`write_results_mtz` with
        ``mc=None``.
    """
    from torchref import Scaler

    model_dark = load_model(
        pdb_dark, max_res=d_min, device=device, verbose=verbose, cif=cif,
    )
    if hydrogenate:
        if verbose > 0:
            print("Adding hydrogens...")
            sys.stdout.flush()
        model_dark = model_dark.hydrogenate(verbose=max(0, verbose - 1))
        model_dark.hydrogens_in_xray = False

    scaler = Scaler(
        model_dark, dc["dark"], device=device, verbose=max(-1, verbose - 1),
    )
    scaler.initialize().refine_lbfgs()
    return model_dark, scaler


def compute_rfactors(model, data, scaler):
    """Compute R-work/R-free with the scaler's own solvent model applied.

    Routes through ``rfactor_work_free`` — the shared source of truth used by the
    refinement targets — so the validity mask is applied and the validation set is
    excluded from both work and free, matching every other reported R-factor.

    Takes either scaler kind; see the branch below.
    """
    from torchref.base.metrics.rfactor import rfactor_work_free

    with torch.no_grad():
        hkl = data.hkl
        fcalc = model(hkl)
        # CollectionScaler needs fractions to mix components; a single-dataset
        # Scaler consumes structure factors directly.
        if hasattr(scaler, "forward_mixed"):
            fcalc_scaled = scaler.forward_mixed(fcalc, model.fractions)
        else:
            fcalc_scaled = scaler(fcalc)
        return rfactor_work_free(data, torch.abs(fcalc_scaled))


def setup_loss_state(dataset_collection, model_collection, scaler,
                     target_weights, device, similarity_alpha=2.0,
                     two_moment=False):
    """Build LossState with collection-aware targets.

    Geometry and ADP restraints are applied only to the light base model
    (the dark model is a frozen reference).

    Parameters
    ----------
    two_moment : bool, optional
        Also register the two-moment intensity target, which fits merged intensities
        under ``|F(alpha)|^2 + sigma_alpha^2 |dF|^2``. Requires I/SIGI on every
        dataset. Default False.
    """
    from torchref.refinement import LossState
    from torchref.refinement.targets import TotalADPTarget, TotalGeometryTarget
    from torchref.refinement.targets.collection import (
        CollectionDifferenceTarget,
        CollectionMLTarget,
    )
    from torchref.refinement.targets.similarity import CoordinateSimilarityTarget

    state = LossState(device=device)

    model_light = model_collection.base_models[1]
    model_dark = model_collection.base_models[0]

    diff_target = CollectionDifferenceTarget(
        dataset_collection, model_collection, scaler=scaler,
    )
    ml_target = CollectionMLTarget(
        dataset_collection, model_collection, scaler=scaler,
    )
    geom_target = TotalGeometryTarget(model_light)
    adp_target = TotalADPTarget(model_light)

    similarity_target = CoordinateSimilarityTarget(
        model_dark=model_dark, model_light=model_light, alpha=similarity_alpha,
    )

    state.register_target("xray/difference", diff_target)
    state.register_target("xray/ml", ml_target)
    state.register_target("geometry", geom_target)
    state.register_target("adp", adp_target)
    state.register_target("similarity", similarity_target)

    if two_moment:
        from torchref.refinement.targets import CollectionTwoMomentIntensityTarget

        two_moment_target = CollectionTwoMomentIntensityTarget(
            dataset_collection, model_collection, scaler=scaler, verbose=1,
        )
        # Match intensity and amplitude gradient norms to balance the targets
        # against the geometry restraints despite their different units.
        two_moment_target.calibrate_base_weight(
            diff_target, list(model_light.parameters())
        )
        state.register_target("xray/two_moment", two_moment_target)

    state.set_weights(target_weights)

    return state


def compute_bayes_extrapolated_amplitudes(
    Fobs_dark, Fobs_light, sig_ext, phi_dark, phi_mixed, f,
    *, tau_sq_floor=1e-4,
):
    """Empirical Bayes shrinkage estimator for extrapolated SF amplitudes.

    Estimates per-reflection shrinkage weights from the propagated variance of the
    extrapolation, then shrinks the phase-aware extrapolated amplitude toward
    Fo_dark, regularising noisy high-resolution and weakly-measured reflections::

        F_ext     = |F_dark*e^(iφ_d) + ΔF/f|         (phase-aware amplitude)
        τ²        = max(<(F_ext - Fo_dark)²> - <σ_ext²>, floor)
        w(h)      = τ² / (τ² + σ_ext²(h))
        F_extb    = w(h)·F_ext + (1-w(h))·Fo_dark    (amplitude shrinkage)

    Parameters
    ----------
    Fobs_dark, Fobs_light : Tensor (N,)
        Observed amplitudes.
    sig_ext : Tensor (N,)
        Propagated uncertainty of the extrapolated amplitude. Taken from the caller
        rather than rebuilt here: ``F_ext`` is linear in the observations with
        ``dF_ext/dF_light = 1/f`` and ``dF_ext/dF_dark = 1 - 1/f = -(1-f)/f``, so the
        dark term carries a ``(1-f)**2`` weight.
    phi_dark, phi_mixed : Tensor (N,)
        Calculated phases (radians) for the dark and mixed models.
    f : float or Tensor
        Excited-state population fraction.
    tau_sq_floor : float
        Floor on the estimated signal variance τ².

    Returns
    -------
    tuple
        ``(F_ext_bayes, var_ext_bayes, w_shrinkage, tau_sq)`` -- the **shrunk**
        extrapolated amplitude, its posterior variance and the shrinkage weight per
        reflection, and the global τ² as a float.
    """
    F_dark_phased = Fobs_dark * torch.exp(1j * phi_dark)
    F_light_phased = Fobs_light * torch.exp(1j * phi_mixed)
    delta_F = F_light_phased - F_dark_phased

    sig_sq_ext = sig_ext**2

    # Phase-aware extrapolated amplitude
    F_ext_complex = F_dark_phased + delta_F / f
    F_ext = torch.abs(F_ext_complex)

    # Estimate signal variance τ²
    residuals_sq = (F_ext - Fobs_dark) ** 2
    tau_sq = max((residuals_sq.mean() - sig_sq_ext.mean()).item(), tau_sq_floor)

    # Per-reflection shrinkage weight (in [0, 1])
    w = tau_sq / (tau_sq + sig_sq_ext)

    # Posterior variance
    var_ext_bayes = (tau_sq * sig_sq_ext) / (tau_sq + sig_sq_ext)

    # Shrink the amplitude toward Fo_dark -- scalar, so no phase interference.
    F_ext_bayes = w * F_ext + (1 - w) * Fobs_dark

    return F_ext_bayes, var_ext_bayes, w, tau_sq


def _two_moment_columns(mc, dc, mask, fcalc_dark_full, fcalc_mixed_full,
                        *, weights, diff_Fobs, Fcalc_diff_amp, Fobs_dark,
                        sig_dark, phi_mixed, F_obs_dark_phased, all_columns=False):
    """Two-moment diagnostic columns, or empty dicts when the model is off.

    The observed light intensity carries a positive, phase-blind contamination
    ``sigma_alpha^2 |dF|^2`` from the spread of activation across crystals. Subtracting
    the model's estimate of it and converting back to an amplitude gives a difference
    amplitude that is comparable across datasets, which the raw one is not.

    ``DDF`` is the diagnostic that matters: smooth and featureless against resolution
    means the correction is collinear with a scale or overall-B error and should be
    distrusted; structure in it is the signal.

    The decontaminated amplitude goes through the dataset's own French-Wilson estimator,
    on the **full** reflection list, because subtracting the variance term pushes weak
    reflections negative and that is exactly where a naive ``sqrt(clamp(I, 0))`` is worst.

    Parameters
    ----------
    mask : torch.Tensor
        The dark-and-light validity intersection the writer uses; the returned columns
        are already reduced to it.
    fcalc_dark_full, fcalc_mixed_full : torch.Tensor
        Scaled complex structure factors on the **full** HKL list.
    weights, diff_Fobs, Fcalc_diff_amp, Fobs_dark, sig_dark : numpy.ndarray
        Masked quantities the writer has already computed, reused so the corrected
        columns are constructed exactly like their uncorrected counterparts.

    Returns
    -------
    tuple
        ``(columns, types)`` -- the values, and the MTZ type letter for each. Carrying
        the type beside the value preserves the crystallographic column type.
    """
    import numpy as np

    empty = ({}, {})
    if float(mc.sigma_alpha_sq) == 0.0:
        return empty

    data_light = dc["light"]
    if data_light.I is None:
        return empty

    with torch.no_grad():
        # Full-size, so French-Wilson sees the reflection list it was fitted on.
        delta_F_full = fcalc_mixed_full - fcalc_dark_full
        variance_full = mc.sigma_alpha_sq * delta_F_full.abs() ** 2

        I_light_full, sig_I_full = data_light.get_corrected_intensities()
        I_corrected_full = I_light_full - variance_full

        # The retained estimator is fitted on the dataset's HKL list *as loaded*;
        # joining a collection expands the dataset onto the common grid, so it can be
        # the wrong length by then. Rebuild against the current list when that happens.
        fw = data_light._FrenchWilson
        if fw is None or len(fw.d_spacings) != len(I_corrected_full):
            from torchref.base.french_wilson import FrenchWilson

            fw = FrenchWilson(
                data_light.hkl, data_light.cell, data_light.spacegroup, verbose=0
            )
        F_corr_full, sig_F_corr_full = fw(I_corrected_full, sig_I_full)

        def _np(t):
            return t[mask].detach().cpu().numpy()

        variance = _np(variance_full)
        I_light = _np(I_light_full)
        sig_I_light = _np(sig_I_full)
        I_coherent = _np(fcalc_mixed_full.abs() ** 2)
        F_corr = _np(F_corr_full)
        sig_F_corr = _np(sig_F_corr_full)

    I_two_moment = I_coherent + variance
    DF_corr = F_corr - Fobs_dark
    DDF = DF_corr - diff_Fobs
    sig_DF_corr = np.sqrt(sig_F_corr**2 + sig_dark**2)

    # Use the modulus of the complex vector difference so the corrected
    # coefficient retains the phase rotation between the dark and light states.
    F_corr_phased = torch.as_tensor(
        F_corr, dtype=F_obs_dark_phased.real.dtype, device=F_obs_dark_phased.device
    ) * torch.exp(1j * phi_mixed)
    Fobs_diff_phased_corr = (
        torch.abs(F_corr_phased - F_obs_dark_phased).detach().cpu().numpy()
    )

    amp_2_corr = (2 * Fobs_diff_phased_corr - Fcalc_diff_amp) * weights
    amp_1_corr = (Fobs_diff_phased_corr - Fcalc_diff_amp) * weights

    # The sigma_alpha^2-aware weight, on the same normalisation as the inverse-variance
    # weight the existing DED coefficients carry, so the two are directly comparable.
    w_two_moment = sig_I_light**2 / np.maximum(sig_I_light**2 + variance, 1e-12)

    columns = {
        # The corrected difference map, on the same dark phases as DELFWT.
        "DELFWT_corr": DF_corr * weights,
        "Fo_light_corr": F_corr,
        "SIGFo_light_corr": sig_F_corr,
        "DF_corr": DF_corr,
        "SIGDF_corr": sig_DF_corr,
        "DDF": DDF,
    }
    types = {
        "DELFWT_corr": "F",
        "Fo_light_corr": "F",
        "SIGFo_light_corr": "Q",
        "DF_corr": "F",
        "SIGDF_corr": "Q",
        "DDF": "F",
    }
    if all_columns:
        columns.update({
            "Io_light": I_light,
            "SIGIo_light": sig_I_light,
            "Ic_light_coh": I_coherent,
            "Ic_light_2mom": I_two_moment,
            "IVAR_ALPHA": variance,
            "W_2MOM": w_two_moment,
            "2mDFop-DFc_corr": amp_2_corr,
            "mDFop-DFc_corr": amp_1_corr,
        })
        types.update({
            "Io_light": "J",
            "SIGIo_light": "Q",
            "Ic_light_coh": "J",
            "Ic_light_2mom": "J",
            "IVAR_ALPHA": "J",
            "W_2MOM": "W",
            "2mDFop-DFc_corr": "F",
            "mDFop-DFc_corr": "F",
        })
    return columns, types


def _difference_columns(data_dark, data_light, mask, hkl_np, *, Fobs_dark, sig_dark,
                        Fobs_light, sig_light, Fcalc_dark, phases_dark, diff_Fobs,
                        sig_diff, weights):
    """The weighted difference map, and the observations behind it.

    ``DELFWT``/``PHDELWT`` is the inverse-variance-weighted amplitude difference carried
    on the **dark** model's phases -- the isomorphous difference Fourier, and the same
    construction ``torchref.validate-ded`` correlates against, so the map in this file
    and the map the validation reports are one object. CCP4 and Coot recognise the names
    and open it as a difference map without being told which columns to use.

    This layer needs no light-state model: the amplitude is ``|Fo_light| - |Fo_dark|``
    and the phase comes from the dark model. Keeping the light state's model out is the
    point -- a phased construction puts its phases into the observed amplitude, biasing
    the map toward the very model the experiment is testing.
    """
    import numpy as np

    n = len(hkl_np)

    def _flags(data):
        if data.rfree_flags is None:
            return np.ones(n, dtype=int)
        return data.rfree_flags[mask].cpu().numpy().astype(int)

    columns = {
        "H": hkl_np[:, 0], "K": hkl_np[:, 1], "L": hkl_np[:, 2],
        "Fo_dark": Fobs_dark, "SIGFo_dark": sig_dark,
        "Fo_light": Fobs_light, "SIGFo_light": sig_light,
        "DF": diff_Fobs, "SIGDF": sig_diff,
        "DELFWT": diff_Fobs * weights, "PHDELWT": phases_dark,
        "Fc_dark": Fcalc_dark,
        # 1 = work, 0 = free. Both are kept: the two datasets can disagree, and
        # picking one would silently report an R-free against the wrong test set.
        "FreeR_flag_dark": _flags(data_dark),
        "FreeR_flag_light": _flags(data_light),
    }
    types = {
        "H": "H", "K": "H", "L": "H",
        "Fo_dark": "F", "SIGFo_dark": "Q",
        "Fo_light": "F", "SIGFo_light": "Q",
        "DF": "F", "SIGDF": "Q",
        "DELFWT": "F", "PHDELWT": "P",
        "Fc_dark": "F",
        "FreeR_flag_dark": "I", "FreeR_flag_light": "I",
    }
    return columns, types


def _phasing_columns(mc, scaler, hkl_all, mask, *, fcalc_dark, Fobs_dark_vals,
                     Fobs_light_vals, phi_dark, Fcalc_dark, weights,
                     all_columns=False):
    """Mixed-model amplitude and phase, and the phased difference residuals.

    Everything here needs the light state's model. ``FC``/``PHIC`` are the mixed model's
    scaled amplitude and phase.

    Under ``all_columns`` the phased difference residual coefficients come too --
    ``(|Fo_light e^{i phi_mixed} - Fo_dark e^{i phi_dark}| - |dFc|) * w`` on
    ``PHIC_diff``. These are a *different object* from the plain difference Fourier in
    ``DELFWT``, not a refinement of it: the light state's model phases enter the observed
    amplitude, so they are model-biased where ``DELFWT`` is not. They are kept because
    they are informative once that is understood, and gated because the name alone does
    not say it.

    Returns ``(columns, types, ctx)``. ``ctx`` carries the intermediates the
    extrapolation and two-moment layers need, so nothing is computed twice.
    """
    mixed_model = mc["light"]

    with torch.no_grad():
        fcalc_mixed_full = scaler.forward_mixed(
            mixed_model(hkl_all), mixed_model.fractions
        )
        fcalc_mixed = fcalc_mixed_full[mask]
    fcalc_diff = fcalc_mixed - fcalc_dark

    phi_mixed = torch.angle(fcalc_mixed)
    F_obs_dark_phased = Fobs_dark_vals * torch.exp(1j * phi_dark)
    F_obs_light_phased = Fobs_light_vals * torch.exp(1j * phi_mixed)

    Fcalc_light = torch.abs(fcalc_mixed).detach().cpu().numpy()
    Fcalc_diff_amp = torch.abs(fcalc_diff).detach().cpu().numpy()

    columns = {
        "FC": Fcalc_light,
        "PHIC": phi_mixed.detach().rad2deg().cpu().numpy(),
    }
    types = {"FC": "F", "PHIC": "P"}

    if all_columns:
        Fobs_diff_phased = torch.abs(
            F_obs_light_phased - F_obs_dark_phased
        ).detach().cpu().numpy()
        columns.update(
            {
                "2mDFop-DFc": (2 * Fobs_diff_phased - Fcalc_diff_amp) * weights,
                "mDFop-DFc": (Fobs_diff_phased - Fcalc_diff_amp) * weights,
                "PHIC_diff": torch.angle(fcalc_diff).detach().rad2deg().cpu().numpy(),
                "DFc": Fcalc_light - Fcalc_dark,
                # This column holds the real modulus of the complex vector difference.
                "DFc_phased": Fcalc_diff_amp,
            }
        )
        types.update({
            "2mDFop-DFc": "F", "mDFop-DFc": "F", "PHIC_diff": "P",
            "DFc": "F", "DFc_phased": "F",
        })

    ctx = {
        "fcalc_mixed_full": fcalc_mixed_full,
        "phi_mixed": phi_mixed,
        "F_obs_dark_phased": F_obs_dark_phased,
        "F_obs_light_phased": F_obs_light_phased,
        "Fcalc_diff_amp": Fcalc_diff_amp,
    }
    return columns, types, ctx


def _extrapolation_columns(mc, dc, hkl, *, Fobs_dark_vals, Fobs_light_vals,
                           sig_dark_vals, sig_light_vals, phi_dark, ctx,
                           rfree_flags_masked, all_columns=False, verbose=1):
    """Extrapolated light-state amplitudes and the map to refine against.

    Three constructions of the same quantity, all needing the light model:

    ``FEXT`` (default, Bayes-shrunk)
        The phase-aware amplitude shrunk toward ``Fo_dark`` by a per-reflection weight
        ``w(h) = tau^2 / (tau^2 + sigma_ext^2(h))``, which quiets the weak and
        high-resolution reflections where the extrapolation is noisiest.
    ``FEXT_PHASED`` (``all_columns``)
        The unshrunk phase-aware amplitude.
    ``FEXT_SCALAR`` (``all_columns``)
        ``(Fo_light - w_dark * Fo_dark) / w_light`` on amplitudes only.

    Each needs ``F_calc`` rescaled against *its own* amplitudes -- the three sets differ
    in overall scale -- so each costs one LBFGS scale fit. Only the default one runs
    unless ``all_columns`` is set.

    ``FWT``/``PHWT`` is ``2 * FEXT - Fc`` with the phase from the Bayes fit. The phase
    matters: the scaler contributes one through ``f_sol``, so the three fits do not agree
    and pairing these coefficients with another fit's phase would be wrong.
    """
    from torchref import ReflectionData, Scaler
    from torchref.base.metrics.rfactor import rfactor_work_free

    data_light = dc["light"]
    fractions = mc["light"].fractions.detach()
    w_dark, w_light = fractions[0], fractions[1]

    def _fit(amp, sig):
        """Rescale the light model against one set of extrapolated amplitudes."""
        data = ReflectionData.from_tensors(
            hkl=hkl, F=amp, F_sigma=sig,
            cell=data_light.cell, spacegroup=data_light.spacegroup,
            rfree_flags=rfree_flags_masked, device=str(hkl.device), verbose=0,
        )
        sc = Scaler(mc.base_models[1], data, device=hkl.device, verbose=-1)
        sc.initialize().refine_lbfgs()
        return data, sc(mc.base_models[1](hkl))

    # The phase-aware amplitude, and the one propagated sigma for this extrapolation.
    F_light_extra = (
        ctx["F_obs_light_phased"] - w_dark * ctx["F_obs_dark_phased"]
    ) / w_light
    sig_light_extra = torch.sqrt(
        sig_light_vals**2 + w_dark**2 * sig_dark_vals**2
    ) / w_light

    F_ext_bayes_amp, var_ext_bayes, w_shrinkage, tau_sq = (
        compute_bayes_extrapolated_amplitudes(
            Fobs_dark_vals, Fobs_light_vals, sig_light_extra,
            phi_dark, ctx["phi_mixed"], w_light,
        )
    )
    sig_ext_bayes = torch.sqrt(var_ext_bayes)

    data_bayes, F_calc_bayes = _fit(F_ext_bayes_amp, sig_ext_bayes)
    amp_calc_bayes = torch.abs(F_calc_bayes)

    def _np(t):
        return t.detach().cpu().numpy()

    columns = {
        "FEXT": _np(F_ext_bayes_amp),
        "SIGFEXT": _np(sig_ext_bayes),
        "FWT": _np(2 * F_ext_bayes_amp - amp_calc_bayes),
        "PHWT": _np(torch.angle(F_calc_bayes).rad2deg()),
    }
    types = {"FEXT": "F", "SIGFEXT": "Q", "FWT": "F", "PHWT": "P"}

    if verbose > 0:
        print("  Bayes extrapolation rfactors:",
              rfactor_work_free(data_bayes, amp_calc_bayes))
        print(f"  Bayes: tau^2 = {tau_sq:.4f}, "
              f"mean w(h) = {w_shrinkage.mean().item():.3f}")

    if all_columns:
        amp_phased = torch.abs(F_light_extra)
        data_phased, F_calc_phased = _fit(amp_phased, sig_light_extra)
        amp_calc_phased = torch.abs(F_calc_phased)

        amp_scalar = (Fobs_light_vals - w_dark * Fobs_dark_vals) / w_light
        data_scalar, F_calc_scalar = _fit(amp_scalar, sig_light_extra)
        amp_calc_scalar = torch.abs(F_calc_scalar)

        columns.update({
            "FEXT_PHASED": _np(amp_phased),
            "SIGFEXT_PHASED": _np(sig_light_extra),
            "2FEXT_PHASED-Fc": _np(2 * amp_phased - amp_calc_phased),
            "FEXT_PHASED-Fc": _np(amp_phased - amp_calc_phased),
            "PHFEXT_PHASED": _np(torch.angle(F_calc_phased).rad2deg()),
            "FEXT_SCALAR": _np(amp_scalar),
            "SIGFEXT_SCALAR": _np(sig_light_extra),
            "2FEXT_SCALAR-Fc": _np(2 * amp_scalar - amp_calc_scalar),
            "FEXT_SCALAR-Fc": _np(amp_scalar - amp_calc_scalar),
            "PHFEXT_SCALAR": _np(torch.angle(F_calc_scalar).rad2deg()),
        })
        types.update({
            "FEXT_PHASED": "F", "SIGFEXT_PHASED": "Q",
            "2FEXT_PHASED-Fc": "F", "FEXT_PHASED-Fc": "F",
            "PHFEXT_PHASED": "P",
            "FEXT_SCALAR": "F", "SIGFEXT_SCALAR": "Q",
            "2FEXT_SCALAR-Fc": "F", "FEXT_SCALAR-Fc": "F",
            "PHFEXT_SCALAR": "P",
        })
        if verbose > 0:
            print("  Phase-aware extrapolation rfactors:",
                  rfactor_work_free(data_phased, amp_calc_phased))
            print("  Scalar extrapolation rfactors:",
                  rfactor_work_free(data_scalar, amp_calc_scalar))

    diagnostics = {
        "tau_sq": float(tau_sq),
        "w_shrinkage_mean": float(w_shrinkage.mean().item()),
    }
    return columns, types, diagnostics


def write_results_mtz(dc, dark_model, scaler, filename, *, mc=None,
                      all_columns=False, verbose=1):
    """Write the difference map, and map coefficients when a light model is given.

    The default output is the **weighted difference map**: ``DELFWT``/``PHDELWT``, the
    inverse-variance-weighted amplitude difference on the dark model's phases. That needs
    no light-state model, which is why ``mc`` is optional -- with a dark model alone this
    writes a difference map and nothing else, and no scale fit is run beyond the one that
    produced ``scaler``.

    Given ``mc``, the layers that need the light state follow: its amplitude and phase,
    the extrapolated amplitudes, and the two-moment correction. ``all_columns`` adds the
    alternatives within each layer -- see :func:`_phasing_columns` and
    :func:`_extrapolation_columns` for what each contains and why it is gated.

    Parameters
    ----------
    dc : DatasetCollection
        Dark and light data, already inter-scaled by :meth:`DatasetCollection.scale`.
    dark_model : Model or _SharedMixedModel
        Supplies ``Fc_dark`` and the phases the difference map is carried on.
    scaler : Scaler or CollectionScaler
        Scales ``dark_model`` against the dark data. A ``CollectionScaler`` when ``mc``
        is given, a single-dataset ``Scaler`` otherwise.
    mc : ModelCollection, optional
        The dark+light collection. Absent means difference map only.
    filename : str
        Output MTZ path.

    Returns
    -------
    dict
        Diagnostics worth recording outside the file -- currently the Bayes shrinkage's
        ``tau_sq`` and mean ``w(h)``, which say whether the default extrapolated map is
        over-shrunk. Empty when no light model was given.
    """
    import reciprocalspaceship as rs

    data_dark = dc[mc.dark_key] if mc is not None else dc["dark"]
    data_light = dc["light"]

    hkl_all = data_dark.hkl
    Fobs_dark_full, sig_dark_full = data_dark.get_corrected_data()
    Fobs_light_full, sig_light_full = data_light.get_corrected_data()

    mask = data_dark.masks().to(torch.bool) & data_light.masks().to(torch.bool)
    hkl = hkl_all[mask]
    Fobs_dark_vals = Fobs_dark_full[mask]
    Fobs_light_vals = Fobs_light_full[mask]
    sig_dark_vals = sig_dark_full[mask]
    sig_light_vals = sig_light_full[mask]
    rfree_flags_masked = (
        data_light.rfree_flags[mask] if data_light.rfree_flags is not None else None
    )

    # Fcalc on the full HKL list then masked, because the scalers were fitted on the
    # full datasets. ``forward_mixed`` exists only on CollectionScaler; the dark-only
    # path carries a single-dataset Scaler, whose ``forward`` is the plain call.
    with torch.no_grad():
        if hasattr(scaler, "forward_mixed"):
            fcalc_dark_full = scaler.forward_mixed(
                dark_model(hkl_all), dark_model.fractions
            )
        else:
            fcalc_dark_full = scaler(dark_model(hkl_all))
        fcalc_dark = fcalc_dark_full[mask]

    phi_dark = torch.angle(fcalc_dark)

    hkl_np = hkl.cpu().numpy()
    Fobs_dark = Fobs_dark_vals.cpu().numpy()
    Fobs_light = Fobs_light_vals.cpu().numpy()
    sig_dark = sig_dark_vals.cpu().numpy()
    sig_light = sig_light_vals.cpu().numpy()
    Fcalc_dark = torch.abs(fcalc_dark).detach().cpu().numpy()
    phases_dark = phi_dark.detach().rad2deg().cpu().numpy()

    diff_Fobs = Fobs_light - Fobs_dark
    sig_diff = (sig_dark**2 + sig_light**2) ** 0.5
    weights = 1 / sig_diff**2
    weights = weights / weights.mean()

    columns, types = _difference_columns(
        data_dark, data_light, mask, hkl_np,
        Fobs_dark=Fobs_dark, sig_dark=sig_dark,
        Fobs_light=Fobs_light, sig_light=sig_light,
        Fcalc_dark=Fcalc_dark, phases_dark=phases_dark,
        diff_Fobs=diff_Fobs, sig_diff=sig_diff, weights=weights,
    )

    diagnostics = {}
    if mc is not None:
        phase_cols, phase_types, ctx = _phasing_columns(
            mc, scaler, hkl_all, mask,
            fcalc_dark=fcalc_dark, Fobs_dark_vals=Fobs_dark_vals,
            Fobs_light_vals=Fobs_light_vals, phi_dark=phi_dark,
            Fcalc_dark=Fcalc_dark, weights=weights, all_columns=all_columns,
        )
        columns.update(phase_cols)
        types.update(phase_types)

        ext_cols, ext_types, diagnostics = _extrapolation_columns(
            mc, dc, hkl,
            Fobs_dark_vals=Fobs_dark_vals, Fobs_light_vals=Fobs_light_vals,
            sig_dark_vals=sig_dark_vals, sig_light_vals=sig_light_vals,
            phi_dark=phi_dark, ctx=ctx, rfree_flags_masked=rfree_flags_masked,
            all_columns=all_columns, verbose=verbose,
        )
        columns.update(ext_cols)
        types.update(ext_types)

        tm_cols, tm_types = _two_moment_columns(
            mc, dc, mask, fcalc_dark_full, ctx["fcalc_mixed_full"],
            weights=weights, diff_Fobs=diff_Fobs,
            Fcalc_diff_amp=ctx["Fcalc_diff_amp"], Fobs_dark=Fobs_dark,
            sig_dark=sig_dark, phi_mixed=ctx["phi_mixed"],
            F_obs_dark_phased=ctx["F_obs_dark_phased"],
            all_columns=all_columns,
        )
        columns.update(tm_cols)
        types.update(tm_types)

    df = rs.DataSet(
        columns,
        cell=data_dark.cell.data.cpu().tolist(),
        spacegroup=data_dark.spacegroup.hm,
    )
    # Carry MTZ types with the values; infer_mtz_dtypes also checks the
    # result using the canonical writer's rules.
    missing = set(columns) - set(types)
    if missing:
        raise AssertionError(f"columns with no declared MTZ type: {sorted(missing)}")
    for name, letter in types.items():
        df[name] = df[name].astype(letter)
    df = df.infer_mtz_dtypes()
    df.set_index(["H", "K", "L"], inplace=True)
    df.write_mtz(filename)

    if verbose > 0:
        print(f"  Results MTZ written to {filename} ({len(columns)} columns)")
        if mc is not None:
            fractions = mc["light"].fractions.detach()
            print(f"  w_dark={fractions[0].item():.3f}, "
                  f"w_light={fractions[1].item():.3f}")

    return diagnostics


def optimize_lbfgs(state, parameters, max_iter, nsteps, n_clean, verbose):
    """Run a block of LBFGS optimisation steps via :meth:`LossState.step`.

    ``state.step`` handles the closure, NaN validation, and automatically
    disables ``requires_grad`` on any loss-relevant leaves outside
    ``parameters`` — in particular the dark model's leaves, which appear in
    the difference target's autograd graph but are intentionally not in the
    optimizer's intent. The dark model effectively becomes a frozen
    reference at the autograd level for the duration of each step.
    """
    parameters = list(parameters)

    def _make_optimizer():
        return torch.optim.LBFGS(
            parameters, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

    optimizer = _make_optimizer()
    for i in range(nsteps):
        if i > 0 and i % n_clean == 0:
            # Periodic LBFGS curvature-history reset.
            optimizer = _make_optimizer()
        state.step(
            optimizer, context="collection_difference_refine.optimize_lbfgs"
        )
        if verbose > 0:
            with torch.no_grad():
                loss = state.aggregate()
                print(f"    LBFGS step {i + 1}/{nsteps}, loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Entry point for ``torchref.difference-refine``; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="torchref.difference-refine",
        description="Collection-based difference refinement with joint "
                    "scaling and bulk solvent correction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  torchref.difference-refine \\
      -dm dark.pdb -lm light.pdb \\
      -dsf dark.mtz -lsf light.mtz \\
      --fraction 0.37 -o output/
        """,
    )

    add_dual_model_args(parser)

    output = parser.add_argument_group("Output")
    add_outdir_arg(output, help="Output directory for refined structures and maps")
    add_output_format_args(output)
    add_metadata_args(output)
    add_all_columns_arg(output)

    refine = parser.add_argument_group("Refinement")
    refine.add_argument(
        "--weight-schedule", type=str, default="5,3,2",
        help="Comma-separated difference-target weights applied in "
             "sequence each macro-cycle (default: '5,3,2')",
    )
    refine.add_argument(
        "--n-cycles", type=int, default=3,
        help="Number of macro-cycles (default: 3)",
    )
    refine.add_argument(
        "--n-steps", type=int, default=2,
        help="LBFGS optimisation rounds per weight step (default: 2)",
    )
    refine.add_argument(
        "--max-iter", type=int, default=100,
        help="Max line-search iterations per LBFGS step (default: 100)",
    )
    refine.add_argument(
        "--n-clean", type=int, default=2,
        help="Reset LBFGS history every N steps (default: 2)",
    )
    add_weights_arg(refine, default_weights=DEFAULT_TARGET_WEIGHTS)
    refine.add_argument(
        "--refine-fractions", action="store_true", default=False,
        help="Refine population fractions during optimisation (default: frozen)",
    )
    refine.add_argument(
        "--similarity-weight", type=float, default=1.0,
        help="Weight for dark/light coordinate similarity restraint "
             "(0 to disable, default: 1.0)",
    )
    two_moment = parser.add_argument_group("Activation heterogeneity (two-moment model)")
    two_moment.add_argument(
        "--two-moment", action="store_true", default=False,
        help="Fit merged intensities with |F(alpha)|^2 + sigma_alpha^2 |dF|^2, "
             "which accounts for crystal-to-crystal spread in activation. "
             "Requires I/SIGI columns in both reflection files.",
    )
    two_moment.add_argument(
        "--lambda-twin", type=float, default=0.0,
        help="Activation dispersion as a fraction of its maximum, in [0, 1]: "
             "sigma_alpha^2 = alpha (1 - alpha) * lambda. 0 (default) is the "
             "coherent model and reproduces the amplitude-only result. Needs "
             "--two-moment: the dispersion belongs in the predicted intensity, not "
             "in a weight.",
    )
    two_moment.add_argument(
        "--refine-lambda-twin", action="store_true", default=False,
        help="Refine --lambda-twin instead of holding it fixed. Off by default: "
             "the sigma_alpha^2 term is smooth and positive, so it is collinear "
             "with a scale or overall-B error and can absorb one.",
    )

    refine.add_argument(
        "--similarity-alpha", type=float, default=2.0,
        help="Log prior odds for spike-and-slab similarity restraint. "
             "Higher = stronger denoising (default: 2.0)",
    )

    res = parser.add_argument_group("Resolution")
    add_dmin_arg(res)

    add_general_args(parser)

    args = parser.parse_args()
    register_timing()

    # --- Parse fractions ---
    if not (0.0 < args.fraction < 1.0):
        print(f"Error: --fraction must be between 0 and 1 (got {args.fraction})",
              file=sys.stderr)
        return 1
    fractions = [1.0 - args.fraction, args.fraction]

    if not (0.0 <= args.lambda_twin <= 1.0):
        print(
            f"Error: --lambda-twin must lie in [0, 1] (got {args.lambda_twin})",
            file=sys.stderr,
        )
        return 1
    if (args.lambda_twin > 0.0 or args.refine_lambda_twin) and not args.two_moment:
        # Activation heterogeneity changes the predicted mean intensity. Treating
        # it as measurement variance downweights the reflections carrying the signal.
        print(
            "Error: --lambda-twin needs --two-moment. The dispersion enters the predicted "
            "intensity, not a weight: as a variance it down-weights the reflections whose "
            "difference signal is largest, which measurably worsens the recovered "
            "displacement.",
            file=sys.stderr,
        )
        return 1

    # --- Parse weight schedule ---
    try:
        weight_schedule = [float(x) for x in args.weight_schedule.split(",")]
        if not weight_schedule:
            raise ValueError
    except ValueError:
        print("Error: --weight-schedule must be comma-separated floats",
              file=sys.stderr)
        return 1

    # --- Parse and merge target weights ---
    target_weights = dict(DEFAULT_TARGET_WEIGHTS)
    target_weights["xray/difference"] = weight_schedule[0]
    target_weights["similarity"] = args.similarity_weight
    target_weights, err = parse_weights(args.weights, defaults=target_weights)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # --- Validate input files ---
    rc = validate_files([
        (args.dark_model, "Dark model"),
        (args.light_model, "Light model"),
        (args.dark_structure_factor, "Dark structure factor"),
        (args.light_structure_factor, "Light structure factor"),
    ])
    if rc:
        return rc
    rc = validate_cif_files(args.cif)
    if rc:
        return rc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = parse_device_str(args.device)
    d_min = args.dmin if args.dmin is not None else 1.0

    # --- Header ---
    if args.verbose > 0:
        print("=" * 72)
        print("TorchRef Collection Difference Refinement")
        print("=" * 72)
        print(f"Dark model:        {args.dark_model}")
        print(f"Light model:       {args.light_model}")
        print(f"Dark data:         {args.dark_structure_factor}")
        print(f"Light data:        {args.light_structure_factor}")
        frac_mode = "refinable" if args.refine_fractions else "frozen"
        print(f"Fractions:         dark={fractions[0]}, light={fractions[1]} ({frac_mode})")
        if args.two_moment:
            lam_mode = "refinable" if args.refine_lambda_twin else "fixed"
            print(
                f"Activation spread: lambda_twin={args.lambda_twin} ({lam_mode})"
            )
        print(f"Output:            {outdir}")
        print(f"Device:            {device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        if args.cif:
            print(f"CIF restraints:    {', '.join(args.cif)}")
        print(f"Weight schedule:   {weight_schedule} x {args.n_cycles} cycles")
        print(f"LBFGS steps/weight: {args.n_steps}  (max_iter={args.max_iter})")
        print()
        print("Target weights:")
        for wk, wv in sorted(target_weights.items()):
            print(f"  {wk}: {wv}")
        print("=" * 72)
        print()
        sys.stdout.flush()

    # --- Setup models ---
    if args.verbose > 0:
        print("Setting up models...")
        sys.stdout.flush()

    mc = setup_model_collection(
        args.dark_model, args.light_model, fractions,
        args.cif, d_min, device, args.verbose,
    )
    dark = mc.dark_model            # _SharedMixedModel (fractions=[1, 0])
    mixed = mc["light"]             # _SharedMixedModel (fractions=[1-f, f] from --fraction)
    model_dark = mc.base_models[0]  # ModelFT (for output)
    model_light = mc.base_models[1] # ModelFT (for output)

    if args.refine_fractions:
        mixed.unfreeze_fractions()
    else:
        mixed.freeze_fractions()

    # --- Setup data ---
    if args.verbose > 0:
        print("Loading reflection data...")
        sys.stdout.flush()

    col_dark, col_light = build_dual_column_names(args)

    dc = setup_dataset_collection(
        args.dark_structure_factor, args.light_structure_factor,
        args.dmin, device,
        column_names_dark=col_dark, column_names_light=col_light,
    )
    data_dark = dc["dark"]
    data_light = dc["light"]

    if args.verbose > 0:
        # Compare R-free flags between datasets
        if data_dark.rfree_flags is not None and data_light.rfree_flags is not None:
            rfree_d = data_dark.rfree_flags.bool()
            rfree_l = data_light.rfree_flags.bool()
            n_agree = (rfree_d == rfree_l).sum().item()
            n_total = len(rfree_d)
            n_free_d = (~rfree_d).sum().item()
            n_free_l = (~rfree_l).sum().item()
            print(f"  R-free flags: dark={n_free_d} free, light={n_free_l} free, "
                  f"agreement={n_agree}/{n_total} ({100*n_agree/n_total:.1f}%)")
        sys.stdout.flush()

    # --- Setup scaler ---
    if args.verbose > 0:
        print("Setting up joint scaler...")
        sys.stdout.flush()

    scaler = setup_scaler(dc, mc, device, args.verbose)

    if args.verbose > 0:
        r_work_d, r_free_d = compute_rfactors(dark, data_dark, scaler)
        r_work_l, r_free_l = compute_rfactors(mixed, data_light, scaler)
        r_work_dl, r_free_dl = compute_rfactors(dark, data_light, scaler)
        print(f"  Initial R-factor (dark  vs dark data):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  Initial R-factor (mixed vs light data): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print(f"  Initial R-factor (dark  vs light data): R_work={r_work_dl:.4f}  R_free={r_free_dl:.4f}")
        print()
        sys.stdout.flush()

    # --- Setup targets ---
    if args.two_moment:
        missing = [k for k in dc.keys() if dc[k].I is None]
        if missing:
            print(
                f"Error: --two-moment needs I/SIGI columns, but {missing} carry "
                f"only amplitudes. Converting back with F**2 would reintroduce the "
                f"French-Wilson distortion the intensity model exists to avoid.",
                file=sys.stderr,
            )
            return 1

    # Set unconditionally: a non-zero dispersion also drives the difference target's
    # per-reflection weighting, which needs no intensity data.
    mc.set_lambda_twin(args.lambda_twin, refinable=args.refine_lambda_twin)

    state = setup_loss_state(dc, mc, scaler, target_weights, device,
                             similarity_alpha=args.similarity_alpha,
                             two_moment=args.two_moment)

    if args.verbose > 0:
        print("Initial loss breakdown:")
        state.summary()
        print()
        sys.stdout.flush()

    total_rounds = args.n_cycles * len(weight_schedule)
    round_idx = 0

    if args.refine_fractions:
        params = list(itertools.chain(
            model_light.parameters(), mc.fraction_parameters()
        ))
    else:
        params = list(model_light.parameters())
        if args.refine_lambda_twin:
            # fraction_parameters() carries lambda once it is refinable; take only
            # that, since the fractions themselves stay frozen here.
            params.append(mc._lambda_logit)

    fraction_history = []
    if args.refine_fractions:
        fraction_history.append(mixed.fractions[1].detach().cpu().item())

    for cycle in range(args.n_cycles):
        for t_weight in weight_schedule:
            round_idx += 1
            if args.verbose > 0:
                frac_str = (
                    f", fractions={mixed.fractions.detach().cpu().tolist()}"
                    if args.refine_fractions else ""
                )
                print(
                    f"[{round_idx}/{total_rounds}] cycle {cycle + 1}/"
                    f"{args.n_cycles}, diff_weight={t_weight}{frac_str}"
                )
                sys.stdout.flush()

            state.set_weights({"xray/difference": t_weight})
            optimize_lbfgs(
                state, params,
                max_iter=args.max_iter,
                nsteps=args.n_steps,
                n_clean=args.n_clean,
                verbose=args.verbose,
            )

            # Update solvent masks and re-refine scaler jointly
            scaler.update_solvent()
            scaler.refine_lbfgs_joint(verbose=(args.verbose > 1))

            if args.verbose > 0:
                rw_d, rf_d = compute_rfactors(dark, data_dark, scaler)
                rw_l, rf_l = compute_rfactors(mixed, data_light, scaler)
                print(f"  R-factor (dark):  Rwork={rw_d:.4f}, Rfree={rf_d:.4f}")
                print(f"  R-factor (light): Rwork={rw_l:.4f}, Rfree={rf_l:.4f}")

            # Reset model caches after no_grad rfactor computation
            # to ensure gradients flow in the next optimisation round.
            model_light.reset_cache()
            model_dark.reset_cache()

            if args.refine_fractions:
                fraction_history.append(mixed.fractions[1].detach().cpu().item())

            if args.verbose > 1:
                state.summary()
                sys.stdout.flush()

    # --- Final statistics ---
    # Computed unconditionally: the deposition metadata and the results MTZ both carry
    # these, so they are not a reporting-only quantity.
    r_work_d, r_free_d = compute_rfactors(dark, data_dark, scaler)
    r_work_l, r_free_l = compute_rfactors(mixed, data_light, scaler)
    r_work_dl, r_free_dl = compute_rfactors(dark, data_light, scaler)

    if args.verbose > 0:
        print()
        print("=" * 72)
        print("Refinement complete")
        print("=" * 72)
        print(f"  Final R-factor (dark  vs dark data):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  Final R-factor (mixed vs light data): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print(f"  Final R-factor (dark  vs light data): R_work={r_work_dl:.4f}  R_free={r_free_dl:.4f}")
        print(f"  Refined fractions:      {mixed.fractions.detach().cpu().numpy()}")
        print()
        sys.stdout.flush()

    # --- Save outputs ---
    prefix = f"fractions_{int(fractions[0]*100)}_{int(fractions[1]*100)}"

    dark_pdb_out = str(outdir / f"{prefix}_dark.pdb")
    light_pdb_out = str(outdir / f"{prefix}_light.pdb")
    diff_mtz_out = str(outdir / f"{prefix}_difference_data.mtz")
    summary_path = str(outdir / f"{prefix}_summary.json")

    # Strip hydrogens for output (H were only needed for VDW restraints).
    # strip_hydrogens() returns new models with consistent pdb + tensors.
    model_dark = model_dark.strip_hydrogens()
    model_light = model_light.strip_hydrogens()

    no_header = getattr(args, "no_header", False)
    output_format = getattr(args, "output_format", "both")
    dark_meta = light_meta = None

    if not no_header:
        from torchref import __version__
        from torchref.io.metadata import RefinementMetadata

        def _build_metadata(model, data, r_work, r_free):
            """Build RefinementMetadata for a model/data pair."""
            meta = RefinementMetadata(
                program_version=__version__,
                refinement_method="difference-refine",
                r_work=float(r_work), r_free=float(r_free),
            )
            # Resolution (from masks, respects cutoff)
            if data.resolution is not None:
                valid = data.masks().to(torch.bool)
                res_valid = data.resolution[valid]
                if len(res_valid) > 0:
                    meta.resolution_high = float(res_valid.min())
                    meta.resolution_low = float(res_valid.max())

            # Reflection counts (standard work/free subset accessors: validity
            # masked, validation carved out of both).
            with torch.no_grad():
                if data.rfree_flags is not None:
                    n_work = data.work.n
                    n_test = data.free.n
                    n_all = n_work + n_test
                    meta.n_reflections_work = n_work
                    meta.n_reflections_test = n_test
                    meta.n_reflections_all = n_all
                    meta.percent_free = 100.0 * n_test / n_all if n_all > 0 else None

            # B-factor statistics
            pdb = model.pdb
            bvals = pdb["tempfactor"]
            meta.b_mean_overall = float(bvals.mean())
            meta.b_min = float(bvals.min())
            meta.b_max = float(bvals.max())

            # Atom counts
            meta.n_atoms_total = len(pdb)
            meta.n_atoms_protein = int((pdb["ATOM"] == "ATOM").sum())
            meta.n_atoms_solvent = int((pdb["ATOM"] == "HETATM").sum())

            # Geometry deviations
            if model.ctx.initialized and model._restraints is not None:
                restraints = model.restraints
                with torch.no_grad():
                    if hasattr(restraints, "bond_deviations"):
                        bond_devs, _ = restraints.bond_deviations()
                        meta.rmsd_bond_lengths = float(torch.sqrt((bond_devs**2).mean()))
                    if hasattr(restraints, "angle_deviations"):
                        angle_devs, _ = restraints.angle_deviations()
                        meta.rmsd_bond_angles = float(torch.sqrt((angle_devs**2).mean()))

            # Solvent model from CollectionScaler
            if hasattr(scaler, "solvent") and scaler.solvent is not None:
                sm = scaler.solvent
                meta.solvent_model_ksol = float(sm.k_solvent().detach())
                meta.solvent_model_bsol = sm.b_solvent_equivalent(scaler._s_half_sq)

            # Cell and spacegroup
            if model.cell is not None:
                meta.cell = [float(x) for x in model.cell.data.tolist()]
            if model.spacegroup is not None:
                meta.spacegroup = model.spacegroup.hm

            # CLI overrides / defaults
            if getattr(args, "title", None):
                meta.title = args.title
            meta.authors = getattr(args, "authors", None) or ["AUTHOR NAME"]

            return meta

        dark_meta = _build_metadata(model_dark, data_dark, r_work_d, r_free_d)
        light_meta = _build_metadata(model_light, data_light, r_work_l, r_free_l)

    if output_format in ("pdb", "both"):
        model_dark.write_pdb(dark_pdb_out, metadata=dark_meta)
        model_light.write_pdb(light_pdb_out, metadata=light_meta)
    if output_format in ("cif", "both"):
        dark_cif_out = str(outdir / f"{prefix}_dark.cif")
        light_cif_out = str(outdir / f"{prefix}_light.cif")
        model_dark.write_cif(dark_cif_out, metadata=dark_meta)
        model_light.write_cif(light_cif_out, metadata=light_meta)

    # --- Write merged deposition CIF (if no altlocs) ---
    merged_cif_out = str(outdir / f"{prefix}_merged.cif")
    has_altloc_dark = (model_dark.pdb["altloc"].astype(str).str.strip() != "").any()
    has_altloc_light = (model_light.pdb["altloc"].astype(str).str.strip() != "").any()

    if not has_altloc_dark and not has_altloc_light:
        import pandas as pd

        from torchref import __version__
        from torchref.io.metadata import RefinementMetadata

        dark_df = model_dark.pdb.copy()
        dark_df["altloc"] = "A"
        dark_df["occupancy"] = fractions[0]

        light_df = model_light.pdb.copy()
        light_df["altloc"] = "B"
        light_df["occupancy"] = fractions[1]

        merged_df = pd.concat([dark_df, light_df], ignore_index=True)
        merged_df = merged_df.sort_values(
            ["chainid", "resseq", "icode", "name", "altloc"]
        ).reset_index(drop=True)
        merged_df["serial"] = range(1, len(merged_df) + 1)
        merged_df.attrs["cell"] = model_dark.cell.data.tolist()
        merged_df.attrs["spacegroup"] = (
            model_dark.spacegroup.hm if model_dark.spacegroup else "P 1"
        )

        merged_meta = RefinementMetadata(
            program_version=__version__,
            refinement_method="difference-refine",
            r_work=float(r_work_l),
            r_free=float(r_free_l),
            authors=getattr(args, "authors", None) or ["AUTHOR NAME"],
        )
        if data_light.resolution is not None:
            valid = data_light.masks().to(torch.bool)
            res_valid = data_light.resolution[valid]
            if len(res_valid) > 0:
                merged_meta.resolution_high = float(res_valid.min())
                merged_meta.resolution_low = float(res_valid.max())
        merged_meta.b_mean_overall = float(merged_df["tempfactor"].mean())
        merged_meta.n_atoms_total = len(merged_df)

        if hasattr(scaler, "solvent") and scaler.solvent is not None:
            sm = scaler.solvent
            merged_meta.solvent_model_ksol = float(sm.k_solvent().detach())
            merged_meta.solvent_model_bsol = sm.b_solvent_equivalent(scaler._s_half_sq)

        ensemble_note = (
            f"Mixed-state ensemble from TorchRef difference refinement. "
            f"Conformer A (occupancy {fractions[0]:.2f}): dark/ground state. "
            f"Conformer B (occupancy {fractions[1]:.2f}): light/excited state. "
            f"R-factors: mixed vs light data Rwork={r_work_l:.4f} Rfree={r_free_l:.4f}; "
            f"dark vs dark data Rwork={r_work_d:.4f} Rfree={r_free_d:.4f}."
        )
        merged_meta.title = ensemble_note

        from torchref.io import cif as cif_io
        cif_io.write_model(merged_df, merged_cif_out, metadata=merged_meta)

        if args.verbose > 0:
            print(f"  Merged deposition CIF written to {merged_cif_out}")
    else:
        if args.verbose > 0:
            print("  Skipping merged CIF: input models contain altlocs")
        merged_cif_out = None

    # --- Write per-dataset structure factor files (MTZ + CIF) ---
    dark_sf_mtz = str(outdir / f"{prefix}_dark-sf.mtz")
    light_sf_mtz = str(outdir / f"{prefix}_light-sf.mtz")
    dark_sf_cif = str(outdir / f"{prefix}_dark-sf.cif")
    light_sf_cif = str(outdir / f"{prefix}_light-sf.cif")

    with torch.no_grad():
        fcalc_dark_full = scaler.forward_mixed(
            dark.models[0](data_dark.hkl), dark.fractions
        )
        fcalc_light_full = scaler.forward_mixed(
            mixed(data_light.hkl), mixed.fractions
        )
    data_dark.write_mtz(dark_sf_mtz, fcalc=fcalc_dark_full)
    data_light.write_mtz(light_sf_mtz, fcalc=fcalc_light_full)

    def _mtz_to_cif(mtz_path, cif_path):
        """Convert MTZ to mmCIF structure factor file via gemmi."""
        import gemmi
        mtz = gemmi.read_mtz_file(mtz_path)
        m2c = gemmi.MtzToCif()
        cif_str = m2c.write_cif_to_string(mtz)
        with open(cif_path, "w") as f:
            f.write(cif_str)

    _mtz_to_cif(dark_sf_mtz, dark_sf_cif)
    _mtz_to_cif(light_sf_mtz, light_sf_cif)

    if args.verbose > 0:
        print(f"  Dark SF written to {dark_sf_mtz}, {dark_sf_cif}")
        print(f"  Light SF written to {light_sf_mtz}, {light_sf_cif}")

    map_diagnostics = write_results_mtz(
        dc, mc.dark_model, scaler, diff_mtz_out,
        mc=mc, all_columns=args.all_columns, verbose=args.verbose,
    )

    # --- JSON summary ---
    summary = {
        "input": {
            "dark_model": args.dark_model,
            "light_model": args.light_model,
            "dark_structure_factor": args.dark_structure_factor,
            "light_structure_factor": args.light_structure_factor,
            "fractions": fractions,
            "cif": args.cif,
            "dmin": args.dmin,
        },
        "dataset_scaling": dc.scaling_metrics,
        "parameters": {
            "weight_schedule": weight_schedule,
            "n_cycles": args.n_cycles,
            "n_steps": args.n_steps,
            "max_iter": args.max_iter,
            "weights": target_weights,
        },
        "results": {
            "r_factor_dark": dict(
                zip(
                    ["r_work", "r_free"],
                    compute_rfactors(dark, data_dark, scaler),
                )
            ),
            "r_factor_light": dict(
                zip(
                    ["r_work", "r_free"],
                    compute_rfactors(mixed, data_light, scaler),
                )
            ),
            "fractions": mixed.fractions.detach().cpu().tolist(),
            "alpha_mean": float(mc.alpha_mean),
            "lambda_twin": float(mc.lambda_twin),
            "sigma_alpha_sq": float(mc.sigma_alpha_sq),
            **map_diagnostics,
        },
        "output_files": {
            "dark_pdb": dark_pdb_out,
            "light_pdb": light_pdb_out,
            "dark_sf_mtz": dark_sf_mtz,
            "dark_sf_cif": dark_sf_cif,
            "light_sf_mtz": light_sf_mtz,
            "light_sf_cif": light_sf_cif,
            "difference_mtz": diff_mtz_out,
            "merged_cif": merged_cif_out,
            "summary": summary_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(convert_to_serializable(summary), f, indent=2)

    if args.verbose > 0:
        print("Output files:")
        print(f"  - {dark_pdb_out}")
        print(f"  - {light_pdb_out}")
        if merged_cif_out:
            print(f"  - {merged_cif_out}")
        print(f"  - {dark_sf_mtz} / {dark_sf_cif}")
        print(f"  - {light_sf_mtz} / {light_sf_cif}")
        print(f"  - {diff_mtz_out}")
        print(f"  - {summary_path}")
        print()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
