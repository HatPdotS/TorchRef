"""
LBFGS refinement of an ~100-member ensemble against an X-ray dataset.

Composes:

- :class:`~torchref.experimental.ensemble.ensemble_model.EnsembleModel` for the model
  (B-factor-free, equal occupancy, ``n_members`` xyz copies).
- :class:`~torchref.experimental.ensemble.wilson_prior.WilsonPriorTarget`
  to keep ``<|F_calc|^2>`` on the Wilson curve.
- Optional :class:`~torchref.experimental.ensemble.ensemble_amber_kl.EnsembleAmberKLTarget`
  for Amber-Boltzmann KL restraints (enabled with ``kT > 0``).
- A third ``xray/validation`` set distinct from R-free for tuning
  ``wilson_weight`` / ``amber_lambda`` without contaminating R-free.

Geometry and ADP targets are intentionally not registered — they would
collapse the ensemble. Restraints come from the Wilson prior + Amber KL.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, Optional

import torch

from torchref.io.datasets import ReflectionData
from .ensemble_model import EnsembleModel
from torchref.refinement.lbfgs_refinement import LBFGSRefinement
from torchref.refinement.loss_state import LossState
from .rank_penalty import RankPenaltyTarget
from .wilson_prior import WilsonPriorTarget
from torchref.refinement.targets.xray.maximum_likelihood import create_xray_target
from torchref.scaling import Scaler

try:
    from .ensemble_amber_kl import EnsembleAmberKLTarget
except ImportError:
    EnsembleAmberKLTarget = None

try:
    from .quasi_crystal_amber import (
        QuasiCrystalAmberTarget,
    )
except ImportError:
    QuasiCrystalAmberTarget = None


def _extract_first_model_pdb(pdb_path: str):
    """Return a path to a single-model copy of ``pdb_path`` (header + MODEL 1).

    The base :class:`Refinement` ``__init__`` loads ``pdb`` into a *throwaway*
    single ``Model`` (it's replaced by the EnsembleModel right after) and eagerly
    builds geometry restraints on it. If ``pdb`` is a multi-MODEL ensemble, that
    base model would hold every member's atoms (96 × 1329 + symmetry expansion),
    and the VDW neighbor search there allocates tens of GB (a padded ``cdist``
    over hugely over-occupied grid cells) → OOM. Feeding the base pre-pass just
    MODEL 1 keeps that scaffolding cheap; the real ensemble is still built from
    the full file via ``from_multimodel_pdb``.

    Returns ``(path, is_temp)``. When the file has no MODEL records, returns the
    original path unchanged (``is_temp=False``).
    """
    with open(pdb_path, "r") as f:
        text = f.read()
    if "MODEL " not in text:
        return pdb_path, False
    lines = text.splitlines(keepends=True)
    out: list = []
    i = 0
    while i < len(lines) and not lines[i].startswith("MODEL "):
        out.append(lines[i])  # header (CRYST1, REMARKs)
        i += 1
    for ln in lines[i:]:
        out.append(ln)
        if ln.startswith("ENDMDL"):
            break
    out.append("END\n")
    fd, tmp = tempfile.mkstemp(suffix=".pdb", prefix="ens_base_model1_")
    with os.fdopen(fd, "w") as f:
        f.writelines(out)
    return tmp, True


class EnsembleRefinement(LBFGSRefinement):
    """
    LBFGS ensemble refinement with Wilson + Amber-KL regularization.

    Parameters
    ----------
    data_file : str
        Path to MTZ.
    pdb : str
        Path to single- or multi-MODEL PDB. If multi-MODEL with at least
        ``n_members`` models, those are used as-is; otherwise members are
        replicated cyclically and perturbed.
    n_members : int
        Number of ensemble members. Default 100.
    perturb_sigma : float
        Std-dev (Å) of Gaussian noise applied to replicated members.
        Default 0.01 Å — only large enough to break gradient degeneracy
        between identical copies and to clear the entropy-eps floor in
        :class:`EnsembleAmberKLTarget` (which uses ``log(var + 1e-4)``).
        Larger values (≥ ~0.05 Å) introduce LJ clashes when atoms walk
        inside each other's vdW radii, producing ~10^15 kJ/mol Amber
        energies. The ensemble's actual disorder develops from the X-ray
        + entropy gradients during refinement, not from initial noise.
    b_const : float
        Fixed isotropic B (Å²) for every atom in every member. Small but
        non-zero to avoid FFT grid aliasing.
    wilson_weight : float
        Weight on the Wilson prior in the LossState.
    amber_lam : float
        Coefficient on the ensemble entropy regularizer inside
        :class:`EnsembleAmberKLTarget`.
    amber_kT : float
        Boltzmann temperature for Amber KL (kJ/mol). 0 disables the energy
        term and uses entropy only.
    val_fraction_of_free : float
        If the loaded MTZ has only an R-free flag and no Validation_flag,
        split this fraction of the free set into a held-out validation set.
    wilson_weight : float
        Weight on the (per-bin-mean) Wilson prior. O(1) because all loss
        terms are normalized to a per-item scale (see ``_create_loss_state``).
    amber_weight : float
        Weight on the (per-atom) Amber-KL restraint. O(1), same rationale.
    xray_mode : str
        Mode for the X-ray targets. Default ``'ml'`` — the maximum-
        likelihood target, which is the Rice distribution NLL for acentric
        reflections (folded-normal for centrics), the statistically correct
        amplitude likelihood. Other options: ``'ls'``, ``'gaussian'``,
        ``'bhattacharyya'``.
    """

    def __init__(
        self,
        data_file: str = None,
        pdb: str = None,
        n_members: int = 100,
        perturb_sigma: float = 0.01,
        b_const: float = 5.0,
        wilson_weight: float = 1.0,
        wilson_mode: str = "rice",
        xray_weight: float = 1.0,
        amber_weight: float = 1.0,
        amber_lam: float = 1.0,
        amber_kT: float = 0.0,
        amber_charge_method: str = "gas",
        amber_relax_on_init: bool = True,
        amber_force_clamp: float = 10000.0,
        low_rank_modes: int = 0,
        rank_weight: float = 0.0,
        rank_weight_start: Optional[float] = None,
        rank_penalty_mode: str = "nuclear",
        rank_target_rank: int = 0,
        rank_freeze_disp: float = 0.2,
        maxent_shrink_weight: float = 1.0,
        maxent_div_weight: float = 1.0,
        rank_adaptive: bool = False,
        rank_adaptive_base: float = 1.0,
        rank_adaptive_doubling_factor: float = 2.0,
        val_fraction_of_free: float = 0.5,
        xray_mode: str = "ml",
        adam_lr: float = 1e-3,
        optimizer_name: str = "adam",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_steps_per_cycle: int = 50,
        warmup_steps: int = 100,
        warmup_start_factor: float = 0.02,
        lr_schedule: str = "cosine",
        lr_cycles: int = 1,
        wilson_weight_start: Optional[float] = None,
        use_dropout: bool = False,
        dropout_min: int = 10,
        dropout_max: int = 40,
        langevin_T: float = 0.0,
        sampling_fraction: float = 0.0,
        sampling_lr_factor: float = 0.2,
        noise_floor_sigma: float = 0.0,
        noise_floor_amp: float = 0.0,
        noise_floor_cycles: float = 1.0,
        xray_adaptive: bool = False,
        xray_adaptive_floor: float = 1e-4,
        xray_adaptive_ema_halflife_steps: int = 50,
        xray_adaptive_doubling_factor: float = 5.0,
        amber_every: int = 1,
        refine_population: bool = False,
        refine_member_b: bool = False,
        n_max: int = 0,
        death_rate: float = 0.0,
        birth_rate: float = 0.0,
        bifurcation_sigma: float = 0.1,
        birth_death_every: int = 1,
        seed: Optional[int] = None,
        verbose: int = 1,
        device=None,
        max_res: float = None,
        nbins: int = 10,
        column_names: Optional[Dict[str, str]] = None,
    ):
        # Empty-shell construction for state_dict loading.
        if data_file is None and pdb is None:
            super().__init__(verbose=verbose, device=device, nbins=nbins)
            self.n_members = n_members
            self.wilson_weight = wilson_weight
            self.wilson_mode = wilson_mode
            self.xray_weight = xray_weight
            self.amber_weight = amber_weight
            self.amber_lam = amber_lam
            self.amber_kT = amber_kT
            self.adam_lr = adam_lr
            self.optimizer_name = optimizer_name
            self.adam_beta1 = adam_beta1
            self.adam_beta2 = adam_beta2
            self.adam_steps_per_cycle = adam_steps_per_cycle
            self.warmup_steps = warmup_steps
            self.warmup_start_factor = warmup_start_factor
            self.lr_schedule = lr_schedule
            self.lr_cycles = lr_cycles
            self.wilson_weight_start = wilson_weight_start
            self.use_dropout = use_dropout
            self.dropout_min = dropout_min
            self.dropout_max = dropout_max
            self.langevin_T = langevin_T
            self.sampling_fraction = sampling_fraction
            self.sampling_lr_factor = sampling_lr_factor
            self.noise_floor_sigma = noise_floor_sigma
            self.noise_floor_amp = noise_floor_amp
            self.noise_floor_cycles = noise_floor_cycles
            self.xray_adaptive = xray_adaptive
            self.xray_adaptive_floor = xray_adaptive_floor
            self.xray_adaptive_ema_halflife_steps = xray_adaptive_ema_halflife_steps
            self.xray_adaptive_doubling_factor = xray_adaptive_doubling_factor
            self.amber_every = amber_every
            self.refine_population = refine_population
            self.refine_member_b = refine_member_b
            self.n_max = n_max
            self.death_rate = death_rate
            self.birth_rate = birth_rate
            self.bifurcation_sigma = bifurcation_sigma
            self.birth_death_every = birth_death_every
            self.amber_charge_method = amber_charge_method
            self.amber_relax_on_init = amber_relax_on_init
            self.amber_force_clamp = amber_force_clamp
            self.low_rank_modes = low_rank_modes
            self.rank_weight = rank_weight
            self.rank_weight_start = rank_weight_start
            self.rank_penalty_mode = rank_penalty_mode
            self.rank_target_rank = rank_target_rank
            self.rank_freeze_disp = rank_freeze_disp
            self.maxent_shrink_weight = maxent_shrink_weight
            self.maxent_div_weight = maxent_div_weight
            self.rank_adaptive = rank_adaptive
            self.rank_adaptive_base = rank_adaptive_base
            self.rank_adaptive_doubling_factor = rank_adaptive_doubling_factor
            return

        # Build the standard scaffolding (data, single-copy model, scaler,
        # standard targets). We then *replace* self.model with the ensemble
        # and re-register only the targets we want.
        #
        # Feed the base pre-pass only MODEL 1 of (a possibly multi-MODEL) pdb:
        # the base builds a throwaway single Model + eager geometry restraints,
        # and doing that on a full ensemble file (all members + sym expansion)
        # OOMs in the VDW neighbor search. The real ensemble is built from the
        # full file below via from_multimodel_pdb.
        base_pdb, _base_pdb_is_temp = _extract_first_model_pdb(pdb)
        try:
            super().__init__(
                data_file=data_file,
                pdb=base_pdb,
                target_mode=xray_mode,
                verbose=verbose,
                device=device,
                max_res=max_res,
                nbins=nbins,
                column_names=column_names,
            )
        finally:
            if _base_pdb_is_temp:
                try:
                    os.unlink(base_pdb)
                except OSError:
                    pass

        self.n_members = int(n_members)
        self.wilson_weight = float(wilson_weight)
        self.wilson_mode = str(wilson_mode)
        self.xray_weight = float(xray_weight)
        self.amber_weight = float(amber_weight)
        self.amber_lam = float(amber_lam)
        self.amber_kT = float(amber_kT)
        self.amber_charge_method = amber_charge_method
        self.amber_relax_on_init = bool(amber_relax_on_init)
        self.amber_force_clamp = float(amber_force_clamp)
        self.low_rank_modes = int(low_rank_modes)
        self.rank_weight = float(rank_weight)
        self.rank_weight_start = (
            float(rank_weight_start) if rank_weight_start is not None else None
        )
        self.rank_penalty_mode = str(rank_penalty_mode)
        self.rank_target_rank = int(rank_target_rank)
        self.rank_freeze_disp = float(rank_freeze_disp)
        self.maxent_shrink_weight = float(maxent_shrink_weight)
        self.maxent_div_weight = float(maxent_div_weight)
        self.rank_adaptive = bool(rank_adaptive)
        self.rank_adaptive_base = float(rank_adaptive_base)
        self.rank_adaptive_doubling_factor = float(rank_adaptive_doubling_factor)
        self.adam_lr = float(adam_lr)
        self.optimizer_name = str(optimizer_name).lower()
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_steps_per_cycle = int(adam_steps_per_cycle)
        self.warmup_steps = int(warmup_steps)
        self.warmup_start_factor = float(warmup_start_factor)
        self.lr_schedule = str(lr_schedule)
        self.lr_cycles = int(lr_cycles)
        self.wilson_weight_start = (
            float(wilson_weight_start) if wilson_weight_start is not None else None
        )
        self.use_dropout = bool(use_dropout)
        self.dropout_min = int(dropout_min)
        self.dropout_max = int(dropout_max)
        self.langevin_T = float(langevin_T)
        self.sampling_fraction = float(sampling_fraction)
        self.sampling_lr_factor = float(sampling_lr_factor)
        self.noise_floor_sigma = float(noise_floor_sigma)
        self.noise_floor_amp = float(noise_floor_amp)
        self.noise_floor_cycles = float(noise_floor_cycles)
        self.xray_adaptive = bool(xray_adaptive)
        self.xray_adaptive_floor = float(xray_adaptive_floor)
        self.xray_adaptive_ema_halflife_steps = int(xray_adaptive_ema_halflife_steps)
        self.xray_adaptive_doubling_factor = float(xray_adaptive_doubling_factor)
        self.amber_every = max(1, int(amber_every))
        self.refine_population = bool(refine_population)
        self.refine_member_b = bool(refine_member_b)
        self.n_max = int(n_max)
        self.death_rate = float(death_rate)
        self.birth_rate = float(birth_rate)
        self.bifurcation_sigma = float(bifurcation_sigma)
        self.birth_death_every = max(1, int(birth_death_every))

        # --- N % N_sym == 0 ---
        # The quasi-crystal Amber target lays out N members as
        # ``n_disorder × N_sym`` (disorder copies × spacegroup sym mates),
        # so n_members must be divisible by N_sym. Round down with a warning
        # rather than fail, so the driver's user-facing N input stays loose.
        n_sym = int(self.reflection_data.spacegroup.n_ops)
        if self.n_members % n_sym != 0:
            new_N = (self.n_members // n_sym) * n_sym
            if new_N < n_sym:
                # Would round to 0 — bump up to one full sym expansion.
                new_N = n_sym
            if verbose > 0:
                print(
                    f"EnsembleRefinement: spacegroup "
                    f"'{self.reflection_data.spacegroup.short_name}' has "
                    f"N_sym={n_sym}; requested n_members={self.n_members} "
                    f"is not a multiple — rounding to {new_N} "
                    f"(n_disorder={new_N // n_sym} per crystallographic site)."
                )
            self.n_members = new_N
        # Birth/death dynamics use a FIXED pool of member slots: the requested
        # ``n_members`` is the initial ALIVE count, ``n_max`` (>= it) the pool
        # size. The amber supercell lays out the FULL pool, so it must remain a
        # multiple of N_sym; the alive subset (x-ray only) is unconstrained.
        # With n_max <= n_members (default), pool == alive — identical to the
        # fixed-ensemble path; bifurcation then only reuses slots freed by deaths.
        n_alive0 = self.n_members
        pool = n_alive0
        if self.refine_population and self.n_max > n_alive0:
            pool = max(n_alive0, (self.n_max // n_sym) * n_sym)
        self._n_alive_init = int(n_alive0)
        self.n_members = int(pool)
        self.n_disorder = self.n_members // n_sym

        # Ensure the reflection data has a validation set.
        if self.reflection_data.validation.n == 0:
            if self.reflection_data.free.n >= 4:
                if verbose > 0:
                    print("EnsembleRefinement: generating validation set from R-free")
                self.reflection_data.generate_validation_set(
                    val_fraction_of_free=val_fraction_of_free, seed=seed,
                )

        # Replace the single-model with an EnsembleModel built from the same PDB.
        # ``n_members`` here is the initial alive count; ``n_max`` the pool.
        self.model = EnsembleModel.from_multimodel_pdb(
            pdb,
            n_members=self._n_alive_init,
            n_max=self.n_members,
            perturb_sigma=perturb_sigma,
            b_const=b_const,
            seed=seed,
            verbose=verbose,
            device=self.device,
            max_res=self.max_res,
        )
        if self.refine_population:
            self.model.enable_population_refinement(
                True, refine_b=self.refine_member_b
            )
            if verbose > 0:
                b_msg = "free B_m" if self.refine_member_b else "B frozen@b_const"
                print(
                    f"EnsembleRefinement: population refinement ON — "
                    f"per-member occupancy(softmax), {b_msg}, pool={self.model.n_members} "
                    f"slots, {self.model.n_alive} alive, death_rate={self.death_rate}, "
                    f"birth_rate={self.birth_rate}, σ_bif={self.bifurcation_sigma}, "
                    f"every {self.birth_death_every} cyc."
                )
        self.model.cell = self.reflection_data.cell
        self.model.spacegroup = self.reflection_data.spacegroup
        self.model.setup_grid(max_res=self.max_res)

        # Rebuild the scaler against the ensemble model.
        self.scaler = Scaler(
            self.model, self.reflection_data,
            verbose=self.verbose, device=self.device, nbins=self.nbins,
        )
        with torch.no_grad():
            fcalc0 = self.model(self.reflection_data.hkl)
        self.scaler.initialize(fcalc0)

        # Re-create our targets pointing at the ensemble model + new scaler.
        self._init_targets(xray_mode=xray_mode)
        # Force loss-state rebuild on next access.
        self.reset_loss_state()

    # ------------------------------------------------------------------
    # Target wiring
    # ------------------------------------------------------------------

    def _init_targets(self, xray_mode: str = "ml"):
        """Register X-ray (work/free/val), Wilson, and optional Amber-KL targets."""
        # Don't error if the ensemble pieces aren't built yet (during the
        # base-class super().__init__ pre-pass). Detect by checking for
        # EnsembleModel.
        if not isinstance(getattr(self, "model", None), EnsembleModel):
            super()._init_targets(xray_mode=xray_mode)
            return

        self.xray_target_work = create_xray_target(
            data=self.reflection_data, model=self.model, scaler=self.scaler,
            mode=xray_mode, use_set="work", verbose=self.verbose,
        )
        self.xray_target_test = create_xray_target(
            data=self.reflection_data, model=self.model, scaler=self.scaler,
            mode=xray_mode, use_set="free", verbose=self.verbose,
        )
        self.xray_target_validation = create_xray_target(
            data=self.reflection_data, model=self.model, scaler=self.scaler,
            mode=xray_mode, use_set="val", verbose=self.verbose,
        )
        self.wilson_target = WilsonPriorTarget(
            data=self.reflection_data, model=self.model, scaler=self.scaler,
            mode=self.wilson_mode,
        )
        # Quasi-crystal Amber: one unified OpenMM System (k·N_sym replicas
        # with PBC + PME), no per-member loop, no KL/entropy term — physical
        # crystal contacts in the supercell are the regularizer. Falls back
        # to "disabled" if either OpenMM is missing or amber_weight == 0.
        if QuasiCrystalAmberTarget is not None and self.amber_weight > 0.0:
            self.amber_target = QuasiCrystalAmberTarget(
                model=self.model,
                cell=self.reflection_data.cell,
                spacegroup=self.reflection_data.spacegroup,
                n_disorder=self.n_disorder,
                charge_method=self.amber_charge_method,
                relax_on_init=self.amber_relax_on_init,
                force_clamp=self.amber_force_clamp,
                verbose=self.verbose,
            )
        else:
            self.amber_target = None

        # Soft rank penalty (nuclear norm of the centered member matrix):
        # "purifies" the ensemble toward fewer effective disorder modes. Built
        # whenever a non-zero weight (or a ramp start) is requested. See
        # RankPenaltyTarget — frozen-basis PCA failed because the disorder is
        # high-rank, so we penalize rank softly instead of truncating it.
        # "maxent" and "diverse" carry their coefficients (shrink/div, i.e.
        # participation/similarity) INTERNALLY, so they register at loss-state
        # weight 1.0 and are "on" whenever either internal coef is non-zero.
        _maxent_on = self.rank_penalty_mode in ("maxent", "diverse") and (
            self.maxent_shrink_weight != 0.0 or self.maxent_div_weight != 0.0
        )
        if (self.rank_weight > 0.0) or (self.rank_weight_start is not None) or _maxent_on:
            self.rank_target = RankPenaltyTarget(
                model=self.model,
                mode=self.rank_penalty_mode,
                target_rank=self.rank_target_rank,
                freeze_disp=self.rank_freeze_disp,
                maxent_shrink=self.maxent_shrink_weight,
                maxent_div=self.maxent_div_weight,
                verbose=self.verbose,
            )
        else:
            self.rank_target = None

        # Set up the component weighting if it exists in the base class.
        if hasattr(self, "setup_component_weighting"):
            try:
                self.setup_component_weighting()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Low-rank reparameterization
    # ------------------------------------------------------------------

    def enable_low_rank(self, K: int) -> float:
        """Project the ensemble onto its top-``K`` PCA modes (frozen basis).

        Delegates the basis computation + ``self.model.xyz`` swap to
        :meth:`EnsembleModel.enable_low_rank` (the only refinable xyz leaf
        becomes the ``(N, K)`` amplitudes), then prints post-swap R-factors
        and per-ASU Amber energy as a verification hook. The scaler is left
        untouched — it re-refines each cycle and clobbering it would discard a
        branched-in scaler. Returns the explained-variance fraction.
        """
        explained = self.model.enable_low_rank(K)
        with torch.no_grad():
            self.model.reset_cache()
            rw, rf = self.get_rfactor()
            amb = None
            if getattr(self, "amber_target", None) is not None:
                amb = float(self.amber_target.forward())
        if self.verbose > 0:
            print(
                f"[low-rank] K={int(K)} enabled "
                f"(explained var {explained * 100:.2f}%); "
                f"post-swap R_work={float(rw):.4f} R_free={float(rf):.4f} "
                f"amber/ASU={amb}",
                flush=True,
            )
        return explained

    def enable_pca(self, K=None) -> float:
        """Reparameterize the ensemble in fully-refinable PCA space (μ, A, V).

        Delegates to :meth:`EnsembleModel.enable_pca`; ``refine()`` detects the
        ``PCAEnsembleParam`` and routes all three factors into the optimizer.
        ``K=None`` → full rank N-1. Prints a post-swap verification line.
        """
        explained = self.model.enable_pca(K)
        with torch.no_grad():
            self.model.reset_cache()
            rw, rf = self.get_rfactor()
            amb = None
            if getattr(self, "amber_target", None) is not None:
                amb = float(self.amber_target.forward())
        if self.verbose > 0:
            print(
                f"[pca] K={self.model.xyz.K} enabled (refine μ,A,V; "
                f"explained var {explained * 100:.2f}%); "
                f"post-swap R_work={float(rw):.4f} R_free={float(rf):.4f} "
                f"amber/ASU={amb}",
                flush=True,
            )
        return explained

    # ------------------------------------------------------------------
    # LossState construction
    # ------------------------------------------------------------------

    def _create_loss_state(self) -> LossState:
        """
        Register the ensemble targets on a single "per-ASU" loss scale.

        Every term is the loss attributable to one asymmetric unit's worth
        of structure:

        - **xray/work** — the Rice (ML) target returns ``sum_NLL`` over the
          observed reflections (the dataset describes one crystal, which is
          built from one ASU via the spacegroup). Units: nats / ASU.
        - **regularization/wilson** (rice mode) — same: ``sum_NLL`` over the
          observed reflections. Units: nats / ASU.
        - **restraints/amber_kl** — :class:`QuasiCrystalAmberTarget` with
          ``normalize_per_asu=True`` returns supercell energy divided by
          the number of ASU copies it contains. Units: kJ/mol / ASU.

        With every term on a per-ASU scale, the user-facing weights
        (``xray_weight``, ``wilson_weight``, ``amber_weight``) are
        dimensionless multipliers applied literally. Default values of 1.0
        give a meaningful physical balance; lowering ``xray_weight`` is the
        primary knob for resisting late-cycle work-set overfit drift.
        """
        state = LossState(device=self.device)

        state.register_target("xray/work", self.xray_target_work)
        state.register_target("xray/free", self.xray_target_test)
        state.register_target("xray/validation", self.xray_target_validation)
        # Work set drives refinement; free/val tracked for monitoring only.
        state.set_weight("xray/work", self.xray_weight)
        state.set_weight("xray/free", 0.0)
        state.set_weight("xray/validation", 0.0)

        state.register_target("regularization/wilson", self.wilson_target)
        state.set_weight("regularization/wilson", self.wilson_weight)

        if self.amber_target is not None:
            state.register_target("restraints/amber_kl", self.amber_target)
            state.set_weight("restraints/amber_kl", self.amber_weight)

        if getattr(self, "rank_target", None) is not None:
            state.register_target("regularization/rank", self.rank_target)
            # maxent carries its shrink/div coefficients INTERNALLY, so the
            # loss-state weight is a constant 1.0. Other modes use the
            # (possibly ramped) rank_weight; refine() updates it via _rank_at.
            if self.rank_penalty_mode in ("maxent", "diverse"):
                w0 = 1.0
            else:
                w0 = (
                    self.rank_weight_start
                    if self.rank_weight_start is not None
                    else self.rank_weight
                )
            state.set_weight("regularization/rank", w0)
        return state

    def complete_loss_state(self) -> LossState:
        """
        Refresh meta + cached losses but keep our explicit static weights.

        The base-class implementation calls ``update_weights`` →
        ``component_weighting`` which (a) overwrites the per-target weights
        and (b) clips them to [0.01, 100]. Both would destroy our
        per-ASU normalization. EnsembleRefinement uses fixed, user-set
        weights from ``_create_loss_state``, so we skip the component-
        weighting step entirely.
        """
        state = self.loss_state
        state.cache_losses()
        return state

    # ------------------------------------------------------------------
    # Birth–death population dynamics
    # ------------------------------------------------------------------

    def _reset_optimizer_rows(self, optimizer, member_idx: int) -> None:
        """Zero the Adam moments for a reborn member slot's parameters.

        A bifurcated child reuses a previously-dead slot whose optimizer state
        still holds stale momentum from its prior life; clear it so the newborn
        starts clean. Shapes never change — pure in-place indexing into the
        per-parameter ``exp_avg``/``exp_avg_sq`` buffers.
        """
        model = self.model
        n_at = int(model.n_atoms_per_member)
        rows = slice(member_idx * n_at, (member_idx + 1) * n_at)
        targets = [
            (model.xyz.refinable_params, rows),
            (model.occ_logits, member_idx),
            (model.b_raw, member_idx),
        ]
        for p, idx in targets:
            st = optimizer.state.get(p, None)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                buf = st.get(key, None)
                if buf is not None:
                    buf[idx] = 0.0

    @torch.no_grad()
    def _birth_death_step(self, optimizer) -> dict:
        """One probabilistic birth–death sweep over the member population.

        Death: each alive member dies with probability rising as its occupancy
        falls below the alive-median (``death_rate·(1−w/median)₊``); the last
        member is never killed. Bifurcation: alive survivors split (child = a
        perturbed copy sharing the parent's weight) with probability
        ``birth_rate·(w/w_max)`` into free (dead) slots. Coordinates / shapes
        never change — only the ``alive`` mask and reborn-slot contents.
        """
        model = self.model
        w = model.member_weights()
        alive = model._alive
        alive_idx = alive.nonzero(as_tuple=False).flatten().tolist()
        n_alive = len(alive_idx)
        if n_alive <= 1:
            return {"n_alive": n_alive, "died": 0, "born": 0}
        w_alive = w[alive]
        w_med = float(w_alive.median())
        w_max = float(w_alive.max())
        died: list = []
        for m in alive_idx:
            if n_alive - len(died) <= 1:
                break
            wm = float(w[m])
            p = self.death_rate * max(0.0, 1.0 - wm / max(w_med, 1e-30))
            if p > 0.0 and float(torch.rand(())) < min(p, 1.0):
                if model.kill_member(m):
                    died.append(m)
        born: list = []
        if self.birth_rate > 0.0:
            survivors = [m for m in alive_idx if m not in died]
            n_free = int((~model._alive).sum().item())
            for m in survivors:
                if n_free <= 0:
                    break
                wm = float(w[m])
                p = self.birth_rate * (wm / max(w_max, 1e-30))
                if float(torch.rand(())) < min(p, 1.0):
                    d = model.bifurcate_member(m, sigma=self.bifurcation_sigma)
                    if d >= 0:
                        self._reset_optimizer_rows(optimizer, d)
                        born.append(d)
                        n_free -= 1
        if died or born:
            model.reset_cache()
        return {"n_alive": model.n_alive, "died": len(died), "born": len(born)}

    # ------------------------------------------------------------------
    # Refinement loop
    # ------------------------------------------------------------------

    def refine(
        self,
        macro_cycles: int = 5,
        snapshot_every: int = 0,
        on_snapshot=None,
        checkpoint_path: Optional[str] = None,
        resume_state: Optional[dict] = None,
    ):
        """
        Macro-cycle loop with Adam over ``xyz + scaler`` parameters.

        Parameters
        ----------
        macro_cycles : int
            Number of macro cycles (each is ``adam_steps_per_cycle`` Adam steps).
        snapshot_every : int, optional
            If > 0 and ``on_snapshot`` is given, invoke ``on_snapshot`` every
            this many macro cycles. Lets long runs persist intermediate state
            (structure + metrics) so a kill mid-run doesn't lose everything and
            the trajectory can be inspected. 0 disables (default).
        on_snapshot : callable, optional
            ``on_snapshot(completed_cycles: int, hist: dict) -> None``. Receives
            the same history dict shape as the final return, truncated to the
            cycles completed so far (``sampling_summary`` is None mid-run). The
            caller owns serialization (e.g. write a checkpoint PDB + JSON).
        checkpoint_path : str, optional
            If given (with ``snapshot_every > 0``), write a full *resume*
            checkpoint here every ``snapshot_every`` cycles (atomically, single
            rolling file). Captures model + scaler + Adam optimizer state +
            ``global_step`` + lr/noise schedule position + adaptive-xray EMA +
            RNG + histories — everything needed to continue with identical
            dynamics. The PDB/JSON snapshot is for humans; this is for resume.
        resume_state : dict, optional
            A checkpoint dict (``torch.load`` of a prior ``checkpoint_path``,
            mapped to this model's device) to resume from. Restores all of the
            above and continues the macro-cycle loop from where it stopped. The
            schedule (``macro_cycles`` × ``adam_steps_per_cycle``) must match the
            original run, else the lr/noise curves would misalign — validated.

        Why Adam, not LBFGS: the loss landscape under (a) the
        ensemble-coordinate redundancy, (b) the Wilson regularizer, and
        (c) the per-member Amber energy is highly non-convex with
        ~30k+ xyz parameters per macro cycle. LBFGS's quasi-Newton
        curvature approximation is wrong for this kind of landscape
        and gets stuck. Adam's per-parameter step sizing + momentum is
        the standard choice for ensemble / variational refinement.

        Driven through :meth:`LossState.run`, which already handles:
        non-finite-loss validation, ``requires_grad`` toggling on leaves
        outside the optimizer's intent set, cache resets, and target
        maintenance hooks. Adam accepts the same ``closure(...) -> loss``
        contract LossState builds for LBFGS.

        One macro cycle = ``self.adam_steps_per_cycle`` Adam steps.
        B-factors and occupancies are frozen at construction time, so
        the optimizer only touches xyz and scaler parameters.

        Two schedules run over the *global* step index (0 .. T-1, with
        T = macro_cycles * adam_steps_per_cycle):

        - **LR** (``lr_schedule``):
          - ``'cosine'`` — one (or ``lr_cycles``) ``(1-cos)/2`` bump(s).
            LR starts at ``warmup_start_factor * adam_lr``, rises to
            ``adam_lr`` at the cycle midpoint, anneals back to the floor
            by the cycle end. Starting low doubles as warmup (Adam's
            early bias-corrected second moment is unstable); annealing
            low at the end gives gentle convergence.
          - ``'warmup'`` — legacy linear warmup then constant.
        - **Wilson weight** (if ``wilson_weight_start`` is set): ramps
          linearly from ``wilson_weight_start`` to ``wilson_weight`` over
          the run. Curriculum: fit the data first, then progressively
          tighten the Wilson prior as the ensemble starts to overfit.
        """
        # PCAEnsembleParam refines THREE leaves (μ, A, V); parameters_of_types
        # returns only the single `.refinable_params`, so collect all of them.
        from .pca_model import PCAEnsembleParam
        if isinstance(self.model.xyz, PCAEnsembleParam):
            xyz_params = [p for p in self.model.xyz.parameters() if p.requires_grad]
        else:
            xyz_params = self.model.parameters_of_types(("xyz",))
        params = xyz_params + list(self.scaler.parameters())
        # Per-member occupancy (softmax logits) + ADP (softplus raw) refine
        # alongside xyz in the same group (Adam adapts the per-parameter scale;
        # the LR schedule drives all groups uniformly). On-demand for population
        # refinement only.
        if self.refine_population:
            params = params + [self.model.occ_logits]
            if self.refine_member_b:
                params = params + [self.model.b_raw]
        if self.optimizer_name == "sgd":
            # Plain SGD (no momentum) + the post-step Langevin/floor noise =
            # proper overdamped SGLD: x ← x − lr·∇L + √(2·lr·T)·ε samples the
            # TRUE posterior p(x) ∝ exp(−L/T), uncorrupted by Adam's adaptive
            # preconditioner (which distorts both the sampled distribution and
            # — via the noise inflating v — the effective lr). Trade-off: one
            # global lr is poorly conditioned on the stiff amber landscape, so
            # it mixes slower and needs its own (smaller) lr.
            # NOTE (3GR5 experiment): plain SGD diverges here at any usable lr
            # (≥1e-5 explodes) — the unnormalized xray NLL gradient is ~1e5+
            # while amber's is force-clamped at 1e4, so a single global lr can't
            # serve both. The ill-conditioning that mandates Adam also dooms
            # clean isotropic SGLD. Kept for completeness / better-conditioned
            # problems; for entropy injection on stiff landscapes use a
            # preconditioned sampler instead.
            optimizer = torch.optim.SGD(params, lr=self.adam_lr, momentum=0.0)
        else:
            optimizer = torch.optim.Adam(
                params, lr=self.adam_lr,
                betas=(self.adam_beta1, self.adam_beta2),
            )

        total_steps = max(int(macro_cycles) * int(self.adam_steps_per_cycle), 1)
        lr_floor = self.warmup_start_factor * self.adam_lr

        # --- Two-phase constant-temperature sampler --------------------------
        # When sampling_fraction > 0 the run splits into:
        #   1. burn-in   [0, burnin_steps): warmup → peak → cosine-anneal down
        #      to the sampling LR. Gets the ensemble mean into the posterior
        #      bulk. Adaptive xray weight (if on) moves during this phase.
        #   2. sampling  [burnin_steps, total_steps): CONSTANT lr = sampling_lr,
        #      hence CONSTANT Langevin noise √(2·sampling_lr·T). With a frozen
        #      potential L (weights held at their end-of-burn-in values), the
        #      chain's stationary distribution is ≈ p(x) ∝ exp(-L/T) — the
        #      ensemble samples the posterior *width* instead of collapsing to
        #      a single MAP point. This is the sustained isotropic thermal bath
        #      that keeps members exploring after the cosine LR would otherwise
        #      have annealed the noise to zero.
        two_phase = self.sampling_fraction > 0.0
        burnin_steps = (
            int(round((1.0 - self.sampling_fraction) * total_steps))
            if two_phase
            else total_steps
        )
        burnin_steps = max(min(burnin_steps, total_steps), 1)
        sampling_lr = self.adam_lr * self.sampling_lr_factor

        def _lr_at(global_step: int) -> float:
            if two_phase:
                if global_step >= burnin_steps:
                    return sampling_lr
                # Burn-in: warmup (lr_floor → peak) then cosine anneal
                # (peak → sampling_lr) over the rest of the burn-in window.
                if self.warmup_steps > 0 and global_step < self.warmup_steps:
                    frac = global_step / float(self.warmup_steps)
                    return lr_floor + (self.adam_lr - lr_floor) * frac
                t = (global_step - self.warmup_steps) / float(
                    max(burnin_steps - self.warmup_steps, 1)
                )
                cos = (1.0 + math.cos(math.pi * min(t, 1.0))) / 2.0  # 1 → 0
                return sampling_lr + (self.adam_lr - sampling_lr) * cos
            if self.lr_schedule == "cosine":
                # (1 - cos) bump over each of lr_cycles periods. frac is 0
                # at the start/end of every period and 1 at its midpoint.
                phase = 2.0 * math.pi * self.lr_cycles * global_step / total_steps
                frac = (1.0 - math.cos(phase)) / 2.0
                return lr_floor + (self.adam_lr - lr_floor) * frac
            if self.lr_schedule == "sawtooth":
                # One smooth opening period (floor->max->floor) conditions the
                # fresh start and fills Adam's buffers, then lr_cycles SGDR
                # sawtooth kicks (abrupt jump to max, cosine-decay to floor,
                # repeat) for un-blunted trap escape. Adam state persists across
                # kicks (optimizer is created once), so the jumps stay
                # well-conditioned and no per-kick warmup is needed.
                n_seg = self.lr_cycles + 1
                seg = total_steps / float(n_seg)
                if global_step < seg:                      # smooth opening bump
                    frac = (1.0 - math.cos(2.0 * math.pi * global_step / seg)) / 2.0
                    return lr_floor + (self.adam_lr - lr_floor) * frac
                t_in = ((global_step - seg) % seg) / seg    # 0->1 within a sawtooth
                decay = (1.0 + math.cos(math.pi * t_in)) / 2.0  # 1->0: max at jump, floor at end
                return lr_floor + (self.adam_lr - lr_floor) * decay
            # legacy linear warmup
            if self.warmup_steps <= 0 or global_step >= self.warmup_steps:
                return self.adam_lr
            frac = global_step / float(self.warmup_steps)
            scale = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * frac
            return self.adam_lr * scale

        def _wilson_at(global_step: int) -> float:
            if self.wilson_weight_start is None:
                return self.wilson_weight
            frac = global_step / float(max(total_steps - 1, 1))
            return (
                self.wilson_weight_start
                + (self.wilson_weight - self.wilson_weight_start) * frac
            )

        def _rank_at(global_step: int) -> float:
            # Soft rank-penalty weight. With rank_weight_start set, ramp
            # linearly from start -> rank_weight over the run ("refine full,
            # then purify"); otherwise constant rank_weight.
            if self.rank_penalty_mode in ("maxent", "diverse"):
                return 1.0  # coefficients are internal to the target
            if self.rank_weight_start is None:
                return self.rank_weight
            frac = global_step / float(max(total_steps - 1, 1))
            return (
                self.rank_weight_start
                + (self.rank_weight - self.rank_weight_start) * frac
            )

        rwork_history: list = []
        rfree_history: list = []
        rval_history: list = []
        loss_history: list = []
        lr_history: list = []
        wilson_history: list = []
        rank_history: list = []
        rank_spectrum_history: list = []
        component_history: list = []
        nll_work_history: list = []
        nll_free_history: list = []
        nll_gap_history: list = []
        adam_precond_history: list = []

        def _adam_precond_stats(lr: float, sigma: float):
            """Read-only Adam diagnostic for the xyz parameter: how non-uniform
            is the per-coordinate effective step (and hence the per-atom
            noise/restore "temperature") under Adam's preconditioning?

            Adam's per-coord step scale is ``eff_lr_i = lr / (sqrt(vhat_i)+eps)``;
            the injected noise is a *uniform* ``sigma`` Å/step, so the per-coord
            noise/restore ratio ``sigma / eff_lr_i`` is the effective temperature.
            Wide spread (p95/p5) means uniform noise heats atoms unevenly — the
            thing to watch before deciding on preconditioned noise. Purely
            observational; nothing here changes the injected noise.
            """
            # Coordinate-space diagnostic; skip under the PCA reparameterization
            # (the leaves are amplitudes/basis, not Å coords).
            from .pca_model import PCAEnsembleParam
            if isinstance(self.model.xyz, PCAEnsembleParam):
                return None
            p = self.model.xyz.refinable_params
            st = optimizer.state.get(p)
            if not st or "exp_avg_sq" not in st:
                return None
            v = st["exp_avg_sq"]
            t_raw = st.get("step", 0)
            t = float(t_raw.item() if hasattr(t_raw, "item") else t_raw)
            if t < 1:
                return None
            with torch.no_grad():
                bias2 = 1.0 - self.adam_beta2 ** t
                vhat = v / max(bias2, 1e-12)
                eff_lr = lr / (vhat.sqrt() + 1e-8)         # per-coord, Å
                ratio = sigma / eff_lr.clamp_min(1e-12)    # noise / restore
                qs = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=eff_lr.device)

                def pct(x):
                    fx = x.flatten().float()
                    qv = torch.quantile(fx, qs).tolist()
                    return {
                        "min": qv[0], "p5": qv[1], "p50": qv[2],
                        "p95": qv[3], "max": qv[4],
                    }

                return {
                    "step": t,
                    "lr": lr,
                    "sigma": sigma,
                    "eff_lr": pct(eff_lr),
                    "ratio": pct(ratio),
                }

        def _histories(burnin=None, sampling=None) -> dict:
            """Assemble the history dict (final return shape) from the lists as
            they stand now. Used for both mid-run snapshots and the final
            return, so the two never drift apart."""
            return {
                "rwork": rwork_history,
                "rfree": rfree_history,
                "rval": rval_history,
                "loss": loss_history,
                "lr": lr_history,
                "wilson_weight": wilson_history,
                "rank_weight": rank_history,
                "rank_spectrum": rank_spectrum_history,
                "component_breakdown": component_history,
                "nll_work": nll_work_history,
                "nll_free": nll_free_history,
                "nll_gap": nll_gap_history,
                "adam_precond": adam_precond_history,
                "burnin_cycles": burnin
                if burnin is not None
                else len(rwork_history),
                "sampling_summary": sampling,
            }

        # In-loss counts (work/free flag AND valid), matching the X-ray mask.
        # Used only for per-reflection NLL reporting in the gap meter below;
        # the loss-state weights themselves are now per-ASU (no /n_work).
        n_work = max(self._inloss_count(self.reflection_data.work.indices), 1)
        n_free = self._inloss_count(self.reflection_data.free.indices)
        # Arm/disarm ensemble dropout on the model for this refinement.
        self.model.configure_dropout(
            self.use_dropout, self.dropout_min, self.dropout_max
        )
        if self.use_dropout and self.verbose > 0:
            print(
                f"[ensemble] dropout ON: each step uses a random "
                f"{self.dropout_min}-{self.dropout_max} of {self.n_members} "
                f"members for the structure factor",
                flush=True,
            )
        do_langevin = self.langevin_T > 0.0
        # Decoupled (lr-independent) noise floor with a DC-offset cosine:
        #   σ_floor(step) = noise_floor_sigma
        #                 + noise_floor_amp · (1 - cos(2π·cycles·step/T))/2
        # so the injected noise never drops below noise_floor_sigma Å/step and
        # pulses up by noise_floor_amp. Added in quadrature with the lr-coupled
        # Langevin term. Unlike √(2·lr·T) — which dies as lr anneals — this
        # sustains a thermal bath throughout, including the constant-lr
        # sampling phase, to keep the ensemble exploring and push the chain
        # off sharp work-overfit minima (drives the work/free gap closed).
        do_noise_floor = self.noise_floor_sigma > 0.0 or self.noise_floor_amp > 0.0
        do_noise = do_langevin or do_noise_floor
        if do_langevin and self.verbose > 0:
            print(
                f"[ensemble] Langevin ON: T={self.langevin_T} — "
                f"injecting N(0, 2·lr·T·I) noise on xyz after each Adam step "
                f"(Adam-preconditioned SGLD; samples p(x) ∝ exp(-L/T) at the "
                f"global LR scale).",
                flush=True,
            )
        if do_noise_floor and self.verbose > 0:
            print(
                f"[ensemble] noise floor ON (lr-independent): "
                f"σ_floor = {self.noise_floor_sigma:.2e} + "
                f"{self.noise_floor_amp:.2e}·(1-cos)/2 Å/step over "
                f"{self.noise_floor_cycles:g} cosine cycle(s); added in "
                f"quadrature with the Langevin term. Sustains exploration "
                f"after lr anneals.",
                flush=True,
            )
        # --- Adaptive xray weight schedule ---------------------------------
        # w_xray(step) = self.xray_weight · max(floor, EMA(nll_gap)^(-log2 F))
        # where nll_gap = (nll_free_per_refl / nll_work_per_refl), updated
        # every Adam step from a fresh no-grad forward of the free target.
        # The slope log2(F) means: each doubling of gap cuts xray weight by F×
        # (with F=5 (default): 2× → 0.2, 4× → 0.04, 8× → 0.008). F=10 was
        # the original try and turned out to be too aggressive — the natural
        # sampling gap of ~2-4 already drove weight to 0.01-0.1, starving
        # the work fit. EMA half-life in Adam steps smooths per-step noise.
        adaptive_slope = math.log2(max(self.xray_adaptive_doubling_factor, 1.0001))
        # Free-set-aware PENALTY: w_rank(step) = rank_adaptive_base · EMA(gap)^(+log2 F_rank)
        # — mirror of adaptive-F but on the regularizer: each doubling of the
        # work/free NLL ratio multiplies the penalty by F_rank (push regularization
        # UP as the ensemble overfits). Uses the SAME ema_gap signal.
        rank_adaptive_slope = math.log2(max(self.rank_adaptive_doubling_factor, 1.0001))
        _use_ema = self.xray_adaptive or self.rank_adaptive
        ema_alpha = (
            1.0 - 0.5 ** (1.0 / max(int(self.xray_adaptive_ema_halflife_steps), 1))
            if _use_ema
            else 0.0
        )
        ema_gap: Optional[float] = None
        # In two-phase mode the adaptive xray multiplier is frozen at its
        # end-of-burn-in value so the sampling phase has a fixed potential L.
        frozen_xray_mul: Optional[float] = None
        frozen_rank_w: Optional[float] = None
        last_rank_w: float = self.rank_adaptive_base
        if two_phase and self.verbose > 0:
            print(
                f"[ensemble] two-phase sampler: burn-in for "
                f"{burnin_steps}/{total_steps} steps "
                f"({(1.0 - self.sampling_fraction) * 100:.0f}% of run), then "
                f"CONSTANT lr={sampling_lr:.2e} (= {self.sampling_lr_factor:g}× "
                f"adam_lr) sampling phase with constant noise "
                f"√(2·lr·T)={math.sqrt(2.0 * sampling_lr * self.langevin_T):.2e} "
                f"Å/step. Weights freeze at the burn-in boundary so the chain "
                f"samples a fixed p(x) ∝ exp(-L/T).",
                flush=True,
            )
        if self.xray_adaptive and self.verbose > 0:
            F = self.xray_adaptive_doubling_factor
            print(
                f"[ensemble] xray_adaptive ON: w_xray_step = "
                f"{self.xray_weight:.4g} · max({self.xray_adaptive_floor:.1e}, "
                f"EMA_{self.xray_adaptive_ema_halflife_steps}(NLL_free/NLL_work) "
                f"^ -log2({F:g})). Each doubling of gap cuts the xray pull by "
                f"{F:g}×; floor at {self.xray_adaptive_floor:.1e} of base.",
                flush=True,
            )
        if self.amber_every > 1 and self.verbose > 0:
            print(
                f"[ensemble] Amber every {self.amber_every} steps "
                f"(skipped on the other steps; aggregate() drops zero-weight "
                f"targets). Amber backward also skipped on those steps — "
                f"this is the dominant per-step cost.",
                flush=True,
            )
        global_step = 0
        start_cycle = 0
        if resume_state is not None:
            # Validate the schedule matches: the lr / noise / Wilson curves are
            # functions of global_step over a FIXED total_steps, so resuming
            # under a different (macro_cycles × adam_steps_per_cycle) would warp
            # them. Refuse rather than silently mis-schedule.
            ck_mc = resume_state.get("macro_cycles")
            ck_spc = resume_state.get("adam_steps_per_cycle")
            if ck_mc != macro_cycles or ck_spc != self.adam_steps_per_cycle:
                raise ValueError(
                    f"[ensemble] resume schedule mismatch: checkpoint had "
                    f"macro_cycles={ck_mc}, adam_steps_per_cycle={ck_spc}; "
                    f"this run has {macro_cycles}, {self.adam_steps_per_cycle}. "
                    f"Resume requires identical schedule."
                )
            # strict=False: the model/scaler were already rebuilt by setup from
            # the same PDB + config, so their custom state_dict metadata (the
            # 'pdb' DataFrame, cell, spacegroup, config scalars) is already
            # correct and shows up as "unexpected" here. We only need to overlay
            # the refinable tensors + buffers. Log the key diff so a genuinely
            # missing parameter can't slip through silently.
            mk_m = self.model.load_state_dict(resume_state["model"], strict=False)
            mk_s = self.scaler.load_state_dict(
                resume_state["scaler"], strict=False
            )
            optimizer.load_state_dict(resume_state["optimizer"])
            self.model.reset_cache()
            if self.verbose > 0:
                print(
                    f"[ensemble] resume load: "
                    f"model(missing={len(mk_m.missing_keys)}, "
                    f"unexpected={len(mk_m.unexpected_keys)}) "
                    f"scaler(missing={len(mk_s.missing_keys)}, "
                    f"unexpected={len(mk_s.unexpected_keys)})",
                    flush=True,
                )
            global_step = int(resume_state["global_step"])
            start_cycle = int(resume_state["next_cycle"])
            ema_gap = resume_state.get("ema_gap")
            frozen_xray_mul = resume_state.get("frozen_xray_mul")
            # Restore RNG so the post-resume noise/dropout stream continues as if
            # uninterrupted (set AFTER setup, which itself consumed RNG).
            rng = resume_state.get("torch_rng")
            if rng is not None:
                torch.set_rng_state(rng.cpu() if hasattr(rng, "cpu") else rng)
            crng = resume_state.get("cuda_rng")
            if crng is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(crng)
                except Exception:
                    pass
            # Seed the history lists so the final result spans the whole run.
            ckh = resume_state.get("histories", {})
            for lst, key in (
                (rwork_history, "rwork"),
                (rfree_history, "rfree"),
                (rval_history, "rval"),
                (loss_history, "loss"),
                (lr_history, "lr"),
                (wilson_history, "wilson_weight"),
                (component_history, "component_breakdown"),
                (nll_work_history, "nll_work"),
                (nll_free_history, "nll_free"),
                (nll_gap_history, "nll_gap"),
            ):
                lst.extend(ckh.get(key) or [])
            if self.verbose > 0:
                print(
                    f"[ensemble] RESUMED from checkpoint at cycle {start_cycle}"
                    f"/{macro_cycles} (global_step={global_step}); "
                    f"{len(rfree_history)} cycles of history restored.",
                    flush=True,
                )

        def _save_checkpoint(next_cycle: int) -> None:
            """Atomically write the full resume state to ``checkpoint_path``."""
            cuda_rng = None
            if torch.cuda.is_available():
                try:
                    cuda_rng = torch.cuda.get_rng_state_all()
                except Exception:
                    cuda_rng = None
            ckpt = {
                "format": "ensemble_resume_v1",
                "model": self.model.state_dict(),
                "scaler": self.scaler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "next_cycle": next_cycle,
                "macro_cycles": macro_cycles,
                "adam_steps_per_cycle": self.adam_steps_per_cycle,
                "ema_gap": ema_gap,
                "frozen_xray_mul": frozen_xray_mul,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": cuda_rng,
                "histories": _histories(),
            }
            tmp = checkpoint_path + ".tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, checkpoint_path)

        for cycle in range(start_cycle, macro_cycles):
            # H positions are derived from heavy atoms via local-frame
            # placement inside ``AmberTarget._place_hydrogens`` on every
            # forward — no per-cycle refresh needed.

            state = self.complete_loss_state()
            last_loss = None
            # Step one at a time so the LR + Wilson schedules can be
            # applied per Adam step (LossState.run hides its inner loop,
            # so we drive nsteps=1 and set the schedules ourselves).
            for _ in range(self.adam_steps_per_cycle):
                lr = _lr_at(global_step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                state.set_weight(
                    "regularization/wilson",
                    _wilson_at(global_step),
                )
                if getattr(self, "rank_target", None) is not None:
                    state.set_weight(
                        "regularization/rank",
                        _rank_at(global_step),
                    )
                # Amber dominates per-step cost (100 sequential OpenMM
                # round-trips). Skip it K-1 of every K steps by setting its
                # weight to 0 — aggregate() drops zero-weight targets and
                # never invokes the forward (and so backward never runs).
                # Geometry drift between kicks is negligible at K<=5 vs bond
                # length, while step time drops ~K-fold.
                if self.amber_target is not None:
                    on_amber = (global_step % self.amber_every) == 0
                    state.set_weight(
                        "restraints/amber_kl",
                        self.amber_weight if on_amber else 0.0,
                    )
                # Adaptive xray weight (uses EMA from the previous step; for
                # the very first step there's no history → full base weight).
                # In two-phase mode the multiplier freezes once we cross into
                # the sampling phase so the sampled potential L is stationary.
                in_sampling = two_phase and (global_step >= burnin_steps)
                if self.xray_adaptive:
                    if in_sampling:
                        if frozen_xray_mul is None:
                            # First sampling step: lock in the burn-in value.
                            if ema_gap is not None and ema_gap > 0.0:
                                frozen_xray_mul = max(
                                    self.xray_adaptive_floor,
                                    ema_gap ** (-adaptive_slope),
                                )
                            else:
                                frozen_xray_mul = 1.0
                            if self.verbose > 0:
                                print(
                                    f"[ensemble] entering sampling phase at "
                                    f"step {global_step}: freezing xray "
                                    f"multiplier at {frozen_xray_mul:.4g} "
                                    f"(w_xray={self.xray_weight * frozen_xray_mul:.4g})",
                                    flush=True,
                                )
                        mul = frozen_xray_mul
                    elif ema_gap is not None and ema_gap > 0.0:
                        mul = max(
                            self.xray_adaptive_floor,
                            ema_gap ** (-adaptive_slope),
                        )
                    else:
                        mul = 1.0
                    state.set_weight("xray/work", self.xray_weight * mul)
                # Free-set-aware penalty: ramp the rank/maxent weight UP with the
                # work-free gap (the regularizer mirror of adaptive-F). Frozen in
                # the sampling phase like the xray multiplier.
                if self.rank_adaptive and getattr(self, "rank_target", None) is not None:
                    if in_sampling and frozen_rank_w is not None:
                        rw = frozen_rank_w
                    elif ema_gap is not None and ema_gap > 0.0:
                        rw = self.rank_adaptive_base * (ema_gap ** rank_adaptive_slope)
                        rw = float(min(max(rw, 0.0), 1e7))  # cap to avoid runaway
                        if in_sampling and frozen_rank_w is None:
                            frozen_rank_w = rw
                    else:
                        rw = self.rank_adaptive_base
                    state.set_weight("regularization/rank", rw)
                    last_rank_w = rw
                # Fresh member subset per step. state.run() resets the SF
                # cache before its forward, so all targets in this step see
                # one consistent subsetted F_calc; the next step redraws.
                if self.use_dropout:
                    self.model.resample_dropout()
                step_loss = state.run(
                    optimizer,
                    nsteps=1,
                    context=f"ensemble_refine.cycle_{cycle}",
                )
                # Hold noise off until warmup completes: during the lr ramp the
                # optimizer can't supply a full restoring step, so injecting the
                # (lr-independent) constant floor there is an unrepresentative
                # shock. After warmup the structure has settled under the
                # restraints at full constant lr, and the floor then sustains a
                # well-posed thermal bath ("settle, then perturb").
                if do_noise and global_step >= self.warmup_steps:
                    # Noise on xyz: x ← x + √(2·lr·T + σ_floor²)·ε.
                    #  - Langevin term 2·lr·T: anneals with lr (→0 in sampling).
                    #  - Floor term σ_floor²: lr-independent DC-offset cosine,
                    #    sustains a thermal bath throughout.
                    # Coords just moved, so invalidate the SF cache.
                    with torch.no_grad():
                        var = 2.0 * lr * self.langevin_T
                        if do_noise_floor:
                            phase = (
                                2.0 * math.pi * self.noise_floor_cycles
                                * global_step / total_steps
                            )
                            sig = self.noise_floor_sigma + self.noise_floor_amp * (
                                (1.0 - math.cos(phase)) / 2.0
                            )
                            var += sig * sig
                        scale = math.sqrt(var)
                        p = self.model.xyz.refinable_params
                        p.add_(torch.randn_like(p) * scale)
                    self.model.reset_cache()
                # Recompute the per-refl NLL ratio on the just-updated params
                # and update the EMA, ready for the next step's weight pick.
                # Cost: one no-grad forward of xray/free (work is in the
                # closure's already-computed loss). SF cache is auto-invalid
                # because Adam (and Langevin) mutated the params in-place.
                # Skipped during the sampling phase: the multiplier is frozen,
                # so the EMA forward would be wasted compute.
                if (self.xray_adaptive or self.rank_adaptive) and not in_sampling:
                    with torch.no_grad():
                        nll_work_sum = float(self.xray_target_work.forward())
                        nll_free_sum = float(self.xray_target_test.forward())
                    nll_w_per = nll_work_sum / max(n_work, 1)
                    nll_f_per = nll_free_sum / max(n_free, 1)
                    if nll_w_per > 1e-9:
                        gap_now = nll_f_per / nll_w_per
                        if ema_gap is None:
                            ema_gap = gap_now
                        else:
                            ema_gap = (
                                ema_alpha * gap_now + (1.0 - ema_alpha) * ema_gap
                            )
                if step_loss is not None:
                    last_loss = step_loss
                global_step += 1
            # Evaluate R-factors / NLLs on the FULL ensemble (dropout is a
            # train-time-only perturbation, like NN dropout at eval).
            if self.use_dropout:
                self.model.set_dropout_full()
                self.model.reset_cache()
            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                rval = self._compute_rval()
            rwork_history.append(rwork)
            rfree_history.append(rfree)
            rval_history.append(rval)
            loss_history.append(
                float(last_loss.item()) if last_loss is not None else float("nan")
            )
            lr_history.append(_lr_at(global_step - 1))
            wilson_history.append(_wilson_at(global_step - 1))
            # Soft rank-penalty weight + ensemble mode-spectrum diagnostics
            # (effective rank, etc.) — tracks the "purification" as λ ramps.
            if getattr(self, "rank_target", None) is not None:
                rank_history.append(
                    last_rank_w if self.rank_adaptive else _rank_at(global_step - 1)
                )
                rank_spectrum_history.append(self.rank_target.spectrum_diagnostics())
            # Snapshot the per-target loss / weight / weighted breakdown from
            # the final in-cycle step (``state._losses`` still holds the last
            # closure's forward) so the relative magnitudes of the weighted
            # terms are visible for tuning.
            component_history.append(self._component_breakdown(state))
            # Per-reflection work/free ML NLL and their gap. The gap is, at
            # the per-set variance MLE, ≈ log(σ²_free/σ²_work) — the
            # overfitting optimism and the calibration target for the prior.
            nll_work = self._xray_nll_per_refl(self.xray_target_work, n_work)
            nll_free = self._xray_nll_per_refl(self.xray_target_test, n_free)
            nll_gap = nll_free - nll_work
            nll_work_history.append(nll_work)
            nll_free_history.append(nll_free)
            nll_gap_history.append(nll_gap)
            # Adam preconditioning diagnostic at this cycle's end (read-only).
            # Use the last in-cycle step's lr and the noise floor σ at that step.
            last_step = max(global_step - 1, 0)
            sigma_now = self.noise_floor_sigma
            if do_noise_floor:
                _ph = (
                    2.0 * math.pi * self.noise_floor_cycles
                    * last_step / total_steps
                )
                sigma_now = self.noise_floor_sigma + self.noise_floor_amp * (
                    (1.0 - math.cos(_ph)) / 2.0
                )
            adam_precond_history.append(
                _adam_precond_stats(_lr_at(last_step), sigma_now)
            )
            # Birth–death population dynamics: evolve the member set AFTER
            # evaluating this cycle's R/NLL on the current population. Gated to
            # start once warmup completes ("settle, then evolve").
            pop_msg = ""
            if (
                self.refine_population
                and (self.death_rate > 0.0 or self.birth_rate > 0.0)
                and (cycle + 1) % self.birth_death_every == 0
                and global_step >= self.warmup_steps
            ):
                pop = self._birth_death_step(optimizer)
                with torch.no_grad():
                    w_now = self.model.member_weights()
                    w_al = w_now[self.model._alive]
                    b_al = self.model.member_b()[self.model._alive]
                pop_msg = (
                    f"  pop: n_alive={pop['n_alive']} (-{pop['died']}/+{pop['born']}) "
                    f"w[min/med/max]={float(w_al.min()):.4f}/"
                    f"{float(w_al.median()):.4f}/{float(w_al.max()):.4f} "
                    f"B[med/max]={float(b_al.median()):.1f}/{float(b_al.max()):.1f}"
                )
            if self.verbose > 0:
                rank_msg = ""
                if rank_spectrum_history:
                    sp = rank_spectrum_history[-1]
                    rank_msg = (
                        f"  w_rank={rank_history[-1]:.3g} "
                        f"eff_rank={sp['participation_ratio']:.1f} "
                        f"H={sp.get('spectral_entropy', float('nan')):.2f} "
                        f"var={sp.get('total_variance', float('nan')):.1f} "
                        f"S={sp.get('conf_entropy_nats', float('nan')):.1f}nats"
                    )
                print(
                    f"[ensemble] cycle {cycle + 1}/{macro_cycles}: "
                    f"R_work={rwork:.4f}  R_free={rfree:.4f}  R_val={rval:.4f}  "
                    f"loss={loss_history[-1]:.4e}  lr={lr_history[-1]:.2e}  "
                    f"w_wilson={wilson_history[-1]:.2f}{rank_msg}{pop_msg}",
                    flush=True,
                )
                print(
                    f"           NLL/refl  work={nll_work:.3f}  "
                    f"free={nll_free:.3f}  gap(free-work)={nll_gap:.3f} nats",
                    flush=True,
                )
                print(state.format_breakdown(), flush=True)
            # Persist intermediate state on long runs: the human snapshot
            # (caller-owned PDB/JSON) and the resume checkpoint (full state).
            # Both are best-effort — a failed write must never kill the run.
            if snapshot_every > 0 and (cycle + 1) % snapshot_every == 0:
                if on_snapshot is not None:
                    try:
                        on_snapshot(cycle + 1, _histories())
                    except Exception as exc:
                        if self.verbose > 0:
                            print(
                                f"[ensemble] snapshot at cycle {cycle + 1} "
                                f"failed: {exc!r} (continuing)",
                                flush=True,
                            )
                if checkpoint_path is not None:
                    try:
                        _save_checkpoint(next_cycle=cycle + 1)
                    except Exception as exc:
                        if self.verbose > 0:
                            print(
                                f"[ensemble] checkpoint at cycle {cycle + 1} "
                                f"failed: {exc!r} (continuing)",
                                flush=True,
                            )
        # Sampling-phase summary: in two-phase mode, the meaningful result is
        # the statistics over the constant-T sampling cycles, not the last
        # snapshot. burnin_cycles = first cycle index that is in the sampling
        # phase (0-based count of burn-in cycles).
        burnin_cycles = (
            int(math.ceil(burnin_steps / float(self.adam_steps_per_cycle)))
            if two_phase
            else macro_cycles
        )
        sampling_summary = None
        if two_phase and burnin_cycles < len(rfree_history):
            samp_rf = rfree_history[burnin_cycles:]
            samp_rw = rwork_history[burnin_cycles:]
            n = len(samp_rf)
            mean_rf = sum(samp_rf) / n
            var_rf = sum((x - mean_rf) ** 2 for x in samp_rf) / max(n - 1, 1)
            sampling_summary = {
                "burnin_cycles": burnin_cycles,
                "n_sampling_cycles": n,
                "rfree_sampling_mean": mean_rf,
                "rfree_sampling_std": var_rf ** 0.5,
                "rfree_sampling_min": min(samp_rf),
                "rwork_sampling_mean": sum(samp_rw) / n,
            }
            if self.verbose > 0:
                print(
                    f"[ensemble] sampling phase ({n} cycles): "
                    f"R_free mean={mean_rf:.4f} ± {var_rf ** 0.5:.4f}  "
                    f"min={min(samp_rf):.4f}  "
                    f"R_work mean={sum(samp_rw) / n:.4f}",
                    flush=True,
                )
        return _histories(burnin=burnin_cycles, sampling=sampling_summary)

    def _inloss_count(self, idx) -> int:
        """Count reflections in ``idx`` that are also valid (in the loss).

        The work/free subset indices already apply the validity masks, so
        intersecting with ``ReflectionData.masks()`` here is idempotent; it is
        retained to make the in-loss count explicit. Intersecting with
        ``ReflectionData.masks()`` (the validity/resolution mask the target
        applies) gives the count actually contributing to the loss.
        """
        if idx is None or idx.numel() == 0:
            return 0
        valid = self.reflection_data.masks().bool().to(idx.device)
        return int(valid.index_select(0, idx).sum())

    def _xray_nll_per_refl(self, target, n: int) -> float:
        """Mean per-reflection ML NLL (nats) for an X-ray target's set.

        The ML targets return the *sum* of per-reflection NLL over their
        masked set; dividing by the set size gives the per-reflection mean.
        The work/free gap of this quantity is, at the per-set variance MLE,
        ``≈ log(σ²_free / σ²_work)`` — the optimism of the training
        likelihood and the natural calibration target for the prior strength
        (drive the gap to ~0). Monitoring only, so evaluated under no_grad.
        """
        if n <= 0:
            return float("nan")
        with torch.no_grad():
            return float(target().item()) / n

    def _compute_rval(self) -> float:
        """R-factor on the validation set (monitoring only)."""
        with torch.no_grad():
            # get_data() returns compact arrays already restricted to the set
            # (the trailing element is the _ReflectionSubset view, not a mask).
            F_obs, F_calc, _sigma, _centric, _sub = (
                self.xray_target_validation.get_data()
            )
            num = (F_obs - F_calc.abs()).abs().sum()
            den = F_obs.abs().sum().clamp(min=1e-8)
            return float((num / den).item())

    def _component_breakdown(self, state: LossState) -> Dict[str, Dict[str, float]]:
        """Per-target ``{loss, weight, weighted}`` from the most recent step.

        Reads the losses cached by the last closure forward inside
        :meth:`LossState.run` (``state.get_loss``), pairs each with its
        effective weight, and returns a flat dict. Zero-weight monitoring
        targets (``xray/free``, ``xray/validation``) are skipped by
        ``aggregate`` and so are absent here — only the terms that actually
        drive the optimizer appear. ``weighted = weight * loss`` is the term's
        true contribution to the total loss, which is what matters for tuning
        the relative weights.
        """
        out: Dict[str, Dict[str, float]] = {}
        for name in state.targets:
            loss = state.get_loss(name)
            if loss is None:
                continue
            w = float(state.get_effective_weight(name))
            lval = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            out[name] = {"loss": lval, "weight": w, "weighted": w * lval}
        return out
