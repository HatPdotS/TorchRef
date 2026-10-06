import math

import torch
from torch.optim import Optimizer


class LangevinSA(Optimizer):
    """BAOAB Langevin dynamics integrator with simulated annealing.

    Implements the BAOAB splitting scheme (Leimkuhler & Matthews, 2013)
    for gradient-guided exploration with thermodynamically correct noise.
    One gradient evaluation per step via staggered B steps.

    Adaptive masses from EMA of squared gradients provide automatic scale
    invariance across all parameter types (xyz, B-factors, occupancies,
    torsions, etc.).

    Args:
        params: Iterable of parameters or param groups.
        dt: Integration timestep.
        friction: Friction coefficient gamma. Controls thermalization speed.
        T_initial: Starting temperature.
        T_final: Final temperature.
        total_steps: Total number of annealing steps.
        cooling_schedule: 'exponential' or 'linear'.
        adaptive_masses: Use EMA of grad² as per-element masses.
        mass_beta: EMA decay for adaptive masses.
        mass_eps: Floor for adaptive masses (numerical stability).
        gradient_clip: Optional max gradient norm (per-parameter).
        max_step_size: Maximum displacement per element per full step.
            Velocities are clamped so ``|v * dt| <= max_step_size``.
    """

    def __init__(
        self,
        params,
        dt=0.01,
        friction=10.0,
        T_initial=2500.0,
        T_final=0.01,
        total_steps=1000,
        cooling_schedule="exponential",
        adaptive_masses=True,
        mass_beta=0.999,
        mass_eps=1e-8,
        gradient_clip=None,
        max_step_size=0.1,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if friction <= 0:
            raise ValueError(f"friction must be positive, got {friction}")
        if T_initial <= 0 or T_final <= 0:
            raise ValueError("Temperatures must be positive")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if cooling_schedule not in ("exponential", "linear"):
            raise ValueError(
                f"cooling_schedule must be 'exponential' or 'linear', "
                f"got '{cooling_schedule}'"
            )

        defaults = dict(
            dt=dt,
            friction=friction,
            T_initial=T_initial,
            T_final=T_final,
            total_steps=total_steps,
            cooling_schedule=cooling_schedule,
            adaptive_masses=adaptive_masses,
            mass_beta=mass_beta,
            mass_eps=mass_eps,
            gradient_clip=gradient_clip,
            max_step_size=max_step_size,
        )
        super().__init__(params, defaults)

        self._current_step = 0

    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------

    def _get_temperature(self):
        """Compute temperature at current step from the cooling schedule."""
        group = self.param_groups[0]
        T_i = group["T_initial"]
        T_f = group["T_final"]
        N = group["total_steps"]
        t = min(self._current_step, N - 1) / max(N - 1, 1)

        if group["cooling_schedule"] == "exponential":
            log_ratio = math.log(T_f / T_i)
            return T_i * math.exp(log_ratio * t)
        else:  # linear
            return T_i + (T_f - T_i) * t

    @property
    def temperature(self):
        """Current temperature from the annealing schedule."""
        return self._get_temperature()

    @property
    def current_step(self):
        """Index of the current annealing step."""
        return self._current_step

    @property
    def total_steps(self):
        """Total number of steps in the annealing schedule."""
        return self.param_groups[0]["total_steps"]

    @property
    def kinetic_energy(self):
        """Sum of 0.5 * m * v^2 over all parameters (diagnostic)."""
        ke = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "velocity" not in state:
                    continue
                v = state["velocity"]
                m = state.get("mass")
                if m is not None:
                    ke += 0.5 * (m * v * v).sum().item()
                else:
                    ke += 0.5 * (v * v).sum().item()
        return ke

    # ------------------------------------------------------------------
    # Physical-mass seeding (genuine thermostatted MD)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_physical_masses(self, mass_by_param, T=None):
        """Seed externally supplied (e.g. physical atomic) masses and
        initialise Maxwell-Boltzmann velocities.

        Use this for a genuine thermostatted MD where the masses are physical
        atomic masses rather than the adaptive grad²-derived preconditioner.
        Set ``adaptive_masses=False`` on the affected param group(s) so the
        BAOAB final B-step does not overwrite ``state["mass"]`` with grad²
        values on the next step.

        Parameters
        ----------
        mass_by_param : dict
            Maps ``id(param) -> mass tensor`` that is ``expand_as``-compatible
            with the parameter (e.g. shape ``(members, atoms, 1)`` for an
            ``(members, atoms, 3)`` coordinate parameter). Parameters absent
            from the dict keep unit mass (``state["mass"] = None``).
        T : float, optional
            Temperature for the Maxwell-Boltzmann velocity init. Defaults to
            the current scheduled temperature.
        """
        if T is None:
            T = self._get_temperature()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["prev_grad"] = None
                m = mass_by_param.get(id(p))
                if m is not None:
                    m = (
                        m.to(device=p.device, dtype=p.dtype)
                        .expand_as(p.data)
                        .clone()
                    )
                    state["mass"] = m
                    state["velocity"] = torch.randn_like(p.data) * (T / m).sqrt()
                else:
                    state["mass"] = None
                    state["velocity"] = torch.randn_like(p.data) * math.sqrt(T)

    # ------------------------------------------------------------------
    # BAOAB step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure):
        """Perform one BAOAB Langevin dynamics step.

        Tracks the best-loss configuration and rolls back to it, zeroing the
        velocities, when the loss becomes non-finite or rises above the best loss
        seen so far by more than twice that loss's magnitude
        (``loss - best > 2 * |best|``). This prevents the dynamics from permanently
        damaging the structure while still allowing uphill exploration.

        Args:
            closure: A callable that re-evaluates the model and returns the
                loss. The closure must call ``loss.backward()`` before
                returning.

        Returns:
            The loss value from the closure evaluation.
        """
        if closure is None:
            raise RuntimeError("LangevinSA requires a closure")

        T = self._get_temperature()
        first_step = self._current_step == 0

        # ---- Snapshot positions before B-A-O-A (for rollback on NaN) ----
        snapshots = {}
        for group in self.param_groups:
            for p in group["params"]:
                snapshots[id(p)] = p.data.clone()

        # ---- B-A-O-A using stored prev_grad (skip B on first step) ----
        for group in self.param_groups:
            dt = group["dt"]
            gamma = group["friction"]
            adaptive = group["adaptive_masses"]
            eps = group["mass_eps"]
            half_dt = 0.5 * dt
            alpha = math.exp(-gamma * dt)
            max_v = group["max_step_size"] / dt

            for p in group["params"]:
                state = self.state[p]

                # --- Initialise state on very first call ---
                if "velocity" not in state:
                    state["prev_grad"] = None
                    if adaptive:
                        state["grad_sq_avg"] = torch.ones_like(p.data)
                    state["mass"] = None
                    state["velocity"] = torch.randn_like(p.data) * math.sqrt(T)

                v = state["velocity"]
                prev_grad = state["prev_grad"]
                m = state["mass"]

                # B: half-kick from stored gradient (skip on first step)
                if not first_step and prev_grad is not None:
                    if m is not None:
                        v.add_(prev_grad / m, alpha=-half_dt)
                    else:
                        v.add_(prev_grad, alpha=-half_dt)

                # A: half-drift (use p.add_ to increment _version
                #    so CachedForwardMixin sees the change)
                p.add_(v, alpha=half_dt)

                # O: Ornstein-Uhlenbeck thermostat
                noise = torch.randn_like(v)
                if m is not None:
                    sigma = ((T / m) * (1.0 - alpha * alpha)).sqrt()
                else:
                    sigma = math.sqrt(T * (1.0 - alpha * alpha))
                v.mul_(alpha).add_(noise * sigma)

                # Velocity clamping: bound displacement per step
                v.clamp_(-max_v, max_v)

                # A: half-drift
                p.add_(v, alpha=half_dt)

        # ---- Evaluate loss + gradient at new position ----
        with torch.enable_grad():
            loss = closure()

        # ---- NaN / loss-explosion protection ----
        rollback = False
        if not torch.isfinite(loss):
            rollback = True
        elif hasattr(self, "_best_loss"):
            # A rise measured in units of the best loss's magnitude. For best > 0 this
            # is ``loss > 3 * best``; written that way it holds for every reachable loss
            # once best < 0, and every step would roll back.
            if loss.item() - self._best_loss > 2.0 * abs(self._best_loss):
                rollback = True

        if rollback:
            # Restore to best-known configuration if available,
            # otherwise to the pre-step snapshot.
            if hasattr(self, "_best_params"):
                for group in self.param_groups:
                    for p in group["params"]:
                        p.data = self._best_params[id(p)].clone()
            else:
                for group in self.param_groups:
                    for p in group["params"]:
                        p.data = snapshots[id(p)].clone()
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["velocity"].zero_()
                    state["prev_grad"] = None
            self._current_step += 1
            return loss

        # ---- Track best configuration ----
        loss_val = loss.item()
        if not hasattr(self, "_best_loss") or loss_val < self._best_loss:
            self._best_loss = loss_val
            self._best_params = {}
            for group in self.param_groups:
                for p in group["params"]:
                    self._best_params[id(p)] = p.data.clone()

        # ---- Final B step: half-kick from new gradient + store ----
        for group in self.param_groups:
            dt = group["dt"]
            adaptive = group["adaptive_masses"]
            beta = group["mass_beta"]
            eps = group["mass_eps"]
            clip = group["gradient_clip"]
            half_dt = 0.5 * dt

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self.state[p]
                v = state["velocity"]

                # Optional gradient clipping
                if clip is not None:
                    grad_norm = grad.norm()
                    if grad_norm > clip:
                        grad = grad * (clip / grad_norm)

                # Update adaptive mass
                if adaptive:
                    sq = state["grad_sq_avg"]
                    sq.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)
                    state["mass"] = sq.sqrt() + eps

                m = state["mass"]

                # B: half-kick from new gradient
                if m is not None:
                    v.add_(grad / m, alpha=-half_dt)
                else:
                    v.add_(grad, alpha=-half_dt)

                # Store gradient for next step's first B
                state["prev_grad"] = grad.clone()

        self._current_step += 1
        return loss
