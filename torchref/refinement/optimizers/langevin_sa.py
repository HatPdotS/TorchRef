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

    Call :meth:`calibrate` before the main loop to probe parameter stiffness
    and warm up the adaptive masses without moving the structure.

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
            Velocities are clamped so |v * dt| <= max_step_size.
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
        self._calibrated = False

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
    # Calibration: probe stiffness then rollback
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(self, closure, n_steps=10):
        """Probe parameter stiffness over n_steps, then rollback.

        Runs small random perturbations to collect gradient statistics,
        sets the adaptive masses from the observed grad², then restores
        all parameters to their original values and initialises velocities
        from Maxwell-Boltzmann with correctly scaled masses.

        Args:
            closure: Same closure as for ``step()`` — must zero_grad,
                compute loss, call backward, and return loss.
            n_steps: Number of probing steps.
        """
        # --- snapshot ---
        snapshots = {}
        for group in self.param_groups:
            for p in group["params"]:
                snapshots[id(p)] = p.data.clone()

        # --- accumulate grad² ---
        grad_sq_sum = {}
        n_valid = {}
        for group in self.param_groups:
            for p in group["params"]:
                grad_sq_sum[id(p)] = torch.zeros_like(p.data)
                n_valid[id(p)] = 0

        for i in range(n_steps):
            # Perturb via data assignment (not in-place add) so that
            # data_ptr changes and CachedForwardMixin sees a cache miss.
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = p.data + torch.randn_like(p.data) * 1e-3

            with torch.enable_grad():
                loss = closure()

            if torch.isfinite(loss):
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            grad_sq_sum[id(p)].add_(p.grad.detach() ** 2)
                            n_valid[id(p)] += 1

            # Restore via assignment to change data_ptr (cache invalidation).
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = snapshots[id(p)].clone()

        # --- compute masses from calibration ---
        T = self._get_temperature()

        # First pass: compute raw masses and collect all nonzero elements
        all_nonzero_masses = []
        for group in self.param_groups:
            eps = group["mass_eps"]
            for p in group["params"]:
                pid = id(p)
                state = self.state[p]
                n = max(n_valid[pid], 1)
                avg_grad_sq = grad_sq_sum[pid] / n
                state["grad_sq_avg"] = avg_grad_sq
                m = avg_grad_sq.sqrt() + eps
                state["mass"] = m
                # Collect elements that actually got gradient signal
                nonzero = m[avg_grad_sq > 0]
                if nonzero.numel() > 0:
                    all_nonzero_masses.append(nonzero)

        # Compute global median of informed masses → use as floor
        if all_nonzero_masses:
            median_mass = torch.cat(all_nonzero_masses).median().item()
        else:
            median_mass = 1.0
        mass_floor = 0.01 * median_mass  # 1% of median

        # Second pass: clamp per-element masses and init velocities
        n_clamped = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                m = state["mass"]
                below = m < mass_floor
                n_clamped += below.sum().item()
                m.clamp_(min=mass_floor)

                # Maxwell-Boltzmann velocity with correct masses
                state["velocity"] = torch.randn_like(p.data) * (T / m).sqrt()
                state["prev_grad"] = None

        self._calibrated = True
        total_elements = sum(
            p.numel() for g in self.param_groups for p in g["params"]
        )
        print(
            f"LangevinSA calibrated over {n_steps} steps. "
            f"Median mass={median_mass:.2e}, floor={mass_floor:.2e}, "
            f"clamped {int(n_clamped)}/{total_elements} elements. "
            f"Mass range per param:"
        )
        for group in self.param_groups:
            for p in group["params"]:
                m = self.state[p]["mass"]
                print(
                    f"  [{p.shape}] mass: "
                    f"min={m.min().item():.2e}, "
                    f"median={m.median().item():.2e}, "
                    f"max={m.max().item():.2e}"
                )

    # ------------------------------------------------------------------
    # Physical-mass seeding (genuine thermostatted MD)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_physical_masses(self, mass_by_param, T=None):
        """Seed externally supplied (e.g. physical atomic) masses and
        initialise Maxwell-Boltzmann velocities, bypassing the grad²
        :meth:`calibrate` path.

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
        self._calibrated = True

    # ------------------------------------------------------------------
    # BAOAB step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure):
        """Perform one BAOAB Langevin dynamics step.

        Tracks the best-loss configuration and rolls back to it when the
        loss becomes non-finite or exceeds 3x the best loss seen so far.
        This prevents the dynamics from permanently damaging the structure
        while still allowing uphill exploration.

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
                    # Velocity: if calibrated, already set; otherwise init now
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
            # Rollback if loss explodes beyond 3x best
            if loss.item() > 3.0 * self._best_loss:
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
