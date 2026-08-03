"""Structure-driven model-error estimation (Fisher-information sigma_m).

The second of two independent routes to a per-reflection model-error estimate:

* **data-driven** --
  :class:`~torchref.refinement.model_error_estimation.sigma_a.SigmaAEstimator`
  infers it from data-model disagreement per resolution shell;
* **structure-driven** -- this module predicts it from the *structure alone*.

It never sees ``F_obs`` or ``F_calc``. The inputs are resolution, a single scalar mean
``sigma_obs``, a validity mask, the ITC92 scattering parameters and the current per-atom
B factors -- so the estimate is available before any comparison with the data, which is
what makes it a genuinely different quantity rather than a reparametrisation.

The two were measured to agree on *shape* (Spearman 0.88-0.99 against ``beta``) but to
differ by ~15x in *magnitude*, which is why a caller-supplied scale exists at all. Keeping
both behind matching interfaces makes that comparison a first-class capability instead of
a one-off diagnostic.

Extracted verbatim (numerics unchanged) from the retired ``BhattacharyyaXrayTarget``,
whose *loss* was removed while this estimation was kept. Three things changed in the move:

1. **Plain tensors in, plain tensors out.** The original reached into
   ``self._scaler._s_half_sq``, ``self._data.get_corrected_data()`` and
   ``self._model.adp()``. Matching :class:`SigmaAEstimator`'s deliberate avoidance of
   ``ReflectionData``/``Scaler`` coupling (it exists to prevent an import cycle), the
   tensors are now passed in.
2. **No ``no_grad`` inside.** The original wrapped the estimate in ``no_grad``, but
   ``RiceSigmaMXrayTarget`` deliberately does not -- it needs gradients to reach the B
   factors *through* sigma_m. That choice belongs to the caller, so it is not made here.
3. **The cache is invalidated.** The original set ``_initialized = True`` once and never
   reset it, so a changed ``d_min``, a swapped ``ReflectionData``, added/removed atoms or
   a changed iso/aniso partition silently reused stale tables. Now fingerprinted, in the
   style of ``ReflectionData._subset_fingerprint``.

The tables also no longer live as ``nn.Module`` buffers, which keeps ``exp_table``
(``b_grid_n x N_refl``, i.e. 100*N floats) out of every checkpoint. The cost is that
device placement is explicit rather than inherited.
"""

from typing import Optional, Tuple

import torch


def _fingerprint(*tensors: Optional[torch.Tensor]) -> tuple:
    """Identity of the tensors the cached tables were built from.

    Uses ``(data_ptr, _version)`` per tensor -- the idiom already used by
    ``ReflectionData._subset_fingerprint`` -- plus shape/device/dtype, so an in-place
    edit, a reallocation or a device move all invalidate.
    """
    out = []
    for t in tensors:
        if t is None:
            out.append(None)
        else:
            out.append(
                (tuple(t.shape), str(t.device), str(t.dtype), t.data_ptr(), t._version)
            )
    return tuple(out)


class SigmaMEstimator:
    """Per-reflection model error predicted from the structure.

    Usage mirrors :class:`SigmaAEstimator`: construct cheaply, then call. Construction
    takes **no data and no model**, so a target can own one before its inputs exist
    (``RiceSigmaMXrayTarget`` is built bare in tests and relies on this).

    >>> est = SigmaMEstimator()                        # doctest: +SKIP
    >>> est.prepare(s_half_sq, sigma_obs, validity, A_iso, B_iso)   # doctest: +SKIP
    >>> sigma_m_sq = est.sigma_m_sq(b_iso)             # differentiable in b_iso

    The returned variance is **unscaled**. The structure-driven estimate is on the raw
    form-factor scale (``f_k^2`` carries electrons squared) and is not guaranteed
    commensurate with a scaler-scaled ``sigma_obs`` -- the measured ~15x magnitude offset
    against ``beta`` is exactly this. Applying a calibration is therefore the caller's
    job, and callers that refine one (``RiceSigmaMXrayTarget``) must not have a second
    scale silently applied underneath them.

    Parameters
    ----------
    b_grid_min, b_grid_max, b_grid_n
        Log-spaced B-factor grid, in A^2. It serves two purposes: the axis of the
        precomputed ``exp(-2 B s_half^2)`` table, and the axis of the soft histogram over
        atoms. Wider or finer costs memory linearly in ``b_grid_n``.
    """

    def __init__(
        self,
        b_grid_min: float = 1.0,
        b_grid_max: float = 200.0,
        b_grid_n: int = 100,
    ):
        self.b_grid_min = float(b_grid_min)
        self.b_grid_max = float(b_grid_max)
        self.b_grid_n = int(b_grid_n)
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Drop the cached tables so the next :meth:`prepare` rebuilds them."""
        self._fp = None
        self.b_grid = None
        self._log_b_min = self._log_b_max = self._log_b_step = None
        self.s_sq = None
        self.s_4 = None
        self.f_sq_kh = None
        self.exp_table = None
        self.g_w_table = None
        self.g_4_table = None
        self.atom_to_element = None
        self.sigma_d_mean = None

    @property
    def ready(self) -> bool:
        return self.exp_table is not None

    # ------------------------------------------------------------------ setup

    def prepare(
        self,
        s_half_sq: torch.Tensor,
        sigma_obs: torch.Tensor,
        validity: torch.Tensor,
        A_iso: torch.Tensor,
        B_iso: torch.Tensor,
    ) -> None:
        """Build (or reuse) the resolution/element tables.

        Everything here is independent of the *moving* structure -- it depends on the
        data, the resolution grid and the model's element composition and iso partition,
        but not on coordinates or B factors. So it is built once and reused across
        refinement steps, and rebuilt only when one of those actually changes.

        Parameters
        ----------
        s_half_sq
            ``(|s|/2)**2`` per reflection, i.e. the scaler's convention.
        sigma_obs
            Per-reflection experimental sigma. Only its valid-set mean enters, as the
            Fisher-information normaliser.
        validity
            Boolean per-reflection validity. Invalid reflections are zeroed out of the
            tables rather than dropped, so every array stays aligned to ``data.hkl``.
        A_iso, B_iso
            ITC92 five-Gaussian scattering parameters of the isotropic atoms,
            ``(n_iso, 5)`` each. Rows are hashed to identify unique element types.
        """
        fp = _fingerprint(s_half_sq, sigma_obs, validity, A_iso, B_iso)
        if self._fp == fp and self.ready:
            return

        device, dtype = s_half_sq.device, s_half_sq.dtype
        s_half_sq = s_half_sq.to(device=device, dtype=dtype)
        s_sq = 4.0 * s_half_sq

        valid_f = validity.to(torch.bool).to(device).to(dtype)
        n_valid = valid_f.sum().clamp(min=1.0)
        sigma_obs = sigma_obs.to(device=device, dtype=dtype)
        self.sigma_d_mean = (sigma_obs * valid_f).sum() / n_valid
        mean_sigma_sq = ((sigma_obs**2 * valid_f).sum() / n_valid).clamp(min=1e-12)

        self.b_grid = torch.exp(
            torch.linspace(
                float(torch.log(torch.tensor(self.b_grid_min))),
                float(torch.log(torch.tensor(self.b_grid_max))),
                self.b_grid_n,
                device=device,
                dtype=dtype,
            )
        )
        self._log_b_min = float(torch.log(self.b_grid[0]))
        self._log_b_max = float(torch.log(self.b_grid[-1]))
        self._log_b_step = (self._log_b_max - self._log_b_min) / (self.b_grid_n - 1)

        self.s_sq = s_sq
        self.s_4 = s_sq * s_sq

        # --- element-type grid: one row per unique (A, B) ITC92 pair -----------
        A_iso = A_iso.to(device=device, dtype=dtype)
        B_iso = B_iso.to(device=device, dtype=dtype)
        ab_rows = torch.cat([A_iso, B_iso], dim=-1)
        unique_rows, atom_to_element = torch.unique(
            ab_rows, dim=0, return_inverse=True
        )
        element_A, element_B = unique_rows[:, :5], unique_rows[:, 5:]
        self.atom_to_element = atom_to_element.to(device=device)

        # f_k(s_h) = sum_m A_km exp(-B_km s_half_sq)
        expon_f = (-element_B.unsqueeze(-1) * s_half_sq.view(1, 1, -1)).clamp(
            min=-80.0, max=80.0
        )
        f_kh = (element_A.unsqueeze(-1) * torch.exp(expon_f)).sum(dim=1)
        self.f_sq_kh = f_kh * f_kh

        # --- exp(-2 B s_half^2) over the B grid, chunked to bound peak memory ---
        n_refl = s_sq.shape[0]
        exp_table = torch.empty(self.b_grid_n, n_refl, device=device, dtype=dtype)
        chunk = 32
        for start in range(0, self.b_grid_n, chunk):
            end = min(start + chunk, self.b_grid_n)
            expon = (
                -2.0 * self.b_grid[start:end].unsqueeze(-1) * s_half_sq.unsqueeze(0)
            ).clamp(min=-80.0, max=80.0)
            exp_table[start:end] = torch.exp(expon)
        invalid = (~validity.to(torch.bool)).nonzero(as_tuple=True)[0]
        if invalid.numel() > 0:
            exp_table[:, invalid] = 0.0
        self.exp_table = exp_table

        # --- per-element Fisher-information tables, (K, b_grid_n) --------------
        inv_sig_sq_valid = valid_f / mean_sigma_sq
        w_w = s_sq.unsqueeze(0) * self.f_sq_kh * inv_sig_sq_valid.unsqueeze(0)
        w_4 = self.s_4.unsqueeze(0) * self.f_sq_kh * inv_sig_sq_valid.unsqueeze(0)
        exp_table_T = exp_table.transpose(0, 1)
        self.g_w_table = torch.matmul(w_w, exp_table_T)
        self.g_4_table = torch.matmul(w_4, exp_table_T)

        self._fp = fp

    # ------------------------------------------------------------------ estimate

    def _log_b_index(self, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(idx_lo, frac)`` for linear interpolation in log-B space."""
        log_b = torch.log(b.clamp(min=1e-6))
        log_b_clamped = torch.clamp(log_b, self._log_b_min, self._log_b_max)
        idx_f = (log_b_clamped - self._log_b_min) / self._log_b_step
        idx_lo = idx_f.floor().long().clamp(0, self.b_grid_n - 2)
        frac = (idx_f - idx_lo.to(idx_f.dtype)).clamp(0.0, 1.0)
        return idx_lo, frac

    def _element_b_histogram(self, b: torch.Tensor) -> torch.Tensor:
        """Soft 2-D histogram of the iso atoms over (element type, log B).

        Each atom splits its unit weight between its two log-B neighbours, which is what
        keeps the estimate differentiable in ``b``: a hard bin assignment would give it
        zero gradient almost everywhere.
        """
        K = self.f_sq_kh.shape[0]
        n_b = self.b_grid_n
        idx_lo, frac = self._log_b_index(b)
        elem = self.atom_to_element
        flat_lo = elem * n_b + idx_lo
        flat_hi = elem * n_b + (idx_lo + 1)
        hist = torch.zeros(K * n_b, device=b.device, dtype=b.dtype)
        hist = hist.scatter_add(0, flat_lo, 1.0 - frac)
        hist = hist.scatter_add(0, flat_hi, frac)
        return hist.view(K, n_b)

    def sigma_m_sq(self, b_iso: torch.Tensor) -> torch.Tensor:
        """Per-reflection model-error variance, ``(N_refl,)``, UNSCALED.

        Differentiable in ``b_iso``: the caller decides whether gradients should flow
        (``RiceSigmaMXrayTarget`` wants them, so it must not be wrapped in ``no_grad``
        here). Apply any calibration outside.
        """
        if not self.ready:
            raise RuntimeError(
                "SigmaMEstimator.prepare() must be called before sigma_m_sq()"
            )
        hist = self._element_b_histogram(b_iso)
        atom_factor_w = torch.matmul(hist / self.g_w_table.clamp(min=1e-30),
                                     self.exp_table)
        atom_factor_4 = torch.matmul(hist / self.g_4_table.clamp(min=1e-30),
                                     self.exp_table)
        per_type = self.f_sq_kh * (
            3.0 * self.s_sq.unsqueeze(0) * atom_factor_w
            + self.s_4.unsqueeze(0) * atom_factor_4
        )
        return per_type.sum(dim=0).clamp(min=1e-12)

    def sigma_m(self, b_iso: torch.Tensor) -> torch.Tensor:
        """``sqrt`` of :meth:`sigma_m_sq`."""
        return torch.sqrt(self.sigma_m_sq(b_iso))
