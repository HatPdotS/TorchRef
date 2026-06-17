"""
Soft rank penalty on the ensemble's coordinate disorder.

Frozen-basis PCA truncation failed because the 3GR5 ensemble disorder is
high-rank (a near-flat SVD spectrum) — there is no low-dimensional subspace to
project onto, and the ensemble *mean* is unphysical. This target takes the
opposite tack: keep refining the full-complexity ensemble, but add a *soft*
penalty that progressively "purifies" it toward fewer effective modes, with the
X-ray data free to push back wherever extra modes are genuinely justified.

The penalty is the **nuclear norm** (sum of singular values) of the centered
per-member coordinate matrix ``Xc`` (shape ``(N, D)``, ``D = n_atoms * 3``) —
the standard convex surrogate for matrix rank::

    L_rank = ||Xc||_*  =  Σ_k σ_k,     Xc = X - mean_member(X)

Minimizing it shrinks the trailing singular values fastest (L1 on the spectrum →
sparsity in modes), collapsing the ensemble toward a low-rank disorder model.
Unlike hard PCA truncation, the mean is *not* frozen (it refines with the rest)
and nothing is discarded outright — the weight is a tunable knob (typically
ramped up over the run), so the free set decides how much rank to keep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchref.refinement.targets.base import ModelTarget

if TYPE_CHECKING:
    from .ensemble_model import EnsembleModel


class RankPenaltyTarget(ModelTarget):
    """Nuclear-norm (soft rank) penalty on the ensemble member spread.

    Parameters
    ----------
    model : EnsembleModel
        The ensemble; ``model.xyz_per_member`` is read each forward.
    normalize : bool
        If True (default), divide the nuclear norm by ``N`` so the loss is a
        per-member ("per-ASU"-ish) scale, consistent with the X-ray / Wilson /
        Amber per-ASU normalization in ``EnsembleRefinement._create_loss_state``.
    verbose : int
        Verbosity.
    """

    name: str = "rank_penalty"

    def __init__(
        self,
        model: "EnsembleModel" = None,
        mode: str = "nuclear",
        target_rank: int = 0,
        freeze_disp: float = 0.2,
        maxent_shrink: float = 1.0,
        maxent_div: float = 1.0,
        normalize: bool = True,
        verbose: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        mode : {"nuclear", "subspace", "entropy", "maxent", "diverse"}
            ``"diverse"``: the orthogonal participation⟂similarity pair.
            ``L = maxent_shrink·PR + maxent_div·sim``, where ``PR=(Σσ²)²/Σσ⁴`` is
            the (scale-invariant) participation ratio — minimized to concentrate
            variance into fewer effective modes (de-overfit) — and ``sim`` is the
            mean pairwise member RBF similarity (median-heuristic bandwidth) —
            minimized to spread the conformers apart (anti-collapse / anti-trap).
            The two act on different objects (eigenvalues vs member positions),
            so they don't fight; the similarity term stabilizes PR-minimization
            (which alone collapses to rank-1). Supersedes ``"maxent"`` (whose
            spectral-entropy "diversity" was on the same axis as participation).
            ``"maxent"``: max-diversity / min-effective-rank, K-free. Decouples
            spectrum *magnitude* from *shape*:
            ``L = maxent_shrink·(Σσ_k²/N)  −  maxent_div·H(p)``, where
            ``p_k = σ_k²/Σσ_j²`` and ``H = −Σ p_k ln p_k`` is the normalized
            spectral (Shannon) entropy. The shrink (trace, L2) term limits total
            disorder → de-overfit; the entropy term is scale-invariant and is
            *maximized* (subtracted), spreading variance across modes (diversity,
            anti-collapse) without fighting the shrink. Unlike ``"entropy"`` this
            is a stable equilibrium, not a runaway rank-collapser. Clean
            gradient (``p ln p → 0``; values-only SVD backward).
            ``"nuclear"`` (default): penalize ``Σ_k σ_k`` (shrinks all modes —
            reduces rank AND magnitude). ``"subspace"``: penalize the variance
            *outside* the top-``target_rank`` subspace, ``Σ_{k>K} σ_k²`` (the
            rank-K projection residual ``‖Xc − Xc_K‖²_F``) — restrains the
            ensemble onto a K-dimensional manifold, leaving the top-K modes
            free. ``"entropy"``: penalize the per-structure quasi-harmonic
            (Schlitter) conformational entropy
            ``S = ½ Σ_k ln(1 + α·μ_k)`` (nats/ASU), where ``μ_k = σ_k²/N`` is
            the per-mode positional variance (Å²) and ``α = 1/freeze_disp²``.
            The ``1/σ``-like gradient pulls hardest on the *smallest* modes, so
            minimizing it preferentially collapses the low-variance tail
            (dimensionality reduction) while sparing the dominant collective
            modes — and S is on the same per-ASU nats scale as the X-ray NLL,
            so the weight is a dimensionless fit-vs-disorder coupling.
        target_rank : int
            Retained subspace dimension ``K`` (``"subspace"`` mode only).
        freeze_disp : float
            Freeze-out RMS displacement (Å) for ``"entropy"`` mode: modes with
            RMS below this contribute ~0 entropy (frozen), above it count as
            full thermal entropy. Sets ``α = 1/freeze_disp²``. ~coordinate
            error at the working resolution (default 0.2 Å).
        normalize : bool
            Divide the penalty by ``N`` for a per-member scale. Ignored for
            ``"entropy"`` (already a per-structure quantity).
        """
        super().__init__(model=model, verbose=verbose)
        self._mode = str(mode)
        self._target_rank = int(target_rank)
        self._freeze_disp = float(freeze_disp)
        self._maxent_shrink = float(maxent_shrink)
        self._maxent_div = float(maxent_div)
        self._normalize = bool(normalize)

    def _centered(self) -> torch.Tensor:
        """Return the centered ``(N, D)`` member-coordinate matrix."""
        xyz = self._model.xyz_per_member                  # (N, n_atoms, 3)
        N = xyz.shape[0]
        X = xyz.reshape(N, -1)                            # (N, D)
        return X - X.mean(dim=0, keepdim=True)

    def forward(self) -> torch.Tensor:
        """Soft rank / subspace penalty on the centered member matrix.

        Both branches are functions of the singular *values* only, so their
        gradients (``U·diag(g)·Vᵀ``) are well-conditioned even when the
        spectrum is near-degenerate — we never backprop through the singular
        *vectors*.
        """
        Xc = self._centered()
        N = float(Xc.shape[0])
        if self._mode == "diverse":
            # ORTHOGONAL PAIR (participation ⟂ similarity), the corrected
            # diversity formulation. The earlier "maxent" used spectral entropy
            # H(p) as "diversity", but H is a function of the eigenVALUES only —
            # the SAME axis as participation — so maximizing it fought the rank
            # reduction (eff_rank climbed 58→80+). Here the two terms act on
            # genuinely different objects:
            #
            #  (1) PARTICIPATION  — minimize the participation ratio
            #      PR = (Σσ²)²/Σσ⁴ (the logged eff_rank). Scale-INVARIANT, a
            #      function of the eigenvalues only → concentrates variance into
            #      fewer effective modes (de-overfit), with no opinion on
            #      magnitude. Weight = maxent_shrink.
            #  (2) SIMILARITY    — minimize the mean pairwise member RBF
            #      similarity exp(−d_ij²/2ℓ²) on the conformer coordinates
            #      (gauge-invariant: depends on member POSITIONS, not the A/V
            #      factorization). Rewards member spread → anti-collapse /
            #      anti-kinetic-trap, and stabilizes (1) (pure PR-min collapses
            #      to rank-1 without it). ℓ² set by the median-distance heuristic
            #      each step so the kernel always sits in its sensitive band.
            #      Weight = maxent_div.
            #
            # Both are well-conditioned: PR uses singular VALUES only (no vector
            # backward); the RBF is smooth and bounded (0,1) (no runaway).
            s2 = torch.linalg.svdvals(Xc) ** 2
            PR = (s2.sum() ** 2) / (s2 ** 2).sum().clamp_min(1e-30)
            Nn = Xc.shape[0]
            d2 = torch.cdist(Xc, Xc) ** 2                   # (N, N), differentiable
            off = ~torch.eye(Nn, dtype=torch.bool, device=Xc.device)
            with torch.no_grad():
                bw = d2[off].median().clamp_min(1e-12)      # median-heuristic ℓ²
            sim = torch.exp(-d2 / (2.0 * bw))
            sim_off = sim[off].mean()                       # mean pairwise sim ∈ (0,1)
            return self._maxent_shrink * PR + self._maxent_div * sim_off
        if self._mode == "maxent":
            # Max-diversity / min-effective-rank (K-free): decouple magnitude
            # (shrink, de-overfit) from shape (spectral entropy, maximized for
            # diversity / anti-collapse). H is scale-invariant so the two terms
            # don't fight. p ln p → 0 ⇒ no 1/σ blow-up.
            s = torch.linalg.svdvals(Xc)
            s2 = s ** 2
            shrink = s2.sum() / N
            p = s2 / s2.sum().clamp_min(1e-30)
            H = -(p * p.clamp_min(1e-30).log()).sum()
            return self._maxent_shrink * shrink - self._maxent_div * H
        if self._mode == "entropy":
            # Per-structure quasi-harmonic (Schlitter) conformational entropy:
            #   S = ½ Σ_k ln(1 + α·μ_k),  μ_k = σ_k²/N (Å²),  α = 1/freeze_disp².
            # Already a per-structure quantity (nats/ASU) — no /N normalization.
            s = torch.linalg.svdvals(Xc)
            mu = (s ** 2) / N                              # per-mode variance (Å²)
            alpha = 1.0 / (self._freeze_disp ** 2)         # Å^-2
            return 0.5 * torch.log1p(alpha * mu).sum()
        if self._mode == "subspace":
            # Penalize variance outside the top-K subspace:
            #   Σ_{k>K} σ_k²  =  ‖Xc − Xc_K‖²_F   (rank-K projection residual)
            s = torch.linalg.svdvals(Xc)                  # sorted descending
            K = max(0, int(self._target_rank))
            tail = s[K:]
            pen = (tail ** 2).sum()
        else:  # "nuclear"
            pen = torch.linalg.matrix_norm(Xc, ord="nuc")  # Σ_k σ_k
        if self._normalize:
            pen = pen / N
        return pen

    @torch.no_grad()
    def spectrum_diagnostics(self) -> dict:
        """Read-only mode-spectrum diagnostics for logging (no grad).

        Returns the participation ratio (``(Σσ²)² / Σσ⁴`` — an effective mode
        count), the stable rank (``||Xc||_F² / σ_max²``), and the fraction of
        variance in the top mode. These show the "purification" as the penalty
        ramps up.
        """
        Xc = self._centered().detach().to(torch.float64)
        s = torch.linalg.svdvals(Xc)                      # (min(N, D),)
        s2 = s ** 2
        total = s2.sum().clamp_min(1e-30)
        part_ratio = float((s2.sum() ** 2) / (s2 ** 2).sum().clamp_min(1e-30))
        stable_rank = float(total / s2.max().clamp_min(1e-30))
        top1 = float(s2.max() / total)
        N = float(Xc.shape[0])
        mu = s2 / N
        alpha = 1.0 / (self._freeze_disp ** 2)
        entropy = float(0.5 * torch.log1p(alpha * mu).sum())  # nats/ASU
        p = s2 / total
        spectral_H = float(-(p * p.clamp_min(1e-30).log()).sum())  # normalized
        # Mean pairwise member RBF similarity (median-heuristic bandwidth) — the
        # "diverse"-mode similarity term; ∈ (0,1), low = members well-spread.
        Nn = Xc.shape[0]
        if Nn > 1:
            d2 = torch.cdist(Xc, Xc) ** 2
            off = ~torch.eye(Nn, dtype=torch.bool, device=Xc.device)
            bw = d2[off].median().clamp_min(1e-12)
            mean_pairwise_sim = float(torch.exp(-d2 / (2.0 * bw))[off].mean())
        else:
            mean_pairwise_sim = float("nan")
        return {
            "participation_ratio": part_ratio,
            "stable_rank": stable_rank,
            "top_mode_frac": top1,
            "nuclear_norm": float(s.sum()),
            "frob_norm": float(total.sqrt()),
            "conf_entropy_nats": entropy,
            "spectral_entropy": spectral_H,
            "mean_pairwise_sim": mean_pairwise_sim,
            "total_variance": float(total),
        }
