"""Joint relative scaling of observed datasets without a privileged reference.

DatasetScaler owns all fitted corrections. ScaledDataset exposes one correction
through the reflection-data interface; no optimization state lives on raw datasets.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchref.io import ReflectionData

import torch
from torch import nn

from torchref.base.targets.xray_likelihoods import SIGMA_FLOOR_ABS, SIGMA_FLOOR_FRAC
from torchref.config import get_float_dtype, get_int_dtype, normalize_device
from torchref.utils.device_mixin import DeviceMixin


def _identity_hkl(data):
    """Keep Bijvoet observations separate while matching dataset identities."""
    return data.hkl if data.friedel_merged else data._hkl_for_sf()


class DatasetScaler(DeviceMixin, nn.Module):
    """Fit N observed datasets to a shared sigma-weighted amplitude consensus.

    Parameters
    ----------
    datasets : Mapping[str, ReflectionData]
        At least two raw datasets in the same space-group setting and Friedel
        convention. Sources are copied without mutation; membership is fixed.
    device : torch.device or str, optional
        Computation device, defaulting to the first dataset's device. Prepared
        fitting arrays and owned copies move here without moving source data.

    Notes
    -----
    Corrections have zero mean in log space over datasets, including anisotropy.
    The six quadratic coefficients use dimensionless, normalized Miller indices;
    they are not Cartesian atomic displacement parameters. Every corrected read
    depends on all parameter rows through centering. ``fit`` freezes parameters
    when it finishes; call ``requires_grad_(True)`` for custom differentiable use.
    """

    def __init__(
        self,
        datasets: Mapping[str, "ReflectionData"],
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if len(datasets) < 2:
            raise ValueError("Dataset scaling requires at least two datasets")
        self.keys = tuple(datasets)
        self.datasets = {}
        for key, data in datasets.items():
            raw = data.raw_data() if hasattr(data, "raw_data") else data
            owned = raw.__select__(torch.arange(len(raw), device=raw.device))
            owned.source = None
            owned.spacegroup = raw.spacegroup.copy()
            self.datasets[key] = owned

        self.device = normalize_device(
            device if device is not None else next(iter(datasets.values())).device
        )
        self.dtype_float = get_float_dtype()
        self.raw_parameters = nn.Parameter(
            torch.zeros((len(self.keys), 7), device=self.device, dtype=self.dtype_float)
        )
        for name in ("hkl", "amplitudes", "sigmas", "fit_mask", "hkl_scale"):
            self.register_buffer(name, None)
        self.n_contrasts = 0
        self._initialized = False
        self.to(self.device)
        self.prepare()

    @property
    def corrections(self) -> torch.Tensor:
        """Centered coefficients, shape (N, 7), in dimensionless log units."""
        return self.raw_parameters - self.raw_parameters.mean(dim=0, keepdim=True)

    def design(self, hkl: torch.Tensor) -> torch.Tensor:
        """Return the log-correction basis, shape (H, 7), for integer HKL (H, 3)."""
        q = hkl.to(device=self.device, dtype=self.raw_parameters.dtype) / self.hkl_scale
        h, k, l = q.unbind(dim=-1)
        return torch.stack(
            (torch.ones_like(h), h * h, k * k, l * l, 2 * h * k, 2 * h * l, 2 * k * l),
            dim=-1,
        )

    def log_corrections(self, hkl: torch.Tensor) -> torch.Tensor:
        """Return dimensionless log amplitude corrections, shape (N, H)."""
        return self.corrections @ self.design(hkl).T

    def forward(self, key: str, hkl: torch.Tensor) -> torch.Tensor:
        """Return positive amplitude factors (H,) for dataset key and HKL (H, 3)."""
        row = self.keys.index(key)
        return (self.design(hkl) @ self.corrections[row]).exp()

    def prepare(self) -> None:
        """Prepare training arrays and reject disconnected or unidentified fits.

        This reads current source masks and measurements. No free/validation
        observation enters initialization, uncertainty floors or the objective.
        Changes to reflection sets or fit masks require a new scaler once fitted.
        """
        data = list(self.datasets.values())
        if tuple(self.datasets) != self.keys:
            raise ValueError("Dataset membership changed; construct a new scaler")
        symmetry = {d.spacegroup.xhm for d in data}
        if len(symmetry) != 1 or len({d.friedel_merged for d in data}) != 1:
            raise ValueError(
                "Datasets require compatible symmetry settings and Friedel conventions"
            )
        hkls = [_identity_hkl(d).to(self.device) for d in data]
        hkl, inverse = torch.unique(torch.cat(hkls), dim=0, return_inverse=True)
        shape = (len(data), len(hkl))
        amplitudes = torch.zeros(shape, device=self.device, dtype=self.dtype_float)
        sigmas = torch.ones_like(amplitudes)
        valid = torch.zeros(shape, device=self.device, dtype=torch.bool)
        held_out = torch.zeros(len(hkl), device=self.device, dtype=torch.bool)
        start = 0
        for row, ds in enumerate(data):
            if ds.F_raw is None or ds.F_sigma_raw is None:
                raise ValueError(f"Dataset {self.keys[row]!r} requires F and SIGF")
            idx = inverse[start : start + len(ds)]
            start += len(ds)
            if len(torch.unique(idx)) != len(idx):
                raise ValueError(
                    f"Dataset {self.keys[row]!r} contains duplicate reflection identities"
                )
            f = ds.F_raw.detach().to(amplitudes)
            sigma = ds.F_sigma_raw.detach().to(sigmas)
            present = ds.masks().to(device=self.device, dtype=torch.bool)
            usable = present & torch.isfinite(f) & torch.isfinite(sigma) & (sigma > 0)
            work = ds.work.mask.to(self.device)
            held_out[idx] |= present & ~work
            valid[row, idx] = usable
            amplitudes[row, idx] = torch.where(usable, f, torch.zeros_like(f))
            sigmas[row, idx] = torch.where(usable, sigma, torch.ones_like(sigma))
        mask = valid & ~held_out.unsqueeze(0)
        mask &= mask.sum(dim=0, keepdim=True) >= 2
        reached = {0}
        for _ in data:
            reached |= {
                j
                for i in tuple(reached)
                for j in range(len(data))
                if bool((mask[i] & mask[j]).any())
            }
        if len(reached) != len(data):
            raise ValueError(
                "Dataset work-set overlap is disconnected; relative scales are unidentified"
            )
        for row in range(len(data)):
            floor = (sigmas[row, mask[row]].median() * SIGMA_FLOOR_FRAC).clamp_min(
                SIGMA_FLOOR_ABS
            )
            sigmas[row] = sigmas[row].clamp_min(floor)
        if self.hkl_scale is None:
            self.hkl_scale = (
                hkl[mask.any(dim=0)].to(amplitudes).abs().amax(dim=0).clamp_min(1)
            )
        self.hkl, self.amplitudes, self.sigmas, self.fit_mask = (
            hkl,
            amplitudes,
            sigmas,
            mask,
        )
        self.n_contrasts = int((mask.sum(dim=0) - 1).clamp_min(0).sum())
        # Rank concerns only geometry and overlap, not observed amplitudes. The
        # small-column check runs on CPU because MPS has no SVD implementation.
        design = self.design(hkl).detach().cpu()
        mask_cpu = mask.cpu()
        first = mask_cpu.to(get_int_dtype()).argmax(dim=0)
        blocks = []
        for row in range(len(data)):
            selected = mask_cpu[row] & (first != row)
            if not bool(selected.any()):
                continue
            block = torch.zeros((int(selected.sum()), len(data), 7), dtype=design.dtype)
            block[:, row] = design[selected]
            block[torch.arange(len(block)), first[selected]] = -design[selected]
            blocks.append(block[:, :-1].reshape(len(block), -1))
        rank = int(torch.linalg.matrix_rank(torch.cat(blocks))) if blocks else 0
        if rank != 7 * (len(data) - 1):
            raise ValueError(
                "Overlapping reflections do not identify overall scale and six anisotropic coefficients"
            )

    def initialize(self) -> None:
        """Seed overall log scales from robust pairwise log-amplitude ratios."""
        rows, values = [], []
        n = len(self.keys)
        for i in range(n):
            for j in range(i):
                mask = (
                    self.fit_mask[i]
                    & self.fit_mask[j]
                    & (self.amplitudes[i] > 0)
                    & (self.amplitudes[j] > 0)
                )
                if not bool(mask.any()):
                    continue
                row = torch.zeros(n, dtype=self.dtype_float)
                row[i], row[j] = 1, -1
                rows.append(row)
                values.append(
                    (self.amplitudes[j, mask].log() - self.amplitudes[i, mask].log())
                    .median()
                    .cpu()
                )
        rows.append(torch.ones(n, dtype=self.dtype_float))
        values.append(torch.zeros((), dtype=self.dtype_float))
        initial = torch.linalg.lstsq(torch.stack(rows), torch.stack(values)).solution
        with torch.no_grad():
            self.raw_parameters.zero_()
            self.raw_parameters[:, 0].copy_(initial.to(self.raw_parameters))
        self._initialized = True

    def fit(self, nsteps: int = 10, max_iter: int = 100) -> dict:
        """Fit joint corrections with L-BFGS and freeze the resulting parameters.

        Parameters
        ----------
        nsteps : int
            Number of outer L-BFGS steps.
        max_iter : int
            Maximum iterations per outer step.

        Returns
        -------
        dict
            Initial/final normalized loss, contrast count and centered coefficients.
        """
        from torchref.refinement.loss_state import LossState
        from torchref.refinement.targets.dataset_scaling import DatasetScalingTarget

        if nsteps < 1 or max_iter < 1:
            raise ValueError("nsteps and max_iter must be positive")
        self.prepare()
        if not self._initialized:
            self.initialize()
        saved = self.raw_parameters.detach().clone()
        self.requires_grad_(True)
        target = DatasetScalingTarget(self)
        state = LossState(device=self.device)
        state.register_target("scaling/datasets", target)
        before = float(target().detach())
        optimizer = torch.optim.LBFGS(
            self.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        try:
            state.run(optimizer, nsteps=nsteps, log=False, context="dataset_scaler.fit")
            after = float(target().detach())
            if not torch.isfinite(self.raw_parameters).all() or not torch.isfinite(
                torch.tensor(after)
            ):
                raise RuntimeError(
                    "Dataset scaling produced non-finite parameters or loss"
                )
        except Exception:
            with torch.no_grad():
                self.raw_parameters.copy_(saved)
            raise
        finally:
            self.requires_grad_(False)
        return {
            "loss_before": before,
            "loss_after": after,
            "n_contrasts": self.n_contrasts,
            "corrections": dict(
                zip(self.keys, self.corrections.detach().cpu().tolist())
            ),
        }

    def get_state(self) -> dict:
        """Return raw source states and the shared fitted parameter state."""
        return {
            "datasets": {k: d._get_state() for k, d in self.datasets.items()},
            "parameters": self.raw_parameters.detach().cpu(),
            "hkl_scale": self.hkl_scale.detach().cpu(),
            "initialized": self._initialized,
        }

    @classmethod
    def from_state(
        cls, state: dict, device: torch.device | str | None = None
    ) -> "DatasetScaler":
        """Restore a shared scaler and its raw sources on the requested device."""
        from torchref.io.datasets.reflection_data import ReflectionData

        obj = cls(
            {
                k: ReflectionData._from_state(dict(v), device)
                for k, v in state["datasets"].items()
            },
            device=device,
        )
        with torch.no_grad():
            obj.raw_parameters.copy_(state["parameters"].to(obj.raw_parameters))
            obj.hkl_scale.copy_(state["hkl_scale"].to(obj.hkl_scale))
        obj._initialized = state["initialized"]
        obj.requires_grad_(False)
        return obj
