"""
:class:`FcalcDataset` -- container for calculated structure factors.

Generates its own HKL set from cell/spacegroup/resolution, decomposes complex
Fcalc into amplitude and phase, and exports to MTZ either as model columns or
as pseudo-observations.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd
import torch

from torchref.config import get_default_device, get_float_dtype, normalize_device
from torchref.symmetry import Cell, SpaceGroup, SpaceGroupLike

from .base import CrystalDataset

if TYPE_CHECKING:
    from .reflection_data import ReflectionData


@dataclass
class FcalcDataset(CrystalDataset):
    """
    Dataset for storing calculated structure factors.

    Unlike :class:`CrystalDataset`, ``spacegroup`` here holds a
    ``torchref.symmetry.SpaceGroup`` object, not a string.

    Parameters
    ----------
    hkl : torch.Tensor, optional
        Miller indices of shape (N, 3).
    resolution : torch.Tensor, optional
        Resolution per reflection of shape (N,).
    cell : Cell, optional
        Unit cell object.
    spacegroup : SpaceGroup, optional
        Space group object (torchref.symmetry.SpaceGroup).
    fcalc : torch.Tensor, optional
        Complex structure factors of shape (N,).
    fcalc_amp, fcalc_phase : torch.Tensor, optional
        ``|Fcalc|`` and its phase in radians, shape (N,); normally derived by
        :meth:`set_fcalc` rather than passed.
    device : torch.device
        Device for tensors.
    """

    # Override spacegroup to use torchref.symmetry.SpaceGroup (not gemmi)
    spacegroup: Optional[SpaceGroup] = None  # type: ignore[assignment]

    # Fcalc-specific fields
    fcalc: Optional[torch.Tensor] = None  # Complex (N,)
    fcalc_amp: Optional[torch.Tensor] = None  # |Fcalc| (N,)
    fcalc_phase: Optional[torch.Tensor] = None  # Phase in radians (N,)
    fobs_sigma: Optional[torch.Tensor] = None  # Amp-space sigma (N,), set by add_noise

    @staticmethod
    def from_cell_and_resolution(
        cell: Union[torch.Tensor, List[float], Cell],
        spacegroup: SpaceGroupLike,
        d_min: float = 2.0,
        d_max: Optional[float] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> "FcalcDataset":
        """
        Create FcalcDataset with HKL generated to given resolution.

        Parameters
        ----------
        cell : torch.Tensor, list, or Cell
            Unit cell [a, b, c, alpha, beta, gamma] or Cell object. A ``Cell``
            is moved IN PLACE if ``device`` disagrees with it.
        spacegroup : SpaceGroupLike
            Space group (str, int, gemmi.SpaceGroup, or torchref.symmetry.SpaceGroup).
        d_min : float, optional
            High resolution limit in Angstroms. Default is 2.0.
        d_max : float, optional
            Low resolution limit in Angstroms. If provided, reflections
            with d-spacing > d_max are removed.
        device : torch.device, optional
            Target device. If None, defaults to ``get_default_device()``.
        dtype : torch.dtype, optional
            Float dtype for tensors. If None, defaults to ``get_float_dtype()``.

        Returns
        -------
        FcalcDataset
            New dataset with HKL (unique ASU reflections) and resolution set.
        """
        import gemmi

        from torchref.base.reciprocal import get_d_spacing

        # With a Cell in hand and no device requested, follow the Cell rather
        # than the global default, or a CPU-default host silently relocates it.
        if device is None and isinstance(cell, Cell):
            device = cell.device
        device = normalize_device(device)
        if dtype is None:
            dtype = get_float_dtype()

        if isinstance(cell, Cell):
            # ``Cell.to`` is in-place, so an explicit disagreeing device moves
            # the *caller's* object (the resolve_device contract).
            cell_obj = cell.to(device=device)
            cell_tensor = cell_obj.data
        else:
            if not isinstance(cell, torch.Tensor):
                cell_tensor = torch.tensor(cell, dtype=dtype, device=device)
            else:
                cell_tensor = cell.to(device=device, dtype=dtype)
            cell_obj = Cell(cell_tensor, dtype=dtype, device=device)

        if isinstance(spacegroup, SpaceGroup):
            sg_obj = spacegroup
        else:
            sg_obj = SpaceGroup(spacegroup)

        cell_list = cell_tensor.cpu().tolist()
        gemmi_cell = gemmi.UnitCell(
            cell_list[0],
            cell_list[1],
            cell_list[2],
            cell_list[3],
            cell_list[4],
            cell_list[5],
        )
        gemmi_sg = sg_obj._gemmi

        # make_miller_array returns unique HKL for the asymmetric unit only.
        hkl_list = gemmi.make_miller_array(gemmi_cell, gemmi_sg, d_min)
        hkl = torch.tensor(hkl_list, dtype=torch.int32, device=device)  # dtype-ok: hkl Miller indices; fixed int32 crystallographic representation, not model-precision data

        resolution = get_d_spacing(hkl.float(), cell_tensor)

        if d_max is not None:
            mask = resolution <= d_max
            hkl = hkl[mask]
            resolution = resolution[mask]

        print(f"Generated dataset with {len(hkl)} reflections.")

        return FcalcDataset(
            hkl=hkl,
            resolution=resolution,
            cell=cell_obj,
            spacegroup=sg_obj,
            device=device,
        )

    def set_fcalc(self, fcalc: torch.Tensor) -> None:
        """
        Assign complex Fcalc, also setting ``fcalc_amp`` and ``fcalc_phase``.

        Parameters
        ----------
        fcalc : torch.Tensor
            Complex structure factors with shape (N,).

        Raises
        ------
        ValueError
            If HKL is unset or ``fcalc`` has a different length.
        """
        if self.hkl is None:
            raise ValueError("HKL not set. Cannot assign Fcalc without HKL indices.")

        if fcalc.shape[0] != len(self.hkl):
            raise ValueError(
                f"Fcalc length {fcalc.shape[0]} != HKL length {len(self.hkl)}"
            )

        self.fcalc = fcalc.to(device=self.device)
        self.fcalc_amp = torch.abs(fcalc).to(device=self.device)
        self.fcalc_phase = torch.angle(fcalc).to(device=self.device)

    def add_noise(
        self,
        reference: Optional["ReflectionData"] = None,
        sigma_lin: float = 0.0,
        sigma_mul: float = 0.05,
        sigma_abs: float = 0.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "FcalcDataset":
        """
        Return a copy with Gaussian **intensity** noise, as the mean of two half-datasets.

        Two sigma sources, chosen by ``reference``:

        **Reference-driven (preferred).** Given a ``ReflectionData`` on the same HKL list,
        its per-reflection ``I_sigma`` is grafted directly. Right when the Fcalc is already
        on the reference's absolute scale.

        **Parametric.** Otherwise a three-term variance model::

            sigma_I^2 = sigma_lin^2 * I  +  sigma_mul^2 * I^2  +  (sigma_abs * <I>_outer)^2

        Either way two independent draws are made and averaged::

            I_h{1,2}     = I + N(0, sigma_I)
            I_mean       = (I_h1 + I_h2) / 2
            sigma_I_mean = sigma_I / sqrt(2)

        R-split and Pearson CC between the halves are reported, which makes the two draws
        a ready-made split-half pair rather than only a noise model.

        **Negative intensities are kept.** ``I_mean`` and ``sigma_I_mean`` are stored
        unclamped on the returned dataset, and only the *amplitude* is clamped, because an
        amplitude cannot be negative. Clamping the intensity instead would put a positive
        bias on exactly the weak reflections where noise dominates -- the same shape as a
        genuine positive perturbation, and enough to swamp effects of order 1e-3.

        The amplitude sigma is propagated against the **true** amplitude, not the noisy
        one, so it is not warped per draw.

        Parameters
        ----------
        reference : ReflectionData, optional
            Sigma donor. Requires ``torch.equal(self.hkl, reference.hkl)``.
        sigma_lin, sigma_mul, sigma_abs : float, optional
            Parametric coefficients, ignored when ``reference`` is given.
        seed : int, optional
            Seed for reproducibility. ``None`` uses the global RNG.
        verbose : bool, optional
            Print the R-split and CC between halves. Default True.

        Returns
        -------
        FcalcDataset
            New dataset carrying the noisy complex Fcalc, the unclamped ``I`` /
            ``I_sigma``, and ``fobs_sigma``.
        """
        if self.fcalc is None or self.fcalc_amp is None or self.fcalc_phase is None:
            raise ValueError("No Fcalc values set. Call set_fcalc() first.")

        shape = self.fcalc_amp.shape
        dtype = self.fcalc_amp.dtype
        dev = self.device
        if seed is not None:
            g = torch.Generator(device=dev).manual_seed(int(seed))
            randn1 = torch.randn(shape, dtype=dtype, device=dev, generator=g)
            randn2 = torch.randn(shape, dtype=dtype, device=dev, generator=g)
        else:
            randn1 = torch.randn(shape, dtype=dtype, device=dev)
            randn2 = torch.randn(shape, dtype=dtype, device=dev)

        intensity = self.fcalc_amp**2

        if reference is not None:
            if reference.I_sigma is None:
                raise ValueError(
                    "reference.I_sigma is None. Load the reference with intensity + "
                    "sigma columns (e.g. load_crystfel_hkl) before passing it here."
                )
            ref_hkl = reference.hkl.to(device=self.hkl.device)
            if ref_hkl.shape != self.hkl.shape or not torch.equal(ref_hkl, self.hkl):
                raise ValueError(
                    "reference.hkl does not match self.hkl -- build the FcalcDataset "
                    "from the reference's HKL list to guarantee 1:1 sigma grafting."
                )
            sigma_I = reference.I_sigma.to(device=dev, dtype=dtype)
            if verbose:
                print(
                    f"add_noise: grafting sigmas from reference ({len(sigma_I)} "
                    f"reflections, <sigma_I>={sigma_I.mean().item():.3g})"
                )
        else:
            if sigma_abs > 0:
                if self.resolution is None:
                    raise ValueError(
                        "sigma_abs > 0 requires self.resolution (used to pick the "
                        "outer resolution shell)."
                    )
                n = len(self.resolution)
                k = max(1, int(0.1 * n))
                outer_idx = torch.argsort(self.resolution)[:k]
                sigma_abs_I = sigma_abs * intensity[outer_idx].mean()
            else:
                sigma_abs_I = torch.zeros((), dtype=dtype, device=dev)

            safe = intensity.clamp(min=0.0)
            sigma_I = torch.sqrt(
                (sigma_lin**2) * safe
                + (sigma_mul**2) * safe * safe
                + sigma_abs_I**2
            )

        I_h1 = intensity + randn1 * sigma_I
        I_h2 = intensity + randn2 * sigma_I

        diff_sum = (I_h1 - I_h2).abs().sum()
        pair_sum = (I_h1 + I_h2).sum()
        r_split = ((1.0 / (2.0**0.5)) * diff_sum / (0.5 * pair_sum)).item()

        x = I_h1 - I_h1.mean()
        y = I_h2 - I_h2.mean()
        cc = (
            (x * y).sum()
            / torch.sqrt((x * x).sum() * (y * y).sum()).clamp(min=1e-30)
        ).item()
        if verbose:
            print(f"add_noise: R-split = {r_split:.4f}, CC(half1, half2) = {cc:.4f}")

        I_mean = 0.5 * (I_h1 + I_h2)
        sigma_I_mean = sigma_I / (2.0**0.5)

        # Only the amplitude is clamped; see the note in the docstring.
        amp_noisy = torch.sqrt(I_mean.clamp(min=0.0))
        sigma_F = sigma_I_mean / (2.0 * self.fcalc_amp.clamp(min=1e-8))

        fcalc_noisy = (
            amp_noisy * torch.exp(1j * self.fcalc_phase)
        ).to(self.fcalc.dtype)

        new = FcalcDataset(
            hkl=self.hkl.clone(),
            resolution=(
                self.resolution.clone() if self.resolution is not None else None
            ),
            cell=self.cell,
            spacegroup=self.spacegroup,
            device=self.device,
        )
        new.set_fcalc(fcalc_noisy)
        new.fobs_sigma = sigma_F.to(self.device)
        new.I = I_mean.to(self.device)
        new.I_sigma = sigma_I_mean.to(self.device)
        return new

    def write_mtz(self, filepath: str) -> None:
        """
        Write Fcalc to MTZ as ``F-model`` / ``PH-model`` (phase in degrees).

        Parameters
        ----------
        filepath : str
            Output MTZ filename.

        Raises
        ------
        ValueError
            If Fcalc, HKL, cell or spacegroup is unset.
        """
        from torchref.io import mtz

        if self.fcalc is None:
            raise ValueError("No Fcalc values set. Call set_fcalc() first.")

        if self.hkl is None:
            raise ValueError("No HKL indices set.")

        if self.cell is None:
            raise ValueError("No cell set.")

        if self.spacegroup is None:
            raise ValueError("No spacegroup set.")

        # Build DataFrame
        hkl_np = self.hkl.cpu().numpy()
        df = pd.DataFrame(
            {
                "H": hkl_np[:, 0],
                "K": hkl_np[:, 1],
                "L": hkl_np[:, 2],
                "F-model": self.fcalc_amp.cpu().numpy(),
                "PH-model": torch.rad2deg(self.fcalc_phase).cpu().numpy(),
            }
        )

        mtz.write(df, self.cell.data, self.spacegroup, filepath)

    def write_mtz_as_fobs(
        self,
        filepath: str,
        sigma_frac: float = 0.05,
        f_column: str = "F-obs",
        sigf_column: str = "SIGF-obs",
        phase_column: str = "PHIF-model",
    ) -> None:
        """
        Write Fcalc to MTZ as pseudo-observations, readable back by
        :meth:`ReflectionData.load_mtz` as if measured.

        Sigmas are fabricated as ``sigma_frac * |F|``, not measured.

        Parameters
        ----------
        filepath : str
            Output MTZ filename.
        sigma_frac : float, optional
            Sigma as a fraction of |F|. Default is 0.05 (5%).
        f_column : str, optional
            Column name for amplitudes. Default is 'F-obs'.
        sigf_column : str, optional
            Column name for sigma. Default is 'SIGF-obs'.
        phase_column : str, optional
            Column name for model phases. Default is 'PHIF-model'.

        Raises
        ------
        ValueError
            If Fcalc, HKL, cell or spacegroup is unset.
        """
        from torchref.io import mtz

        if self.fcalc_amp is None:
            raise ValueError("No Fcalc values set. Call set_fcalc() first.")
        if self.hkl is None:
            raise ValueError("No HKL indices set.")
        if self.cell is None:
            raise ValueError("No cell set.")
        if self.spacegroup is None:
            raise ValueError("No spacegroup set.")

        amp = self.fcalc_amp.cpu().numpy()
        sigma = amp * sigma_frac

        hkl_np = self.hkl.cpu().numpy()
        columns = {
            "H": hkl_np[:, 0],
            "K": hkl_np[:, 1],
            "L": hkl_np[:, 2],
            f_column: amp,
            sigf_column: sigma,
        }
        if self.fcalc_phase is not None:
            columns[phase_column] = torch.rad2deg(self.fcalc_phase).cpu().numpy()

        df = pd.DataFrame(columns)
        mtz.write(df, self.cell.data, self.spacegroup, filepath)

    # ========== SERIALIZATION OVERRIDES ==========

    def _get_state(self) -> Dict[str, Any]:
        """As the base, but ``spacegroup`` is flattened via its ``hm`` symbol."""
        state = super()._get_state()
        if self.spacegroup is not None:
            state["spacegroup"] = self.spacegroup.hm
        return state

    @classmethod
    def _from_state(cls, state: Dict[str, Any], device=None) -> "FcalcDataset":
        """Rebuild from a :meth:`_get_state` dict, rewrapping the H-M string as a
        ``SpaceGroup``. Pops ``"masks"``, so ``state`` is mutated.
        """
        from torchref.utils.utils import TensorMasks

        device = normalize_device(device)

        masks_state = state.pop("masks", {})
        state = cls._drop_stale_state_keys(state)

        if "device" in state:
            state["device"] = torch.device(state["device"])

        if "spacegroup" in state and state["spacegroup"] is not None:
            if isinstance(state["spacegroup"], str):
                state["spacegroup"] = SpaceGroup(state["spacegroup"])

        if "cell" in state and state["cell"] is not None:
            if isinstance(state["cell"], torch.Tensor):
                state["cell"] = Cell(
                    state["cell"], dtype=get_float_dtype(), device=device
                )

        obj = cls(**state)

        if masks_state:
            obj.masks = TensorMasks(data=masks_state, device=device)

        return obj.to(device)

    # ========== UTILITY METHODS ==========

    def __repr__(self) -> str:
        """String representation of dataset."""
        n_refl = len(self)
        sg = self.spacegroup.name if self.spacegroup else "unknown"
        has_fcalc = "yes" if self.fcalc is not None else "no"
        return (
            f"{self.__class__.__name__}(n_reflections={n_refl}, "
            f"spacegroup='{sg}', fcalc={has_fcalc}, device={self.device})"
        )

    @property
    def spacegroup_name(self) -> Optional[str]:
        """Get space group name as string (short form, e.g., 'P212121')."""
        if self.spacegroup is None:
            return None
        return self.spacegroup.name

    @property
    def spacegroup_hm(self) -> Optional[str]:
        """Get space group Hermann-Mauguin name with spaces (e.g., 'P 21 21 21')."""
        if self.spacegroup is None:
            return None
        return self.spacegroup.hm

    @property
    def spacegroup_number(self) -> Optional[int]:
        """Get space group number (1-230)."""
        if self.spacegroup is None:
            return None
        return self.spacegroup.number
