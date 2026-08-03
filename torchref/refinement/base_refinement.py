"""
Base class for crystallographic refinement.
"""

from typing import Any, Dict, Optional

import torch
from torch.nn import Module as nnModule

from torchref.config import normalize_device
from torchref.io import ReflectionData
from torchref.model.model_ft import ModelFT
from torchref.refinement.logger import Logger
from torchref.refinement.loss_state import LossState
from torchref.refinement.targets.adp.scaler_log_scale import (
    ScalerLogScaleTrendTarget,
)
from torchref.refinement.targets.adp.scaler_u import ScalerURegularizationTarget
from torchref.refinement.targets.combined import (
    TotalADPTarget,
    TotalGeometryTarget,
)

# Target system imports
from torchref.refinement.model_error_estimation.sigma_a import SHRINK_ENABLED, SIGMA_A_MAX
from torchref.refinement.targets.xray import create_xray_target
from torchref.refinement.weighting import ManualWeighting
from torchref.scaling.scaler import Scaler
from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device


# Default LossState weights, balancing the data term against the priors with the
# x-ray data term as the reference (xray=1). Weights are hierarchical and
# MULTIPLICATIVE: a target's effective weight is the product of its path levels
# (e.g. geometry/ramachandran = weight[geometry] * weight[geometry/ramachandran];
# see LossState.get_effective_weight), so a component key scales *within* its
# group. Tuned on the AlphaFold-start benchmark (paper/figure2_alphafold_start,
# 10x10 log weight screen over geometry x adp, validation-landscape analysis):
# geometry=0.2 pulls bond/angle RMSZ close to the ideal 1.0 (vs ~1.4 at 0.1) at no
# R-free cost, and adp=0.02 keeps the main-chain B-factor distribution near ideal
# (MC-bond B RMSZ ~1.6, vs a badly under-restrained ~3.5 at adp=0.005) — also
# essentially free in R-free. (The earlier reference xray=10/geometry=1/adp=0.1
# normalizes — divide by 10 — to xray=1/geometry=0.1/adp=0.01.)
#
# geometry/ramachandran=0 DISABLES the Ramachandran restraint by default (0.2 * 0
# = 0, so aggregate() skips it): the AF-start ablation (745 paired structures)
# showed it has no measurable effect (median dR-free +0.0002, geometry RMSZ
# unchanged). Set it back to a positive value via --weights to re-enable.
DEFAULT_GROUP_WEIGHTS = {
    "xray": 1.0,
    "geometry": 0.2,
    "geometry/ramachandran": 0.0,
    "adp": 0.02,
}


class Refinement(DeviceMixin, DebugMixin, nnModule):
    """
    Refinement class to handle the overall crystallographic refinement process.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        refinement = Refinement()  # Creates empty shell with submodules
        refinement.load_state_dict(torch.load('refinement.pt'))

    2. Full initialization with file paths::

        refinement = Refinement(data_file='data.mtz', pdb='model.pdb')

    Parameters
    ----------
    data_file : str, optional
        Path to MTZ or CIF file containing reflection data.
    pdb : str, optional
        Path to PDB or CIF file containing initial model.
    cif : str, optional
        Path to CIF file for restraints.
    verbose : int, optional
        Verbosity level. Default is 1.
    max_res : float, optional
        Maximum resolution for reflections.
    device : torch.device, optional
        Computation device. Defaults to the configured default device.
    nbins : int, optional
        Number of resolution bins. Default is 10.
    column_names : dict, optional
        Mapping of logical column roles to MTZ column labels.
    wavelength : float, optional
        X-ray wavelength in Angstroms for the anomalous (f'/f'') correction.
        Default 1.0; ``0`` disables anomalous refinement entirely.
    anomalous_threshold : float, optional
        Threshold controlling anomalous data handling. Default 0.5.
    french_wilson : bool, optional
        Whether to derive amplitudes from intensities via French-Wilson.
        Default True.
    anomalous : bool, optional
        Anomalous (Bijvoet) load preference; None auto-detects. Default None.
    adp_mode : str, optional
        ADP parametrization, ``"isotropic"`` (default) or ``"anisotropic"``.
    aniso_selection : str, optional
        Phenix-style selection of atoms refined anisotropically when
        ``adp_mode="anisotropic"``.

    See the :meth:`__init__` docstring for the full parameter descriptions.

    Attributes
    ----------
    device : torch.device
        Computation device.
    verbose : int
        Verbosity level.
    reflection_data : ReflectionData
        Reflection data container.
    model : ModelFT
        Structure factor model (includes lazy restraints via model.restraints).
    scaler : Scaler
        Scale factor calculator.
    weighting : BaseWeighting
        Loss weighting scheme holding the data/prior group weights. Defaults to
        ``ManualWeighting(DEFAULT_GROUP_WEIGHTS)``; reassign to change the scheme.
    weighter : None
        Vestigial state-dict placeholder, always ``None``. The live weighting
        knob is :attr:`weighting`; ``weighter`` is retained only for state-dict
        compatibility.
    """

    def __init__(
        self,
        data_file: str = None,
        pdb: str = None,
        cif=None,
        verbose: int = 1,
        max_res: float = None,
        device: Optional[torch.device] = None,
        nbins: int = 10,
        column_names: Optional[Dict[str, str]] = None,
        wavelength: Optional[float] = 1.0,
        anomalous_threshold: float = 0.5,
        french_wilson: bool = True,
        anomalous: Optional[bool] = None,
        adp_mode: str = "isotropic",
        xray_mode: str = "ml",
        sigma_a_max: float = SIGMA_A_MAX,
        shrink: bool = SHRINK_ENABLED,
        scale_target: str = DEFAULT_SCALE_TARGET,
        aniso_selection: Optional[str] = None,
    ):
        """
        Initialize Refinement.

        If data_file and pdb are provided, fully initializes the refinement.
        If not provided (empty init), creates a shell with empty submodules
        ready for load_state_dict().

        Parameters
        ----------
        data_file : str, optional
            Path to MTZ or CIF file containing reflection data.
        pdb : str, optional
            Path to PDB or CIF file containing initial model.
        cif : str, optional
            Path to CIF file for restraints.
        verbose : int, optional
            Verbosity level. Default is 1.
        max_res : float, optional
            Maximum resolution for reflections.
        device : torch.device, optional
            Computation device. Defaults to the configured default device.
        nbins : int, optional
            Number of resolution bins. Default is 10.
        french_wilson : bool, optional
            Whether to derive amplitudes from intensities via French-Wilson.
            Default True. Set False to use existing French-Wilson-corrected
            ``F``/``SIGF`` columns directly when the MTZ also carries
            intensities.
        anomalous : bool, optional
            Anomalous (Bijvoet) load preference. If None (default), anomalous
            ``F(+)/F(-)`` (or ``I(+)/I(-)``) data are auto-detected and loaded as
            Friedel pairs when present, enabling the model's f'' term. True forces
            this; False forces a merged load (f'' disabled).
        wavelength : float, optional
            X-ray wavelength in Angstroms for the anomalous (f'/f'') scattering
            correction. Default 1.0. A value of ``0`` means "no anomalous
            refinement": it disables the correction (model wavelength ``None``)
            and forces a Friedel-merged read (overrides ``anomalous`` to False).
        adp_mode : str, optional
            ADP parametrization: ``"isotropic"`` (default) refines a per-atom
            B-factor; ``"anisotropic"`` refines a 6-component U tensor for the
            atoms selected by ``aniso_selection``. The model is converted between
            representations accordingly (see :meth:`Model.set_adp_mode`).
        aniso_selection : str, optional
            Phenix-style atom selection of atoms refined anisotropically when
            ``adp_mode="anisotropic"``. Defaults to
            ``"not resname HOH and not element H"`` (all non-water heavy atoms).
        """
        super().__init__()
        # Refinement constructs its own submodules from file paths, so
        # there is nothing to reconcile yet — ``resolve_device`` with no
        # modules just normalises ``device`` (or returns the default).
        self.device = resolve_device(device=device)
        self.verbose = verbose
        self.data_file = data_file
        self.pdb = pdb
        self.history = dict()
        self.max_res = max_res
        self.nbins = nbins
        self.lr = 1e-3
        # Wavelength drives f'/f'' anomalous scattering corrections in ModelFT.
        # Default 1.0 preserves prior behavior; set to the experimental wavelength
        # for anomalous (Bijvoet) refinement, or None to disable entirely.
        self.wavelength = wavelength
        self.anomalous_threshold = anomalous_threshold
        self.french_wilson = french_wilson
        # Anomalous (Bijvoet) load preference: None auto-detects and prefers
        # anomalous data when present; True forces it; False forces a merged load.
        self.anomalous = anomalous
        # ADP parametrization: 'isotropic' (default) refines per-atom B;
        # 'anisotropic' refines a 6-component U for atoms matched by
        # aniso_selection (default all non-water heavy atoms). Applied to the
        # model right after load, before scaling/restraints/targets.
        self.adp_mode = adp_mode
        self.aniso_selection = aniso_selection
        # Everything the x-ray targets are built from must be set BEFORE
        # _init_targets() further down this __init__ (it also calls get_scales()).
        # They are read back through _xray_target_kwargs(), which is the single
        # source of truth for target construction -- see the note there.
        self.scale_target = scale_target
        self.xray_mode = xray_mode
        self.sigma_a_max = sigma_a_max
        self.shrink = shrink
        # A wavelength of 0 means "no anomalous refinement": disable the f'/f''
        # correction (model wavelength None) and force a Friedel-merged read so
        # F(+)/F(-) are not loaded as Bijvoet pairs.
        if self.wavelength is not None and float(self.wavelength) == 0.0:
            self.wavelength = None
            self.anomalous = False

        # Persistent state and logger (created lazily)
        self._loss_state: Optional[LossState] = None
        self._logger: Optional[Logger] = None

        # Static weighting scheme holding the default group base weights. This
        # is the transparent, first-class home for the data/prior balance;
        # _create_loss_state applies it to the LossState. Reassign self.weighting
        # to a different BaseWeighting to change the scheme.
        self.weighting = ManualWeighting(DEFAULT_GROUP_WEIGHTS)

        # Empty initialization - create empty submodules for state_dict loading
        if data_file is None and pdb is None:
            # Create empty submodules so state_dict keys exist
            self.reflection_data = ReflectionData(
                verbose=self.verbose, device=self.device
            )
            self.model = ModelFT(
                verbose=self.verbose,
                device=self.device,
                wavelength=self.wavelength,
                anomalous_threshold=self.anomalous_threshold,
            )
            self.scaler = Scaler(
                verbose=self.verbose, device=self.device, nbins=self.nbins
            )
            # Restraints are now lazy-loaded via model.restraints property
            self.weighter = None
            return

        # Full initialization with file paths
        try:
            self.to(self.device)
            if isinstance(data_file, str):
                self.reflection_data = ReflectionData(
                    verbose=self.verbose, device=self.device
                )
                if data_file.endswith(".mtz"):
                    self.reflection_data.load_mtz(
                        data_file,
                        column_names=column_names,
                        french_wilson=self.french_wilson,
                        anomalous=self.anomalous,
                    )
                elif data_file.endswith(".cif"):
                    self.reflection_data.load_cif(data_file, anomalous=self.anomalous)
                else:
                    raise ValueError(
                        f"Unsupported data file format: {data_file}. Supported formats are .mtz and .cif"
                    )
            if max_res is not None:
                try:
                    max_res_val = float(max_res)
                except (TypeError, ValueError):
                    raise ValueError(f"max_res must be a float > 0, got {max_res!r}")
                if max_res_val <= 0:
                    raise ValueError(f"max_res must be > 0, got {max_res_val}")
                self.reflection_data = self.reflection_data.cut_res(max_res_val)
                self.max_res = max_res_val
            else:
                self.max_res = self.reflection_data.get_max_res()
            self.model = ModelFT(
                verbose=self.verbose,
                max_res=self.max_res,
                device=self.device,
                wavelength=self.wavelength,
                anomalous_threshold=self.anomalous_threshold,
                # Apply the f'' (Bijvoet) term only when the data were loaded as
                # explicit Friedel pairs; merged data gate it off.
                apply_bijvoet=not self.reflection_data.friedel_merged,
            )
            if pdb.endswith(".cif"):
                self.model.load_cif(pdb)
            elif pdb.endswith(".pdb"):
                self.model.load_pdb(pdb)
            else:
                raise ValueError(
                    f"Unsupported model file format: {pdb}. Supported formats are .pdb and .cif"
                )

            self._sync_model_cell_to_data()
            # Set ADP parametrization (iso/aniso) before scaling/restraints/targets
            # so all structure-factor evaluation sees the chosen representation.
            self.model.set_adp_mode(self.adp_mode, self.aniso_selection)
            self.setup_scaler()
            # Configure CIF path for lazy restraint building (restraints built on first access)
            self.model.set_restraints_cif(cif)
            self.model._build_restraints()
            self._freeze_unrestrained_residues()

            # Initialize target functions (instantiated once, evaluated each iteration)
            self._init_targets()

        except Exception as e:
            if self.verbose > 1:
                self.debug_on_error(e)
            raise e

    def _freeze_unrestrained_residues(self):
        """Freeze xyz of multi-atom, non-water residues that contain an atom with
        no geometry restraint.

        After restraints are built, an atom that appears in no bond / angle /
        torsion / plane / chiral restraint has nothing holding its position;
        refining it against the X-ray target alone distorts the geometry (the
        usual cause is a ligand whose monomer-library CIF could not be found).
        Such an atom's whole residue is frozen in xyz, *except*:

        - waters (legitimately restraint-free), and
        - single-atom residues (ions: no internal geometry to break).

        B-factors and occupancy remain refinable.
        """
        import pandas as pd

        model = self.model
        pdb = getattr(model, "pdb", None)
        acc = getattr(getattr(model, "_restraints", None), "restraints", None)
        if pdb is None or acc is None:
            return
        n = len(pdb)

        # 1. atoms that appear in at least one geometry restraint
        restrained = set()

        def mark(idx):
            if idx is None or len(idx) == 0:
                return
            for v in torch.as_tensor(idx).reshape(-1).tolist():
                if 0 <= v < n:
                    restrained.add(int(v))

        for rtype in ("bond", "angle", "torsion", "plane"):
            try:
                if rtype in acc:
                    for _origin, data in acc[rtype].items():
                        if isinstance(data, dict):
                            mark(data.get("indices"))
            except Exception:
                pass
        try:
            if "chiral" in acc:
                mark(acc["chiral"]["indices"])
        except Exception:
            pass

        # 2. group atoms into residues (positional, aligned with xyz)
        resname = pdb["resname"].astype(str).str.strip().tolist()
        icode = (pdb["icode"].astype(str).tolist() if "icode" in pdb.columns
                 else [""] * n)
        chainid = pdb["chainid"].astype(str).tolist()
        resseq = pdb["resseq"].astype(str).tolist()
        res_atoms = {}
        for i in range(n):
            res_atoms.setdefault(
                (chainid[i], resseq[i], icode[i], resname[i]), []
            ).append(i)

        # 3. residues with an unrestrained atom (skip water + single-atom residues)
        WATER = {"HOH", "WAT", "DOD", "H2O", "SOL", "TIP", "TIP3", "TIP4"}
        freeze_idx, frozen_res = [], []
        for (c, rs, ic, rn), atoms in res_atoms.items():
            if rn in WATER or len(atoms) <= 1:
                continue
            if any(i not in restrained for i in atoms):
                freeze_idx.extend(atoms)
                frozen_res.append(f"{c}/{rn}{rs}")
        if not freeze_idx:
            return

        # 4. freeze xyz of those atoms (same path as freeze_selection)
        model.xyz_mask[torch.tensor(freeze_idx, dtype=torch.long)] = False
        model.apply_mask_to_parameter("xyz")
        if self.verbose > 0:
            shown = frozen_res[:20] + (["..."] if len(frozen_res) > 20 else [])
            print(
                f"Froze xyz for {len(freeze_idx)} atom(s) in {len(frozen_res)} "
                f"residue(s) with incomplete restraints: {shown}"
            )

    def _xray_target_kwargs(self) -> dict:
        """Every configuration value the x-ray targets are built from, in one place.

        **Both** construction paths go through this, and that is the point. Until
        2026-07-31 ``set_xray_target_mode`` rebuilt the targets with only a subset of
        the configuration, and ``LBFGSRefinement.__init__`` called it *after*
        ``_init_targets`` had already built them correctly -- so the second build
        silently reverted ``sigma_a_estimator``, ``sigma_in_variance``,
        ``ml_estimation_set``, ``use_alpha`` and ``subtract_sigma_from_beta`` to
        factory defaults. Those five CLI flags were dead for every ``LBFGSRefinement``,
        which is the CLI and every paper harness. Keep this the only place the kwargs
        are spelled out, so a new option cannot be dropped by one path and honoured by
        the other.

        ``getattr`` fallbacks are required: the ensemble and ``create_from_state_dict``
        paths call target construction before these attributes exist.
        """
        return dict(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            verbose=self.verbose,
            sigma_a_max=getattr(self, "sigma_a_max", SIGMA_A_MAX),
            shrink=getattr(self, "shrink", SHRINK_ENABLED),
        )

    def _build_xray_targets(self, mode: str) -> None:
        """Build the work/test x-ray targets for ``mode`` with the full configuration."""
        kw = self._xray_target_kwargs()
        self.xray_target_work = create_xray_target(
            mode=mode, use_work_set=True, **kw
        )
        self.xray_target_test = create_xray_target(
            mode=mode, use_work_set=False, **kw
        )
        self.xray_mode = mode

    def _init_targets(self, xray_mode: str = None):
        """
        Initialize target functions.

        Parameters
        ----------
        xray_mode : str, optional
            X-ray target mode. Defaults to ``self.xray_mode`` (itself defaulting to
            ``'ml'``), so the deserialization path picks up the stored mode instead of
            silently reverting to the default.
        """
        if xray_mode is None:
            xray_mode = getattr(self, "xray_mode", "ml")
        self._build_xray_targets(xray_mode)

        # Total geometry target (handles bond, angle, torsion internally)
        # Geometry targets now accept model directly instead of refinement
        self.geometry_target = TotalGeometryTarget(self.model, verbose=self.verbose)

        self.adp_target = TotalADPTarget(self.model, verbose=self.verbose)

        # Initialize scaler scales (overall scale, anisotropic U, bulk solvent)
        # so the scaler-regularization targets have valid parameters to read.
        self.get_scales()

        if self.verbose > 0:
            print(f"Initialized targets with xray_mode='{xray_mode}'")

    def set_xray_target_mode(self, mode: str):
        """
        Change the X-ray target mode.

        Parameters
        ----------
        mode : str
            X-ray target mode: 'ml' (default), 'ml_noalpha', 'ml_full', 'nll_beta',
            'nll', 'ls', 'ls_wunit_k1'. See
            :mod:`torchref.refinement.targets.xray._specs` for the taxonomy.
        """
        self._build_xray_targets(mode)
        # Reset loss state since targets changed
        self.reset_loss_state()
        if self.verbose > 0:
            print(f"Changed X-ray target mode to '{mode}'")

    @property
    def data(self):
        """
        Expose reflection_data as 'data' for weighting module compatibility.

        Returns
        -------
        ReflectionData
            The reflection data container.
        """
        return self.reflection_data

    @property
    def loss_state(self) -> LossState:
        """
        Get or create the persistent LossState.

        The LossState is created once and reused across refinement cycles.
        Targets are registered once; weights are updated each cycle.

        Returns
        -------
        LossState
            The persistent loss state with targets registered.
        """
        if self._loss_state is None:
            self._loss_state = self._create_loss_state()
        return self._loss_state

    @property
    def logger(self) -> Logger:
        """
        Get or create the Logger for this refinement.

        Returns
        -------
        Logger
            Logger instance linked to the persistent LossState.
        """
        if self._logger is None:
            self._logger = Logger(
                state=self.loss_state,
                verbose=self.verbose,
            )
        return self._logger

    def reset_loss_state(self) -> None:
        """
        Reset the persistent LossState and Logger.

        Call this if targets need to be re-registered (e.g., after changing
        target modes or reinitializing targets).
        """
        self._loss_state = None
        self._logger = None

    def get_scales(self):
        """
        Initialize and refit the scaler against the current model.

        Calls ``scaler.initialize()``, flags outliers in the reflection data
        (``z_threshold=5.0``), refits the scaler with LBFGS, then re-flags
        outliers against the refitted scales. No-op when the scaler is ``None``
        (e.g. targets such as ``ls_wunit_k1`` in ``binwise_optimal`` mode that
        compute their own scale and intentionally leave ``self.scaler`` unset).

        The scale-fit objective is chosen by ``self.scale_target`` (see
        :meth:`torchref.scaling.scaler_base.ScalerBase.refine_lbfgs`). It is
        deliberately *not* the body target: for ``ml_full`` that would put a
        32-node quadrature inside every L-BFGS line-search evaluation, and scaling
        is a nuisance-magnitude fit that need not carry a model-error term at all.
        """
        if not hasattr(self, "scaler") or self.scaler is None:
            # Targets that compute their own scale (e.g. ls_wunit_k1 in
            # binwise_optimal mode) intentionally leave ref.scaler=None.
            # Nothing to initialize or refit here.
            return
        self.scaler.initialize()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)
        self.scaler.refine_lbfgs(scale_target=getattr(self, "scale_target", DEFAULT_SCALE_TARGET))
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)

    def setup_scaler(self):
        """
        Construct the scaler and assign it to ``self.scaler``.

        Uses the class named by ``self._scaler_class`` if set, otherwise the
        default :class:`Scaler`, and wires it to the current model, reflection
        data, ``nbins``, verbosity, and device.
        """
        cls = getattr(self, "_scaler_class", None) or Scaler
        self.scaler = cls(
            self.model,
            self.reflection_data,
            nbins=self.nbins,
            verbose=self.verbose,
            device=self.device,
        )

    def _sync_model_cell_to_data(
        self,
        axis_rtol: float = 0.01,
        angle_atol_deg: float = 1.0,
    ) -> None:
        """Always cast the reflection-data cell onto the model.

        Mirrors cctbx's ``xrs.customized_copy(crystal_symmetry=data_sym)``:
        the data cell is authoritative because HKL indices are defined
        relative to it. Atoms keep their Cartesian coordinates and are
        reinterpreted in the new cell — fractional coordinates implicitly
        shift. Without this sync, ``F_calc`` is computed with the wrong
        basis at HKLs that mean something else, producing a pose-dependent
        phase bias that distorts the gradient (9RTS case: a 2.4% b-axis
        mismatch was pulling rigid-body LBFGS into a worse local minimum).

        The sync runs unconditionally. A warning is emitted only when the
        disagreement is large enough to flag for the user:

        - any axis differs by ≥ ``axis_rtol`` relative (default 1%)
        - any angle differs by ≥ ``angle_atol_deg`` absolute (default 1°)
        """
        if self.model is None or self.model.cell is None:
            return
        if self.reflection_data is None or self.reflection_data.cell is None:
            return
        m_t = self.model.cell.data
        d_t = self.reflection_data.cell.data
        m = m_t.detach().cpu().tolist()
        d = d_t.detach().cpu().tolist()
        axes_m, angles_m = m[:3], m[3:]
        axes_d, angles_d = d[:3], d[3:]
        axis_off = any(
            abs(am - ad) / max(abs(ad), 1e-12) >= axis_rtol
            for am, ad in zip(axes_m, axes_d)
        )
        angle_off = any(
            abs(gm - gd) >= angle_atol_deg
            for gm, gd in zip(angles_m, angles_d)
        )
        if axis_off or angle_off:
            import warnings
            warnings.warn(
                "PDB CRYST1 cell disagrees with reflection-data cell beyond "
                f"tolerance (axes ≥ {axis_rtol * 100:g}% or angles ≥ "
                f"{angle_atol_deg:g}°). Casting the data cell onto the model "
                "(HKL indices are defined relative to it). Atoms keep "
                "Cartesian coordinates; fractional coordinates implicitly "
                "shift. Mirrors cctbx's "
                "`xrs.customized_copy(crystal_symmetry=data_sym)`.\n"
                f"  PDB  cell: {m}\n"
                f"  data cell: {d}",
                stacklevel=2,
            )
        self.model.cell = self.reflection_data.cell
        self.model.reset_cache()

    def parameters(self, recurse: bool = True):
        """
        Return unique parameters from this module and all submodules.

        Uses the default Module.parameters() to gather parameters, then removes
        duplicates while preserving order to avoid passing the same tensor
        multiple times to the optimizer.

        Parameters
        ----------
        recurse : bool, optional
            If True, yields parameters of this module and all submodules.
            Default is True.

        Returns
        -------
        list
            List of unique parameter tensors.
        """
        params = list[Any](super().parameters(recurse))
        seen = set()
        unique_params = []
        for p in params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                unique_params.append(p)
        return unique_params

    def get_fcalc(self, hkl=None, recalc=False):
        """
        Compute complex structure factors ``F_calc`` from the model.

        Parameters
        ----------
        hkl : array_like, optional
            Reflection indices. If None, uses ``reflection_data.hkl_for_sf()``
            (signed HKL, so Bijvoet mates at +h/-h get distinct ``|F_calc|``
            under anomalous scattering; falls back to canonical hkl otherwise).
        recalc : bool, optional
            Force recomputation rather than reusing the cached SF. Default False.

        Returns
        -------
        torch.Tensor
            Complex ``F_calc`` per reflection.
        """
        if hkl is None:
            # Signed HKL so Bijvoet mates (which share a canonical ASU index)
            # are evaluated at +h/-h and get distinct |F_calc| under anomalous
            # scattering. Falls back to canonical hkl when unavailable.
            hkl = self.reflection_data.hkl_for_sf()
        return self.model(hkl, recalc=recalc)

    def get_fcalc_scaled(self, hkl=None, recalc=False):
        """
        Compute scaled complex structure factors (``scaler(F_calc)``).

        See :meth:`get_fcalc` for the ``hkl`` and ``recalc`` parameters.

        Returns
        -------
        torch.Tensor
            Complex scaled ``F_calc`` per reflection.
        """
        fcalc = self.get_fcalc(hkl, recalc=recalc)
        fcalc_scaled = self.scaler(fcalc)
        return fcalc_scaled

    def adp_loss(self):
        """
        Compute total ADP loss using TotalADPTarget.

        This combines the components registered by ``TotalADPTarget``:

        - ``simu``: bond-based B-factor similarity (SIMU-like)
        - ``locality``: local-neighbourhood B-factor smoothness restraint
        - ``KL``: KL-divergence spread control of the B-factor distribution

        Returns
        -------
        torch.Tensor
            Total ADP loss value.
        """
        return self.adp_target()

    def get_F_calc(self, hkl=None, recalc=False):
        """
        Compute structure-factor amplitudes ``|F_calc|`` from the model.

        See :meth:`get_fcalc` for the ``hkl`` and ``recalc`` parameters.

        Returns
        -------
        torch.Tensor
            Real ``|F_calc|`` per reflection.
        """
        return torch.abs(self.get_fcalc(hkl, recalc=recalc))

    def get_F_calc_scaled(self, hkl=None, recalc=False):
        """
        Compute scaled structure-factor amplitudes ``|scaler(F_calc)|``.

        See :meth:`get_fcalc` for the ``hkl`` and ``recalc`` parameters.

        Returns
        -------
        torch.Tensor
            Real scaled ``|F_calc|`` per reflection.
        """
        return torch.abs(self.get_fcalc_scaled(hkl, recalc=recalc))

    def nll_xray(self):
        """
        Compute X-ray negative log-likelihood for work and test sets.

        Returns
        -------
        tuple of torch.Tensor
            Tuple of (work_nll, test_nll) tensors.
        """
        return self.xray_target_work(), self.xray_target_test()

    def xray_loss_work(self) -> torch.Tensor:
        """
        Compute X-ray loss on work set using instantiated target.

        Returns
        -------
        torch.Tensor
            X-ray loss on work set.
        """
        return self.xray_target_work()

    def xray_loss_test(self) -> torch.Tensor:
        """
        Compute X-ray loss on test set using instantiated target.

        Returns
        -------
        torch.Tensor
            X-ray loss on test set.
        """
        return self.xray_target_test()

    def bond_loss(self) -> torch.Tensor:
        """
        Compute bond length NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Bond length NLL loss.

        """
        return self.geometry_target.target_losses()["bond_target"]

    def angle_loss(self) -> torch.Tensor:
        """
        Compute angle NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Angle NLL loss.
        """
        return self.geometry_target.target_losses()["angle_target"]

    def torsion_loss(self) -> torch.Tensor:
        """
        Compute torsion angle NLL via geometry_target.

        Returns
        -------
        torch.Tensor
            Torsion angle NLL loss.
        """
        return self.geometry_target.target_losses()["torsion_target"]

    def geometry_loss(self) -> torch.Tensor:
        """
        Compute total geometry NLL using TotalGeometryTarget.

        Returns
        -------
        torch.Tensor
            Total geometry NLL loss.
        """
        return self.geometry_target()

    def _create_loss_state(self) -> LossState:
        """
        Create a configured LossState for optimization (internal).

        Sets up a LossState with all targets registered as callables with
        hierarchical naming (e.g., 'geometry/bond', 'adp/simu'), then applies the
        default group base weights ``DEFAULT_GROUP_WEIGHTS`` (xray 1 / geometry
        0.2 / adp 0.02) — the single, transparent source of truth for the
        data/prior balance (see that constant for the calibration rationale).

        Returns
        -------
        LossState
            Configured LossState with targets registered and default group
            weights applied.
        """
        state = LossState(device=self.device)

        # Register X-ray target
        state.register_target("xray", self.xray_target_work)

        # Register geometry targets
        state.register_targets(self.geometry_target)

        # Register ADP targets
        state.register_targets(self.adp_target)
        # Scaler regularization targets require the scaler to actually have
        # the regularized parameters. Solvent-only scalers (rigid-body
        # ls_wunit_k1) delete U and freeze log_scale.
        if self.scaler is not None:
            n_ref = int(self.reflection_data.hkl.shape[0])
            if hasattr(self.scaler, "U"):
                state.register_target(
                    "adp/scaler_U",
                    ScalerURegularizationTarget(self.scaler, n_reflections=n_ref),
                )
            if (hasattr(self.scaler, "log_scale")
                    and self.scaler.log_scale.requires_grad):
                state.register_target(
                    "adp/scaler_log_scale",
                    ScalerLogScaleTrendTarget(self.scaler, n_reflections=n_ref),
                )

        # Apply the weighting scheme (default: ManualWeighting holding the
        # DEFAULT_GROUP_WEIGHTS — xray 1 / geometry 0.2 / adp 0.02). The scheme
        # returns a {component: weight} dict; we apply it to the state. This is
        # the single, transparent place the data/prior balance lives — see
        # DEFAULT_GROUP_WEIGHTS for the calibration-offset rationale.
        state.set_weights(self.weighting(state))

        return state

    def create_loss_state(self) -> LossState:
        """
        Create a configured LossState for optimization.

        Prefer the :attr:`loss_state` property for the persistent, reused
        state. This method is retained for backwards compatibility and simply
        delegates to :meth:`_create_loss_state` (it does not emit a runtime
        ``DeprecationWarning``).

        Returns
        -------
        LossState
            Configured LossState with targets registered and the default group
            weights applied.
        """
        return self._create_loss_state()

    def complete_loss_state(self) -> "LossState":
        """
        Update and return the persistent LossState.

        Updates the persistent LossState with current meta, target info,
        cached losses, and weights. The state is reused across cycles.

        The cached active-parameter leaf set is *not* refreshed here. Stale
        leaves are not a correctness hazard: a leaf that's in the set but
        whose Parameter object was replaced externally (e.g. by
        ``Model.freeze``) just gets ignored by ``_freeze_graph_extras``,
        which costs a marginal amount of wasted backward work but never
        produces wrong answers. If you do call ``Model.freeze`` /
        ``Model.unfreeze`` between LossState creation and a refinement
        step, call ``state.refresh_loss_leaves()`` explicitly.

        Returns
        -------
        LossState
            Complete LossState with targets, meta, losses, and weights.
        """
        state = self.loss_state
        state.cache_losses()
        return state

    def xray_loss(self):
        """
        Compute X-ray loss on work set.

        Returns
        -------
        torch.Tensor
            X-ray loss on work set.
        """
        return self.xray_loss_work()

    def restraints_loss(self):
        """
        Compute total geometry restraints loss.

        Returns
        -------
        torch.Tensor
            Total geometry restraints loss.
        """
        return self.geometry_loss()

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect refinement metrics (R-factors, geometry, ADP) for logging.

        This is the standard method for gathering refinement metrics for logging.
        Returns full unfiltered stats - filtering is done at display time.

        Returns
        -------
        dict
            Dictionary with all metrics (unfiltered, with StatEntry objects).
        """
        metrics = {}

        with torch.no_grad():
            # R-factors (always essential)
            rwork, rfree = self.get_rfactor()
            metrics["rwork"] = (
                rwork
                if isinstance(rwork, float)
                else rwork.item() if hasattr(rwork, "item") else float(rwork)
            )
            metrics["rfree"] = (
                rfree
                if isinstance(rfree, float)
                else rfree.item() if hasattr(rfree, "item") else float(rfree)
            )
            metrics["rfree_gap"] = metrics["rfree"] - metrics["rwork"]

            if hasattr(self, "geometry_target"):
                metrics["geometry"] = self.geometry_target.stats()
            if hasattr(self, "adp_target"):
                metrics["adp"] = self.adp_target.stats()

        return metrics

    def add_target_info_to_state(self, state: "LossState") -> "LossState":
        """
        Add target information from geometry and ADP targets to LossState.meta.

        .. deprecated::
            This method is no longer needed. Use :meth:`complete_loss_state`
            instead, which handles all state setup in one call.

        Parameters
        ----------
        state : LossState
            Current loss state. Meta will be updated with target info.
        Returns
        -------
        LossState
            Updated loss state (unchanged).
        """
        import warnings

        warnings.warn(
            "add_target_info_to_state is deprecated and is a no-op. "
            "Use complete_loss_state() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return state

    def get_rfactor(self):
        """``(R_work, R_free)`` for the current model.

        Delegates to the work X-ray target, the single source of truth: R is
        computed from exactly the scaled ``|F_calc|`` the target's loss sees
        (the scaler's scaling, or the target's own closed-form per-bin scale for
        ``binwise_optimal``). See :meth:`XrayTarget.get_rfactor`.
        """
        return self.xray_target_work.get_rfactor()

    def plot_fcalc_vs_fobs(self, outpath="fcalc_vs_fobs.png"):
        """
        Scatter-plot calculated vs observed amplitudes and save to ``outpath``.

        Parameters
        ----------
        outpath : str, optional
            Output PNG path. Default ``"fcalc_vs_fobs.png"``.
        """
        import matplotlib.pyplot as plt

        with torch.no_grad():
            F_obs = self.reflection_data.get_corrected_data()[0]
            self.rfree_flags = self.reflection_data.rfree_flags
            F_calc = self.get_F_calc()
            F_obs_amp = torch.abs(F_obs).cpu().numpy()
            F_calc_amp = torch.abs(F_calc).cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.scatter(F_obs_amp, F_calc_amp, alpha=0.5)
            plt.plot(
                [0, max(F_obs_amp)], [0, max(F_obs_amp)], color="red", linestyle="--"
            )
            plt.xlabel("Observed |F|")
            plt.ylabel("Calculated |F|")
            plt.title("F_calc vs F_obs")
            plt.grid()
            plt.savefig(outpath)

    def write_out_mtz(self, out_mtz_path="refined_output.mtz", anomalous=None):
        """Write refined map coefficients to an MTZ file.

        Parameters
        ----------
        out_mtz_path : str
            Output MTZ path.
        anomalous : bool, optional
            If True, emit a phenix-style anomalous MTZ: display maps and merged
            columns in the canonical ASU (Friedel mates merged by mean
            amplitude) plus unstacked ``F-obs(+/-)`` / ``F-model(+/-)`` columns
            on the same ASU index. If False, the legacy per-row layout is
            written. If None (default), chosen automatically from the data:
            anomalous output when the data were loaded as Bijvoet pairs
            (``reflection_data.friedel_merged`` is False), legacy otherwise.
        """
        with torch.no_grad():
            # Signed HKL so the per-row fcalc carries the anomalous (Bijvoet)
            # difference; write_mtz consumes it row-aligned with reflection_data.
            hkl = self.reflection_data.hkl_for_sf()
            fcalc = self.scaler(self.get_fcalc(hkl), use_mask=False)
            self.reflection_data.write_mtz(out_mtz_path, fcalc, anomalous=anomalous)

    def collect_deposition_metadata(self, metadata=None):
        """Collect refinement statistics into a RefinementMetadata object.

        Reuses existing statistics from ``collect_metrics()``,
        ``get_rfactor()``, and reflection data attributes.

        Parameters
        ----------
        metadata : RefinementMetadata, optional
            Existing metadata to merge with (e.g. from input file pass-through).
            Refinement statistics take precedence over pass-through values.

        Returns
        -------
        RefinementMetadata
            Metadata populated with final refinement statistics.
        """
        from torchref.io.metadata import RefinementMetadata

        refinement_meta = RefinementMetadata.from_refinement(self)

        # Merge with input file metadata if available
        if metadata is not None:
            return metadata.merge(refinement_meta)

        # Merge with pass-through headers from input file
        if hasattr(self.model, "_input_file") and self.model._input_file:
            input_file = self.model._input_file
            if input_file.endswith(".pdb"):
                input_meta = RefinementMetadata.from_pdb_file(input_file)
            elif input_file.endswith((".cif", ".mmcif")):
                input_meta = RefinementMetadata.from_cif_file(input_file)
            else:
                input_meta = None
            if input_meta is not None:
                return input_meta.merge(refinement_meta)

        return refinement_meta

    def write_out_pdb(self, out_pdb_path="refined_output.pdb", metadata=None):
        """Write refined PDB with optional metadata header.

        Parameters
        ----------
        out_pdb_path : str
            Output PDB file path.
        metadata : RefinementMetadata, optional
            Metadata for PDB header. If None, auto-collected from refinement.
        """
        if metadata is None:
            metadata = self.collect_deposition_metadata()
        self.model.write_pdb(out_pdb_path, metadata=metadata)

    def write_out_cif(self, out_cif_path="refined_output.cif", metadata=None):
        """Write refined coordinates as mmCIF with metadata.

        Parameters
        ----------
        out_cif_path : str
            Output mmCIF file path.
        metadata : RefinementMetadata, optional
            Metadata for mmCIF categories. If None, auto-collected from refinement.
        """
        if metadata is None:
            metadata = self.collect_deposition_metadata()
        self.model.write_cif(out_cif_path, metadata=metadata)

    def save_state(self, path: str):
        """
        Save the complete state of the refinement to a file.

        Parameters
        ----------
        path : str
            Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved refinement state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the refinement from a file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, optional
            Whether to strictly enforce that keys match. Default is True.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded refinement state from {path}")

    @classmethod
    def create_from_state_dict(
        cls,
        state_dict: dict,
        device: torch.device = None,
        verbose: int = 1,
    ) -> "Refinement":
        """
        Create a fully initialized Refinement from a state dictionary.

        This is the recommended way to restore a Refinement from a saved state.
        It rebuilds the core submodules (reflection data, model, scaler) using
        their respective factory methods, then calls PyTorch's default
        load_state_dict. (Restraints are normally lazy-loaded via
        ``model.restraints``; the standalone restraints handling here is a
        legacy state-dict path and does not make restraints a first-class
        persisted submodule.)

        Parameters
        ----------
        state_dict : dict
            State dictionary from torch.save(refinement.state_dict(), ...)
            or from loading a checkpoint file.
        device : torch.device, optional
            Device to place tensors on. Defaults to the configured default
            device.
        verbose : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        Refinement
            Fully initialized instance with restored state.

        Examples
        --------
        Save and load refinement state::

            # Save
            torch.save(refinement.state_dict(), 'refinement.pt')

            # Load
            state = torch.load('refinement.pt')
            refinement = Refinement.create_from_state_dict(state)

            # Continue refinement
            rwork, rfree = refinement.get_rfactor()
            print(f"Restored at R-work={rwork:.4f}, R-free={rfree:.4f}")
        """

        device = normalize_device(device)

        # Helper to extract submodule state from flattened state_dict
        def extract_submodule_state(state_dict: dict, prefix: str) -> dict:
            """Extract keys starting with prefix and strip the prefix."""
            result = {}
            prefix_with_dot = prefix + "."
            for key, value in state_dict.items():
                if key.startswith(prefix_with_dot):
                    result[key[len(prefix_with_dot) :]] = value
            return result

        # Extract submodule states from flattened keys
        model_state = extract_submodule_state(state_dict, "model")
        reflection_data_state = extract_submodule_state(state_dict, "reflection_data")
        scaler_state = extract_submodule_state(state_dict, "scaler")
        restraints_state = extract_submodule_state(state_dict, "restraints")
        weighter_state = extract_submodule_state(state_dict, "weighter")

        if verbose > 0:
            print(
                f"Extracted state dict sizes: model={len(model_state)}, data={len(reflection_data_state)}, "
                f"scaler={len(scaler_state)}, restraints={len(restraints_state)}"
            )

        # Create submodules using their factory methods
        # These properly set up structure before loading values
        # ReflectionData is now a dataclass with _from_state() method
        reflection_data = ReflectionData._from_state(
            reflection_data_state, device=str(device)
        )

        model = ModelFT.create_from_state_dict(
            model_state, device=device, verbose=verbose
        )

        # Create Scaler with model and data (required for proper setup)
        scaler = Scaler(model, reflection_data, verbose=verbose, device=device)

        # Create Restraints with model (required for proper setup)
        from torchref.restraints import Restraints

        restraints = Restraints(model, verbose=verbose)

        # Create empty instance
        instance = cls.__new__(cls)
        nnModule.__init__(instance)

        # Set basic attributes
        instance.device = device
        instance.verbose = verbose
        instance.data_file = None
        instance.pdb = None
        instance.history = {}
        instance.max_res = model_state.get("_metadata_max_res", None)
        instance.nbins = 10
        instance.lr = 1e-3
        instance.effective_weights = {}

        # Register the properly created submodules
        instance.reflection_data = reflection_data
        instance.model = model
        instance.scaler = scaler
        instance.restraints = restraints
        instance.weighter = None

        # Now load the state dict - PyTorch's default will fill in values
        # Use strict=False since we may have metadata keys and properly created submodules
        instance.load_state_dict(state_dict, strict=False)

        # Reconnect model and data to scaler after loading
        instance.scaler.set_model_and_data(instance.model, instance.reflection_data)

        # Initialize targets if model is available
        if instance.model is not None and instance.model.initialized:
            try:
                instance._init_targets()
            except Exception as e:
                if verbose > 0:
                    print(f"Note: Could not initialize targets: {e}")

        if verbose > 0:
            n_atoms = len(instance.model.pdb) if instance.model.pdb is not None else 0
            n_refl = (
                instance.reflection_data.hkl.shape[0]
                if instance.reflection_data.hkl is not None
                else 0
            )
            print(
                f"Created Refinement from state_dict: {n_atoms} atoms, {n_refl} reflections"
            )

        return instance
