"""
Base class for crystallographic refinement.
"""

from typing import Any, Dict, Optional

import math
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
# group. Calibrated on the AlphaFold-start benchmark against geometry/ADP RMSZ.
#
# geometry/ramachandran=0 DISABLES the Ramachandran restraint by default (0.2 * 0
# = 0, so aggregate() skips it). Set it back to a positive value via --weights to
# re-enable.
DEFAULT_GROUP_WEIGHTS = {
    "xray": 1.0,
    "geometry": 0.2,
    "geometry/ramachandran": 0.0,
    "adp": 0.02,
    # Sub-weight on the SIGD distribution prior. Weights multiply down the path
    # (see LossState.get_effective_weight), so this scales adp/sigd alone: it is a
    # per-atom sum, whereas adp/simu and adp/locality already sum over pairs and
    # neighbours, and the log-normal KL term it replaced was a single intensive
    # scalar. Pending the R_free weight scan, 1.0 leaves it at the group weight.
    "adp/sigd": 1.0,
    # Load balancing for the node-field ADP representation. Sub-weight on the adp
    # group, and inert on the per-atom path, so it only acts in field mode. Set
    # above the group weight because it is a barrier against a degenerate direction
    # rather than a prior competing with the data.
    "adp/node_load": 10.0,
    # Magnitude prior on the node values. Off pending its own measurement: the load
    # barrier acts only on the weights, so this is what actually bounds an extreme
    # node B, but it has not been screened yet. Same convention as
    # 'geometry/ramachandran'.
    "adp/node_smoothness": 0.0,
}

#: Weight overrides a node-field ADP representation needs, applied by
#: :meth:`BaseRefinement.set_adp_representation`.
#:
#: Work reflections per ADP parameter that :meth:`set_adp_representation` targets when
#: sizing a field. PDB-REDO holds ~7 across its whole resolution range and switches model
#: form to stay there; measured on 179 of their entries, 7 is also where a node field
#: peaks, and both directions from it are worse.
DEFAULT_REFLECTIONS_PER_ADP_PARAMETER = 7.0


class Refinement(DeviceMixin, DebugMixin, nnModule):
    """
    Refinement class to handle the overall crystallographic refinement process.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        refinement = Refinement()  # Creates empty shell with submodules
        refinement.load_state_dict(torch.load('refinement.pt'))

    2. Full initialization with file paths::

        refinement = Refinement(data_file='data.mtz', pdb='model.pdb')

    Constructor parameters are documented on :meth:`__init__`.

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
        Vestigial state-dict placeholder, always ``None``; the live weighting knob
        is :attr:`weighting`.
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
        n_iso_coeff: int = 6,
        column_names: Optional[Dict[str, str]] = None,
        wavelength: Optional[float] = 1.0,
        anomalous_threshold: float = 0.5,
        french_wilson: bool = True,
        anomalous: Optional[bool] = None,
        adp_mode: str = "isotropic",
        adp_mode_set: str = None,
        n_nodes: int = None,
        reflections_per_adp_parameter: float = DEFAULT_REFLECTIONS_PER_ADP_PARAMETER,
        xray_mode: str = "ml",
        sigma_a_max: float = SIGMA_A_MAX,
        shrink: bool = SHRINK_ENABLED,
        scale_target: str = DEFAULT_SCALE_TARGET,
        aniso_selection: Optional[str] = None,
    ):
        """Initialize Refinement, fully if ``data_file`` and ``pdb`` are given.

        Without them this is an empty init: a shell with empty submodules, ready
        for :meth:`load_state_dict`.

        Parameters
        ----------
        data_file : str, optional
            Path to the MTZ or CIF file holding reflection data.
        pdb : str, optional
            Path to the PDB or CIF file holding the initial model.
        cif : str, optional
            Path to a CIF file of restraints (monomer library).
        verbose : int, optional
            Verbosity level. Default 1.
        max_res : float, optional
            High-resolution cutoff; defaults to the data's own limit.
        device : torch.device, optional
            Computation device. Defaults to the configured default device.
        nbins : int, optional
            Number of resolution bins used to seed the scaler's scale. Default 10.
        n_iso_coeff : int, optional
            Number of Chebyshev terms in the scaler's isotropic scale. Default 6.
        column_names : dict, optional
            Mapping of logical column roles to MTZ column labels.
        wavelength : float, optional
            X-ray wavelength in Angstroms for the anomalous (f'/f'') correction.
            ``0`` means "no anomalous refinement": it disables the correction and
            forces a Friedel-merged read, **overriding** ``anomalous`` to False.
        anomalous_threshold : float, optional
            Threshold controlling anomalous data handling. Default 0.5.
        french_wilson : bool, optional
            Derive amplitudes from intensities via French-Wilson. Set False to use
            existing ``F``/``SIGF`` columns when the MTZ also carries intensities.
        anomalous : bool, optional
            Anomalous (Bijvoet) load preference. None auto-detects ``F(+)/F(-)``
            (or ``I(+)/I(-)``) and loads Friedel pairs when present, enabling the
            model's f'' term; True forces it, False forces a merged load.
        adp_mode : str, optional
            ADP parametrization: ``"isotropic"`` (default) refines a per-atom
            B-factor, ``"anisotropic"`` a 6-component U tensor for the atoms
            selected by ``aniso_selection`` (see :meth:`Model.set_adp_mode`).
            ``"field"`` / ``"field_aniso"`` replace it with a node field, sized and
            reweighted by :meth:`set_adp_representation`; ``"preserve"`` leaves the
            file's own ADPs untouched.
        adp_mode_set : str, optional
            Displacement-mode set for ``adp_mode="field_aniso"`` --- ``"rigid"`` is TLS,
            ``"rigid_dilation"`` adds uniform breathing. See
            :data:`~torchref.model.disorder_field.MODE_SETS`.
        n_nodes : int, optional
            Explicit node count for a field mode. Default None sizes it from the data
            through ``reflections_per_adp_parameter``.
        reflections_per_adp_parameter : float, optional
            Work reflections per ADP parameter a field is sized to hold. Default 7,
            which is where a node field peaks and what PDB-REDO holds.
        xray_mode : str, optional
            X-ray target taxonomy row; see :meth:`set_xray_target_mode`.
        sigma_a_max, shrink : optional
            ``sigma_A`` estimator knobs; see
            :mod:`torchref.refinement.model_error_estimation.sigma_a`.
        scale_target : str, optional
            Objective the scale fit minimises; see
            :meth:`torchref.scaling.scaler_base.ScalerBase.refine_lbfgs`.
        aniso_selection : str, optional
            Phenix-style selection of atoms refined anisotropically when
            ``adp_mode="anisotropic"``. Defaults to all non-water heavy atoms.
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
        self.n_iso_coeff = n_iso_coeff
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
        # Node-field settings. adp_mode_set names the displacement-mode set; n_nodes
        # None means "size it from the data", which is what set_adp_representation does.
        self.adp_mode_set = adp_mode_set
        self.n_nodes = n_nodes
        self.reflections_per_adp_parameter = reflections_per_adp_parameter
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
                verbose=self.verbose, device=self.device, nbins=self.nbins,
                n_iso_coeff=self.n_iso_coeff,
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
            # Set the ADP parametrization before scaling/restraints/targets so all
            # structure-factor evaluation sees the chosen representation. Routed through
            # set_adp_representation rather than straight to the model: a field mode has
            # to be sized from the reflection count and reweighted, and the model can do
            # neither. Targets do not exist yet, so it will not try to rebuild them.
            self.set_adp_representation(
                self.adp_mode,
                mode_set=self.adp_mode_set,
                n_nodes=self.n_nodes,
                reflections_per_parameter=self.reflections_per_adp_parameter,
            )
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
        """Freeze xyz of multi-atom, non-water residues holding an unrestrained atom.

        An atom in no bond/angle/torsion/plane/chiral restraint has nothing holding
        its position (usually a ligand whose monomer CIF was not found), so refining
        it against X-ray alone distorts the geometry. Waters and single-atom residues
        are exempt; B-factors and occupancy stay refinable. Must run after restraints
        are built.
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
        model.xyz_mask[torch.tensor(freeze_idx, dtype=torch.long)] = False  # dtype-ok: freeze index used to index xyz_mask; PyTorch requires int64
        model.apply_mask_to_parameter("xyz")
        if self.verbose > 0:
            shown = frozen_res[:20] + (["..."] if len(frozen_res) > 20 else [])
            print(
                f"Froze xyz for {len(freeze_idx)} atom(s) in {len(frozen_res)} "
                f"residue(s) with incomplete restraints: {shown}"
            )

    def _xray_target_kwargs(self) -> dict:
        """Every configuration value the x-ray targets are built from, in one place.

        **Keep this the only place the kwargs are spelled out.** Both construction
        paths go through it; a second build site silently reverts whatever it forgets
        to pass, which once made five CLI flags no-ops. The ``getattr`` fallbacks are
        required: the ensemble and ``create_from_state_dict`` paths build targets
        before these attributes exist.
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

    # ------------------------------------------------------------------
    # ADP representation.
    # ------------------------------------------------------------------

    FIELD_MODES = ("field", "field_aniso")

    def _field_parameters_per_node(self, mode, mode_set, refine_node_positions):
        """Storage columns one node costs: payload + log sigma + optional offset.

        Read off the payload rather than tabulated, so a new payload cannot silently
        desynchronise the budget arithmetic from what the field actually allocates.
        """
        from torchref.model.disorder_field import (
            AnisotropicPayload,
            IsotropicPayload,
            ModeCovariancePayload,
        )

        if mode_set is not None:
            payload = ModeCovariancePayload(mode_set)
        elif mode == "field_aniso":
            payload = AnisotropicPayload()
        else:
            payload = IsotropicPayload()
        return payload.width + 1 + (3 if refine_node_positions else 0)

    def nodes_for_reflection_budget(
        self,
        mode: str = "field_aniso",
        mode_set: str = None,
        reflections_per_parameter: float = DEFAULT_REFLECTIONS_PER_ADP_PARAMETER,
        refine_node_positions: bool = True,
    ) -> int:
        """Node count giving ``reflections_per_parameter`` work reflections per ADP parameter.

        The reason this lives on the refinement and not on :class:`Model`: the model has
        no idea how much data there is, and node count is set by the data rather than by
        the structure. Measured on 179 PDB-REDO entries, node count correlates with
        reflection count far more strongly than with atom count, and the model's own
        default (one node per 25 atoms) is unrelated to either.

        The work set is the denominator because it is what the refinement fits, and it is
        what PDB-REDO's ``NREFCNT`` counts, so the ratio is comparable to theirs.

        Returns
        -------
        int
            At least 2 --- a single node has no spatial structure to express.
        """
        per_node = self._field_parameters_per_node(
            mode, mode_set, refine_node_positions
        )
        n_work = int(self.data.work.n)
        budget = n_work / float(reflections_per_parameter)
        return max(2, int(round(budget / per_node)))

    def flatten_adp_field(self) -> bool:
        """Discard the field's spatial structure, keeping its level. Returns whether it ran.

        A node field fits its structure once, at the moment it is installed, and then only
        refines from there. Early in a refinement that structure is derived against
        coordinates that are still wrong, and nothing later re-derives it -- the same shape
        of mistake as fitting bulk solvent to the starting model and never revisiting it,
        which cost 11.5% error by cycle 4. Calling this between macro cycles throws away
        the accumulated structure so the data rebuilds it against the coordinates as they
        now are.

        The level is preserved: only the spatial variation is reset. Deliberately a hard
        reset rather than a pull toward flat, because a soft version is another weight to
        tune and the point is to test whether re-deriving helps at all.

        No-op when the model is not in field mode, so a driver can call it unconditionally.
        """
        field = self.model.adp_field
        if field is None:
            return False
        with torch.no_grad():
            per_atom = field().detach()
            if per_atom.ndim == 2:
                # A U6 field. Flatten through the equivalent isotropic B, NOT by taking a
                # median over all six components: setting the off-diagonals to the same
                # value as the diagonals gives a matrix with eigenvalues (3L, 0, 0), which
                # is singular, and the Cholesky encode of it is NaN. refit lifts a 1-D B
                # target to U_iso * I, which is the flat U that is actually meant.
                b = (8.0 * math.pi**2 / 3.0) * per_atom[:, :3].sum(dim=1)
            else:
                b = per_atom
            finite = torch.isfinite(b)
            if not bool(finite.any()):
                return False
            level = b[finite].median()
            target = torch.where(finite, level.expand_as(b), b)
        # refit replaces refinable_params, so any cached leaf set or optimizer state
        # referring to the old tensor is stale.
        field.refit(target)
        self.reset_loss_state()
        if self.verbose > 0:
            print(f"Flattened the ADP field to a level of {float(level):.2f}")
        return True

    def set_adp_representation(
        self,
        mode: str,
        mode_set: str = None,
        n_nodes: int = None,
        reflections_per_parameter: float = DEFAULT_REFLECTIONS_PER_ADP_PARAMETER,
        k_neighbors: int = 12,
        refine_node_positions: bool = True,
        aniso_selection: str = None,
    ):
        """Switch the ADP parametrization, sizing and reweighting it for this data set.

        :meth:`Model.set_adp_mode` changes the representation but cannot size it: node
        count follows from the reflection count, and the model has no idea how much data
        there is. It also cannot swap the ADP restraint set, which is a property of the
        representation rather than a weight to tune.

        The loss is **not** rebalanced for a field. The point of the representation is that
        smoothness comes from the parametrisation, so a field should need *less*
        regularisation than a per-atom model, not a reweighted version of the same
        priors. :data:`DEFAULT_GROUP_WEIGHTS` already carries everything a field needs,
        and an earlier attempt to raise the ``adp`` group for field mode had two side
        effects worth remembering: ``adp/scaler_U`` and ``adp/scaler_log_scale`` sit under
        that group, so it multiplied the scaler regularisation by the same factor, and it
        made the field's configuration differ from every per-atom baseline in a way that
        had nothing to do with ADPs.

        Safe to call after construction: the targets and scales are rebuilt afterwards,
        which is what the model's own "run once at setup" caveat is about.

        Parameters
        ----------
        mode : str
            Any mode :meth:`Model.set_adp_mode` accepts. ``"field"`` and
            ``"field_aniso"`` are sized and reweighted; the per-atom modes just pass
            through, with any field weight overrides removed again.
        mode_set : str, optional
            Displacement-mode set for ``mode="field_aniso"``; see
            :data:`~torchref.model.disorder_field.MODE_SETS`.
        n_nodes : int, optional
            Explicit node count, bypassing the reflection budget entirely.
        reflections_per_parameter : float, optional
            Target work reflections per ADP parameter when ``n_nodes`` is not given.

        Returns
        -------
        dict
            What was applied: mode, mode set, node count, parameter count and the
            reflections-per-parameter actually achieved. Worth logging --- the achieved
            ratio differs from the requested one by the integer rounding of node count.
        """
        is_field = mode in self.FIELD_MODES
        if mode_set is not None and mode != "field_aniso":
            raise ValueError(
                f"mode_set={mode_set!r} describes an anisotropic displacement field; "
                'use mode="field_aniso".'
            )

        if is_field and n_nodes is None:
            n_nodes = self.nodes_for_reflection_budget(
                mode, mode_set, reflections_per_parameter, refine_node_positions
            )

        self.model.set_adp_mode(
            mode,
            aniso_selection if aniso_selection is not None else self.aniso_selection,
            n_nodes=n_nodes,
            k_neighbors=min(k_neighbors, n_nodes) if is_field else k_neighbors,
            refine_node_positions=refine_node_positions,
            mode_set=mode_set,
        )
        self.adp_mode = mode
        self.adp_mode_set = mode_set

        # Targets hold per-atom index tensors keyed off the old parametrization, the
        # scales were fitted against the old F_calc, and which ADP restraints even apply
        # is a property of the representation -- so the component set changes, not just
        # the weights. reset_loss_state is what makes the next access register the new
        # set; it also drops the Logger, which holds a reference to the old state and
        # would otherwise keep recording into it.
        if getattr(self, "adp_target", None) is not None:
            self._init_targets()
            self.reset_loss_state()

        n_par = sum(p.numel() for p in self.model.parameters_of_types(("adp", "u")))
        applied = dict(
            mode=mode, mode_set=mode_set, n_nodes=n_nodes, n_adp_parameters=int(n_par),
            reflections_per_parameter=(
                float(self.data.work.n) / n_par if n_par else float("inf")
            ),
        )
        if self.verbose > 0:
            label = mode if mode_set is None else f"{mode}/{mode_set}"
            print(
                f"ADP representation: {label}"
                + (f", {n_nodes} nodes" if is_field else "")
                + f", {n_par} parameters, "
                f"{applied['reflections_per_parameter']:.1f} work reflections each"
            )
        return applied

    def _init_targets(self, xray_mode: str = None):
        """Build the x-ray, geometry and ADP targets and initialise the scales.

        ``xray_mode`` defaults to ``self.xray_mode`` (itself ``'ml'``), so the
        deserialization path picks up the stored mode rather than the default.
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
        """``reflection_data`` under the name the weighting modules expect."""
        return self.reflection_data

    @property
    def loss_state(self) -> LossState:
        """The persistent :class:`LossState`, created on first access and reused
        across refinement cycles (targets registered once, weights re-applied)."""
        if self._loss_state is None:
            self._loss_state = self._create_loss_state()
        return self._loss_state

    @property
    def logger(self) -> Logger:
        """The :class:`Logger` bound to :attr:`loss_state`, created on first access."""
        if self._logger is None:
            self._logger = Logger(
                state=self.loss_state,
                verbose=self.verbose,
            )
        return self._logger

    def reset_loss_state(self) -> None:
        """Drop the persistent LossState and Logger so targets are re-registered.

        Required after changing target modes or reinitializing targets; without it
        the stale state keeps serving the old targets and weights.
        """
        self._loss_state = None
        self._logger = None

    def refine_scaler(self):
        """Refit the scaler against the current model.

        **The only place a scale gets fitted.** Every driver routes here, so the scale the
        in-run R-factor is computed from is the one an external 0-cycle score would produce.

        Warm: the existing scale is the starting point and carries across macrocycles. For a
        cold start -- fresh ``c_iso``, rebuilt solvent and anisotropy -- call
        :meth:`get_scales`.

        The objective is ``self.scale_target``, not the body target: scaling is a
        nuisance-magnitude fit that need not carry a model-error term, ``alpha`` is degenerate
        with the scale being fitted, and for ``ml_full`` the body target would put a 32-node
        quadrature inside every line-search evaluation. The fit runs on the same
        :class:`LossState` machinery as the body steps, differing only in the loss and in
        exposing only the scaler's parameters to the optimizer; see
        :meth:`~torchref.scaling.scaler_base.ScalerBase.refine_lbfgs`.

        **No-op when the scaler is ``None``** -- targets such as ``ls_wunit_k1`` in
        ``binwise_optimal`` mode compute their own scale and leave it unset.

        Returns
        -------
        dict or None
            The scaler's per-step metrics, or None when there is no scaler.
        """
        if not hasattr(self, "scaler") or self.scaler is None:
            return None
        return self.scaler.refine_lbfgs(
            scale_target=getattr(self, "scale_target", DEFAULT_SCALE_TARGET)
        )

    def get_scales(self):
        """Cold-start the scaler against the current model: ``initialize()`` then
        :meth:`refine_scaler`.

        ``initialize()`` *replaces* ``c_iso`` with a fresh parameter and rebuilds the
        solvent and anisotropy terms, so this discards a refined scale. Use it at
        construction, when the model has been swapped, or for a one-shot 0-cycle score;
        inside a macrocycle loop call :meth:`refine_scaler` instead.
        """
        if not hasattr(self, "scaler") or self.scaler is None:
            # Targets that compute their own scale (e.g. ls_wunit_k1 in
            # binwise_optimal mode) intentionally leave ref.scaler=None.
            # Nothing to initialize or refit here.
            return None
        self.scaler.initialize()
        return self.refine_scaler()

    def setup_scaler(self):
        """Construct ``self.scaler`` from ``self._scaler_class`` (default
        :class:`Scaler`), wired to the current model, data, ``nbins``,
        ``n_iso_coeff`` and device."""
        cls = getattr(self, "_scaler_class", None) or Scaler
        self.scaler = cls(
            self.model,
            self.reflection_data,
            nbins=self.nbins,
            n_iso_coeff=self.n_iso_coeff,
            verbose=self.verbose,
            device=self.device,
        )

    def _sync_model_cell_to_data(
        self,
        axis_rtol: float = 0.01,
        angle_atol_deg: float = 1.0,
    ) -> None:
        """Always cast the reflection-data cell onto the model, warning if it differs.

        Mirrors cctbx's ``xrs.customized_copy(crystal_symmetry=data_sym)``: the data
        cell is authoritative because HKL indices are defined relative to it. **Atoms
        keep their Cartesian coordinates**, so fractional coordinates implicitly
        shift. Skipping the sync computes ``F_calc`` on the wrong basis and biases the
        gradient. The warning fires only past tolerance: any axis off by
        ``axis_rtol`` relative, or any angle by ``angle_atol_deg`` absolute.
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
        """Unique parameters of this module and, with ``recurse``, its submodules.

        Deduplicates ``Module.parameters()`` in order, so a tensor shared between two
        submodules is not handed to the optimizer twice. Returns a list, not a
        generator.
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
        """Complex ``F_calc`` per reflection, from the model.

        Parameters
        ----------
        hkl : array_like, optional
            Reflection indices. None evaluates on the data's own reflections via
            ``reflection_data.structure_factors``, returning the canonical-ASU
            convention. An explicit ``hkl`` is used as given.
        recalc : bool, optional
            Force recomputation rather than reusing the cached SF.
        """
        if hkl is None:
            return self.reflection_data.structure_factors(self.model, recalc=recalc)
        return self.model(hkl, recalc=recalc)

    def get_fcalc_scaled(self, hkl=None, recalc=False):
        """``scaler(F_calc)``; see :meth:`get_fcalc` for ``hkl`` and ``recalc``."""
        fcalc = self.get_fcalc(hkl, recalc=recalc)
        fcalc_scaled = self.scaler(fcalc)
        return fcalc_scaled

    def adp_loss(self):
        """Total ADP loss: bond-based B similarity, locality smoothness, and the
        shifted inverse-gamma distribution prior registered by ``TotalADPTarget``."""
        return self.adp_target()

    def get_F_calc(self, hkl=None, recalc=False):
        """``|F_calc|``; see :meth:`get_fcalc` for ``hkl`` and ``recalc``."""
        return torch.abs(self.get_fcalc(hkl, recalc=recalc))

    def get_F_calc_scaled(self, hkl=None, recalc=False):
        """``|scaler(F_calc)|``; see :meth:`get_fcalc` for ``hkl`` and ``recalc``."""
        return torch.abs(self.get_fcalc_scaled(hkl, recalc=recalc))

    def nll_xray(self):
        """``(work_nll, test_nll)`` from the two instantiated x-ray targets."""
        return self.xray_target_work(), self.xray_target_test()

    def xray_loss_work(self) -> torch.Tensor:
        """X-ray loss on the work set."""
        return self.xray_target_work()

    def xray_loss_test(self) -> torch.Tensor:
        """X-ray loss on the test set."""
        return self.xray_target_test()

    def bond_loss(self) -> torch.Tensor:
        """Bond-length NLL component of the geometry target."""
        return self.geometry_target.target_losses()["bond_target"]

    def angle_loss(self) -> torch.Tensor:
        """Bond-angle NLL component of the geometry target."""
        return self.geometry_target.target_losses()["angle_target"]

    def torsion_loss(self) -> torch.Tensor:
        """Torsion-angle NLL component of the geometry target."""
        return self.geometry_target.target_losses()["torsion_target"]

    def geometry_loss(self) -> torch.Tensor:
        """Total geometry NLL (all components of ``TotalGeometryTarget``)."""
        return self.geometry_target()

    def _create_loss_state(self) -> LossState:
        """Register every target under its hierarchical name and apply
        :data:`DEFAULT_GROUP_WEIGHTS` via ``self.weighting``."""
        state = LossState(device=self.device)

        # Register X-ray target
        state.register_target("xray", self.xray_target_work)

        # Register geometry targets
        state.register_targets(self.geometry_target)

        # Register ADP targets
        state.register_targets(self.adp_target)
        # Scaler regularization targets require the scaler to actually have
        # the regularized parameters.
        if self.scaler is not None:
            n_ref = int(self.reflection_data.hkl.shape[0])
            if hasattr(self.scaler, "U"):
                state.register_target(
                    "adp/scaler_U",
                    ScalerURegularizationTarget(self.scaler, n_reflections=n_ref),
                )
            if (hasattr(self.scaler, "c_iso")
                    and self.scaler.c_iso.requires_grad):
                state.register_target(
                    "adp/scaler_log_scale",
                    ScalerLogScaleTrendTarget(self.scaler, n_reflections=n_ref),
                )

        # The weighting scheme returns a {component: weight} dict. This is the single
        # place the data/prior balance lives -- see DEFAULT_GROUP_WEIGHTS.
        state.set_weights(self.weighting(state))

        return state

    def create_loss_state(self) -> LossState:
        """A fresh configured LossState; prefer the persistent :attr:`loss_state`."""
        return self._create_loss_state()

    def complete_loss_state(self) -> "LossState":
        """Refresh the persistent LossState's cached losses and return it.

        The cached active-parameter leaf set is *not* refreshed. A stale leaf is
        only wasted backward work, never a wrong answer -- but after calling
        ``Model.freeze``/``unfreeze`` mid-run, call ``state.refresh_loss_leaves()``
        yourself.
        """
        state = self.loss_state
        state.cache_losses()
        return state

    def xray_loss(self):
        """Alias for :meth:`xray_loss_work`."""
        return self.xray_loss_work()

    def restraints_loss(self):
        """Alias for :meth:`geometry_loss`."""
        return self.geometry_loss()

    def collect_metrics(self) -> Dict[str, Any]:
        """R-factors, geometry and ADP stats for logging, unfiltered.

        Filtering by verbosity happens at display time, so the returned dict holds
        ``StatEntry`` objects rather than plain values.
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
        """Deprecated no-op that returns ``state`` unchanged; use
        :meth:`complete_loss_state`, which does all state setup in one call."""
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
        """Scatter-plot calculated vs observed amplitudes, saved as a PNG at
        ``outpath``."""
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
            True emits a phenix-style anomalous MTZ: maps and merged columns in the
            canonical ASU (Friedel mates merged by mean amplitude) plus unstacked
            ``F-obs(+/-)`` / ``F-model(+/-)`` on the same ASU index. False writes the
            per-row layout. None picks anomalous when the data were loaded as Bijvoet
            pairs (``reflection_data.friedel_merged`` is False).
        """
        with torch.no_grad():
            # Canonical-ASU convention, row-aligned with reflection_data.hkl --
            # the index write_mtz emits as H,K,L.
            fcalc = self.scaler(self.get_fcalc(), use_mask=False)
            self.reflection_data.write_mtz(out_mtz_path, fcalc, anomalous=anomalous)

    def collect_deposition_metadata(self, metadata=None):
        """Collect refinement statistics into a ``RefinementMetadata``.

        Parameters
        ----------
        metadata : RefinementMetadata, optional
            Existing metadata to merge with (e.g. input-file pass-through).
            Refinement statistics take precedence over pass-through values.
        """
        from torchref.io.metadata import RefinementMetadata

        refinement_meta = RefinementMetadata.from_refinement(self)

        # Merge with input file metadata if available
        if metadata is not None:
            return metadata.merge(refinement_meta)

        # Merge with pass-through headers from input file
        if self.model.ctx.input_file:
            input_file = self.model.ctx.input_file
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
        """``torch.save`` the full refinement state dict to ``path``."""
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved refinement state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """Load a saved state dict from ``path`` into this instance.

        Requires submodules that already match the checkpoint's structure; to build
        one from scratch use :meth:`create_from_state_dict`. ``strict`` enforces an
        exact key match.
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
        """Rebuild a fully initialized Refinement from a saved state dict.

        The recommended restore path: it rebuilds reflection data, model and scaler
        through their own factories before calling ``load_state_dict``, which
        :meth:`load_state` cannot do. Restraints are normally lazy via
        ``model.restraints``; the standalone handling here is a legacy state-dict
        path and does not make them a first-class persisted submodule.

        Parameters
        ----------
        state_dict : dict
            From ``torch.save(refinement.state_dict(), ...)`` or a checkpoint file.
        device : torch.device, optional
            Device to place tensors on. Defaults to the configured default device.
        verbose : int, optional
            Verbosity level. Default 1.

        Returns
        -------
        Refinement
            Fully initialized instance with restored state.
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
        from torchref.topology.restraints import Restraints

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
        instance.n_iso_coeff = 6
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
        if instance.model is not None and instance.model.ctx.initialized:
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
