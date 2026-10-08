"""One object between TorchRef coordinates and an OpenMM system.

:class:`OpenMMAdapter` owns the OpenMM ``System``, its ``Context`` and the map from
TorchRef rows to OpenMM particles, and is the only place coordinates and forces cross
between the two. :meth:`OpenMMAdapter.from_model` builds the system straight from the
model -- identity and bonds through :mod:`torchref.experimental.mm.topology`, ligand
templates through :mod:`torchref.experimental.mm.ligands`, copies and periodic box
through a :class:`~torchref.experimental.mm.layout.CrystalLayout` -- without writing
or re-reading a structure file. :meth:`OpenMMAdapter.energy` is differentiable: the
gradient is OpenMM's force, returned through the layout to the coordinates.
"""

from __future__ import annotations

import io
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from torchref.config import get_int_dtype
from torchref.experimental.mm.layout import CrystalLayout
from torchref.experimental.mm.topology import (
    MIN_HYDROGEN_FRACTION,
    AtomSelection,
    build_asu_topology,
    check_hydrogens,
    missing_hydrogens,
)

if TYPE_CHECKING:
    from torchref.model.model import Model

#: Force fields the adapter loads by default: AMBER ff14SB for protein and nucleic
#: acids, with TIP3P-FB water and its ion parameters.
DEFAULT_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")

#: Platforms tried, in order, after the one matching the coordinates' device.
_PLATFORM_FALLBACK = ("OpenCL", "CPU", "Reference")

_NM_PER_ANGSTROM = 0.1


class _OpenMMEnergy(torch.autograd.Function):
    """OpenMM energy of particle positions in nm, with OpenMM's forces as gradient.

    Forward returns the potential energy in kJ/mol; backward returns minus the force in
    kJ/mol/nm, clipped per particle at ``adapter.max_force``.
    """

    @staticmethod
    def forward(ctx, positions_nm: torch.Tensor, adapter: "OpenMMAdapter"):
        energy, forces = adapter._evaluate(positions_nm.detach().cpu().numpy())
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces * np.minimum(adapter.max_force / np.maximum(norms, 1e-10), 1.0)
        ctx.save_for_backward(
            torch.as_tensor(
                forces, dtype=positions_nm.dtype, device=positions_nm.device
            )
        )
        return positions_nm.new_tensor(energy)

    @staticmethod
    def backward(ctx, grad_output):
        (forces,) = ctx.saved_tensors
        return -forces * grad_output, None


class OpenMMAdapter:
    """An OpenMM system over copies of a TorchRef model's atoms.

    Build with :meth:`from_model`. Coordinates go in as TorchRef Cartesian Å, energies
    come out in kJ/mol.

    Parameters
    ----------
    system : openmm.System
        One particle per atom of every present copy, in ``topology`` order.
    topology : openmm.app.Topology
    layout : CrystalLayout
        How the copies are placed from the coordinate sets.
    particles : numpy.ndarray
        For each particle, its row of the flattened ``(C, n_model_atoms, 3)`` copy
        coordinates, shape ``(n_particles,)``. This is the TorchRef-to-OpenMM map.
    n_model_atoms : int
        Atoms per coordinate set.
    present : numpy.ndarray
        Which copies hold each molecule, shape ``(C, n_molecules)``.
    platform : str, optional
        OpenMM platform name; by default the one matching the coordinates' device.
    max_force : float
        Per-particle force clip in kJ/mol/nm applied to the gradient (not the energy).
    verbose : int

    Notes
    -----
    Every evaluation copies the positions to host memory and the forces back, a
    GPU-to-CPU synchronisation when the coordinates live on an accelerator. The
    ``Context`` is created on first evaluation. Only first derivatives exist.
    """

    def __init__(
        self,
        system,
        topology,
        layout: CrystalLayout,
        particles: np.ndarray,
        n_model_atoms: int,
        present: np.ndarray,
        platform: Optional[str] = None,
        max_force: float = 10000.0,
        verbose: int = 0,
    ) -> None:
        self.system = system
        self.topology = topology
        self.layout = layout
        self.particles = np.asarray(particles, dtype=np.int64)
        self.n_model_atoms = int(n_model_atoms)
        self.present = present
        self.platform = platform
        self.max_force = float(max_force)
        self.verbose = int(verbose)
        self.platform_name = "none"
        self._context = None
        self._device_type = "cpu"
        self._index: Dict[torch.device, torch.Tensor] = {}
        if system.getNumParticles() != len(self.particles):
            raise ValueError(
                f"System has {system.getNumParticles()} particles but the map names "
                f"{len(self.particles)}"
            )
        for group, force in enumerate(system.getForces()):
            force.setForceGroup(group)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: "Model",
        *,
        atoms: AtomSelection = None,
        layout: Optional[CrystalLayout] = None,
        xyz: Optional[torch.Tensor] = None,
        forcefield: Sequence[str] = DEFAULT_FORCEFIELD,
        charge_method: str = "gas",
        residue_charges: Optional[Dict[str, int]] = None,
        nonbonded: str = "cutoff",
        cutoff: float = 5.0,
        ewald_error_tolerance: float = 5e-4,
        overlap_cutoff: float = 1.5,
        hydrogens_added: Optional[bool] = None,
        platform: Optional[str] = None,
        max_force: float = 10000.0,
        verbose: int = 0,
    ) -> "OpenMMAdapter":
        """Parameterise a model with AMBER force fields and lay it out for OpenMM.

        Parameters
        ----------
        model : Model
            Single-conformation model carrying the hydrogens the force field needs.
            Not modified.
        atoms : str, array-like or None
            Whole molecules to include (selection string, mask or rows); default all.
        layout : CrystalLayout, optional
            Copies and periodic box; default :meth:`CrystalLayout.isolated`.
        xyz : torch.Tensor, optional
            Coordinates the copy presence is decided on, shape ``(S, n_atoms, 3)`` in Å,
            one set per layout source. Default ``model.xyz()``.
        forcefield : sequence of str
            OpenMM force-field XML files. Residues none of them match are
            parameterised with GAFF2 from their monomer dictionary.
        charge_method : {"gas", "bcc"}
            antechamber charge method for those residues: Gasteiger or AM1-BCC.
        residue_charges : dict, optional
            Net charge per residue name, overriding the dictionary's formal charges.
        nonbonded : {"cutoff", "pme", "none"}
            ``"cutoff"`` is reaction-field electrostatics within ``cutoff`` (periodic or
            not, following the layout), ``"pme"`` particle-mesh Ewald (periodic layouts
            only), ``"none"`` every pair (non-periodic only).
        cutoff : float
            Non-bonded cutoff in Å.
        ewald_error_tolerance : float
            PME accuracy.
        overlap_cutoff : float
            Distance in Å below which two copies of a molecule are the same site
            (:meth:`CrystalLayout.presence`); ``0`` keeps every copy.
        hydrogens_added : bool, optional
            Whether TorchRef generated the model's missing hydrogens; when False every
            build warns. Default: ``model.ctx.hydrogens_generated``.
        platform : str, optional
            OpenMM platform name.
        max_force : float
            Per-particle force clip in kJ/mol/nm.
        verbose : int

        Returns
        -------
        OpenMMAdapter

        Raises
        ------
        ValueError
            If fewer than half the dictionary's hydrogens are present, a residue
            matches no template, or the non-bonded method does not fit the layout.
        """
        import openmm.app as app
        import openmm.unit as unit

        layout = CrystalLayout.isolated() if layout is None else layout
        method = _nonbonded_method(nonbonded, layout.periodic)
        asu = build_asu_topology(model, atoms)
        if hydrogens_added is None:
            hydrogens_added = model.ctx.hydrogens_generated
        check_hydrogens(asu.restraints.topology.atoms, asu.rows, hydrogens_added)

        ff = app.ForceField(*forcefield)
        templates, unmatched = _match_templates(ff, asu)
        _refuse_incomplete(ff, asu, unmatched)
        if unmatched:
            from torchref.experimental.mm.ligands import gaff2_forcefield

            xml = gaff2_forcefield(
                asu,
                unmatched,
                model.xyz().detach().cpu().numpy(),
                charge_method=charge_method,
                residue_charges=residue_charges,
                verbose=verbose,
            )
            ff.loadFile(io.StringIO(xml))
            templates, _ = _match_templates(ff, asu, require_all=True)

        if xyz is None:
            xyz = model.xyz()
        sources = xyz.detach().cpu().numpy().astype(np.float64)
        if sources.ndim == 2:
            sources = sources[None]
        if sources.shape[:2] != (layout.n_sources, asu.n_model_atoms):
            raise ValueError(
                f"xyz must be ({layout.n_sources}, {asu.n_model_atoms}, 3) for this "
                f"layout, got {tuple(xyz.shape)}"
            )
        present = layout.presence(
            sources[:, asu.rows], asu.molecule_of, asu.is_heavy, overlap_cutoff
        )
        clashes = layout.overlaps(
            sources[:, asu.rows], asu.molecule_of, asu.is_heavy, present, overlap_cutoff
        )
        if clashes:
            examples = [
                f"{_molecule_label(asu, a)} (copy {ca}) / "
                f"{_molecule_label(asu, b)} (copy {cb})"
                for ca, a, cb, b in clashes[:5]
            ]
            warnings.warn(
                f"{len(clashes)} pairs of different molecules overlap within "
                f"{overlap_cutoff} Å in the crystal, e.g. {examples}; their copies "
                "stack on each other. Check the model there.",
                UserWarning,
                stacklevel=2,
            )
        box_nm = None if layout.box is None else layout.box * _NM_PER_ANGSTROM
        topology, particles, origin = asu.to_openmm(present, box_nm)
        residue_templates = {
            residue: templates[origin[k]]
            for k, residue in enumerate(topology.residues())
        }
        system = ff.createSystem(
            topology,
            nonbondedMethod=getattr(app, method),
            nonbondedCutoff=cutoff * unit.angstrom,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False,
            ewaldErrorTolerance=ewald_error_tolerance,
            residueTemplates=residue_templates,
        )
        adapter = cls(
            system,
            topology,
            layout,
            particles,
            asu.n_model_atoms,
            present,
            platform=platform,
            max_force=max_force,
            verbose=verbose,
        )
        if verbose > 0:
            absent = int((~present).sum())
            print(
                f"[OpenMMAdapter] {len(particles)} particles in {layout.n_copies} "
                f"cop{'y' if layout.n_copies == 1 else 'ies'}, {len(unmatched)} "
                f"GAFF2 residue(s), {absent} molecule copies absent as overlapping"
            )
        return adapter

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @property
    def n_particles(self) -> int:
        """Number of OpenMM particles."""
        return len(self.particles)

    @property
    def context(self):
        """The ``openmm.Context``, created on first access."""
        if self._context is None:
            self._context, self.platform_name = self._create_context()
        return self._context

    def positions(self, xyz: torch.Tensor) -> torch.Tensor:
        """Particle positions in nm, differentiable in ``xyz``.

        Parameters
        ----------
        xyz : torch.Tensor
            Shape ``(S, n_model_atoms, 3)`` or ``(n_model_atoms, 3)``, Cartesian Å.

        Returns
        -------
        torch.Tensor
            Shape ``(n_particles, 3)``, nm, on ``xyz``'s device and dtype.
        """
        if self._context is None:
            self._device_type = xyz.device.type
        copies = self.layout.positions(xyz)
        if copies.shape[1] != self.n_model_atoms:
            raise ValueError(
                f"Expected {self.n_model_atoms} atoms per coordinate set, got "
                f"{copies.shape[1]}; rebuild after changing the model's atoms."
            )
        flat = copies.reshape(-1, 3).index_select(0, self._index_on(xyz.device))
        return flat * _NM_PER_ANGSTROM

    def energy(self, xyz: torch.Tensor) -> torch.Tensor:
        """Potential energy of the system, differentiable in ``xyz``.

        Parameters
        ----------
        xyz : torch.Tensor
            Shape ``(S, n_model_atoms, 3)`` or ``(n_model_atoms, 3)``, Cartesian Å.

        Returns
        -------
        torch.Tensor
            Scalar, kJ/mol, summed over every particle of every copy. The gradient is
            OpenMM's force clipped at :attr:`max_force` per particle, so it is the exact
            derivative only below the clip.
        """
        return _OpenMMEnergy.apply(self.positions(xyz), self)

    def energy_and_forces(self, xyz: torch.Tensor) -> Tuple[float, np.ndarray]:
        """Energy and unclipped forces on the coordinate sets, without autograd.

        Parameters
        ----------
        xyz : torch.Tensor
            Shape ``(S, n_model_atoms, 3)`` or ``(n_model_atoms, 3)``, Cartesian Å.

        Returns
        -------
        energy : float
            kJ/mol.
        forces : numpy.ndarray
            ``-dE/dxyz`` in kJ/mol/Å, shape ``(S, n_model_atoms, 3)``: each copy's
            particle forces rotated back to its source and summed.
        """
        with torch.no_grad():
            positions = self.positions(xyz).cpu().numpy()
        energy, forces = self._evaluate(positions)
        per_copy = np.zeros((self.layout.n_copies * self.n_model_atoms, 3))
        per_copy[self.particles] = forces * _NM_PER_ANGSTROM
        per_copy = per_copy.reshape(self.layout.n_copies, self.n_model_atoms, 3)
        back = np.einsum("cji,cnj->cni", self.layout.rotation, per_copy)
        result = np.zeros((self.layout.n_sources, self.n_model_atoms, 3))
        np.add.at(result, self.layout.source, back)
        return energy, result

    def group_energies(self, xyz: torch.Tensor) -> Dict[str, float]:
        """Energy of each force in kJ/mol, keyed by its class name.

        Parameters
        ----------
        xyz : torch.Tensor
            Shape ``(S, n_model_atoms, 3)`` or ``(n_model_atoms, 3)``, Cartesian Å.

        Returns
        -------
        dict
        """
        import openmm.unit as unit

        with torch.no_grad():
            self._set_positions(self.positions(xyz).cpu().numpy())
        result = {}
        for group, force in enumerate(self.system.getForces()):
            state = self.context.getState(getEnergy=True, groups={group})
            result[type(force).__name__] = state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
        return result

    def minimize(self, xyz: torch.Tensor, max_iterations: int = 200) -> torch.Tensor:
        """Minimise the system's energy from ``xyz`` and map the result back.

        Parameters
        ----------
        xyz : torch.Tensor
            Shape ``(S, n_model_atoms, 3)`` or ``(n_model_atoms, 3)``, Cartesian Å.
        max_iterations : int

        Returns
        -------
        torch.Tensor
            Same shape, device and dtype as ``xyz``. Each atom is the average of its
            minimised copies taken back to the source frame; atoms in no copy keep
            their input coordinates. A copy moved by OpenMM is generally no longer an
            exact symmetry image of the others, hence the average.
        """
        import openmm
        import openmm.unit as unit

        squeeze = xyz.dim() == 2
        with torch.no_grad():
            copies = (
                self.layout.positions(xyz).detach().cpu().numpy().astype(np.float64)
            )
            self._set_positions(self.positions(xyz).cpu().numpy())
        openmm.LocalEnergyMinimizer.minimize(self.context, maxIterations=max_iterations)
        state = self.context.getState(getPositions=True)
        minimised = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        )
        flat = copies.reshape(-1, 3)
        flat[self.particles] = minimised
        back = self.layout.to_source_frame(flat.reshape(copies.shape))
        held = np.zeros(self.layout.n_copies * self.n_model_atoms)
        held[self.particles] = 1.0
        held = held.reshape(self.layout.n_copies, self.n_model_atoms)
        total = np.zeros((self.layout.n_sources, self.n_model_atoms, 3))
        count = np.zeros((self.layout.n_sources, self.n_model_atoms))
        np.add.at(total, self.layout.source, back * held[..., None])
        np.add.at(count, self.layout.source, held)
        source_xyz = xyz.detach().cpu().numpy().reshape(total.shape)
        result = np.where(
            count[..., None] > 0, total / np.maximum(count, 1.0)[..., None], source_xyz
        )
        out = torch.as_tensor(result, dtype=xyz.dtype, device=xyz.device)
        return out[0] if squeeze else out

    # ------------------------------------------------------------------
    # OpenMM boundary
    # ------------------------------------------------------------------

    def _index_on(self, device: torch.device) -> torch.Tensor:
        """:attr:`particles` as an index tensor on ``device``, cached."""
        if device not in self._index:
            self._index[device] = torch.as_tensor(
                self.particles, dtype=get_int_dtype(), device=device
            )
        return self._index[device]

    def _set_positions(self, positions_nm: np.ndarray) -> None:
        self.context.setPositions(np.asarray(positions_nm, dtype=np.float64))

    def _evaluate(self, positions_nm: np.ndarray) -> Tuple[float, np.ndarray]:
        """Energy in kJ/mol and forces in kJ/mol/nm at particle positions in nm."""
        import openmm.unit as unit

        self._set_positions(positions_nm)
        state = self.context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        return float(energy), np.asarray(forces, dtype=np.float64)

    def _create_context(self):
        """A working ``(Context, platform name)``, trying platforms in order."""
        import openmm

        import torchref

        if self.platform is not None:
            names = [self.platform]
        else:
            preferred = "CUDA" if self._device_type == "cuda" else "CPU"
            names = list(dict.fromkeys([preferred, *_PLATFORM_FALLBACK]))
        errors = []
        for name in names:
            try:
                platform = openmm.Platform.getPlatformByName(name)
                properties = {"Threads": str(torchref.N_CPUS)} if name == "CPU" else {}
                context = openmm.Context(
                    self.system, openmm.VerletIntegrator(0.001), platform, properties
                )
                if self.verbose > 0:
                    print(f"[OpenMMAdapter] OpenMM platform: {name}")
                return context, name
            except Exception as exc:  # noqa: BLE001 -- any failure means try the next
                errors.append(f"{name}: {exc}")
        raise RuntimeError("No usable OpenMM platform: " + "; ".join(errors))


def template_coverage(
    model: "Model", forcefield: Sequence[str] = DEFAULT_FORCEFIELD
) -> np.ndarray:
    """Which model atoms belong to residues a force-field XML template matches.

    Parameters
    ----------
    model : Model
        Single-conformation model.
    forcefield : sequence of str
        OpenMM force-field XML files.

    Returns
    -------
    numpy.ndarray
        Boolean mask over model rows, shape ``(N,)``: False for the residues that would
        need GAFF2 templates. Usable as ``atoms=`` of :meth:`OpenMMAdapter.from_model`
        unless such a residue is covalently bonded to a matched one.
    """
    import openmm.app as app

    asu = build_asu_topology(model)
    _, unmatched = _match_templates(app.ForceField(*forcefield), asu)
    mask = np.zeros(asu.n_model_atoms, dtype=bool)
    mask[asu.rows] = True
    for r in unmatched:
        mask[asu.rows[asu.residue_start[r] : asu.residue_start[r + 1]]] = False
    return mask


def _molecule_label(asu, molecule: int) -> str:
    """Label of a molecule's first residue."""
    first = int(np.flatnonzero(asu.molecule_of == molecule)[0])
    return asu.residue_label(int(asu.residue_of[first]))


def _refuse_incomplete(ff, asu, unmatched) -> None:
    """Raise for unmatched residues that are incomplete rather than unknown.

    A residue named like a force-field template (a water, an amino acid) or bonded to
    another residue failed to match because atoms are missing: a model to prepare, not
    a ligand for GAFF2. So is a ligand holding fewer than
    :data:`~torchref.experimental.mm.topology.MIN_HYDROGEN_FRACTION` of its
    dictionary's hydrogens; one short of a few is a protonation state GAFF2 takes.
    """
    known = getattr(ff, "_templates", {})
    graph = asu.restraints.topology.atoms
    lacking = missing_hydrogens(asu, unmatched)
    template = (
        graph.template_h_count.cpu().numpy()[asu.rows]
        if graph.template_h_count is not None
        else np.zeros(asu.n_atoms, dtype=np.int64)
    )
    refused = []
    for r in unmatched:
        start, end = asu.residue_start[r], asu.residue_start[r + 1]
        inside = (asu.bonds >= start) & (asu.bonds < end)
        expected = int(template[start:end][template[start:end] > 0].sum())
        missing = lacking.get(asu.residue_label(r), 0)
        if (
            asu.residue_name[r] in known
            or (inside[:, 0] != inside[:, 1]).any()
            or (expected and expected - missing < MIN_HYDROGEN_FRACTION * expected)
        ):
            refused.append(asu.residue_label(r))
    if refused:
        examples = {label: lacking[label] for label in refused[:10] if label in lacking}
        raise ValueError(
            "The model is not AMBER-compatible; prepare missing atoms, terminal groups "
            f"and protonation in TorchRef. Residues matching no template: "
            f"{refused[:10]}; lacking hydrogens their dictionary names: {examples}."
        )


def _nonbonded_method(nonbonded: str, periodic: bool) -> str:
    """The ``openmm.app`` non-bonded method name for a choice and layout."""
    methods = {
        ("cutoff", False): "CutoffNonPeriodic",
        ("cutoff", True): "CutoffPeriodic",
        ("pme", True): "PME",
        ("none", False): "NoCutoff",
    }
    try:
        return methods[(nonbonded, periodic)]
    except KeyError:
        kind = "periodic" if periodic else "non-periodic"
        raise ValueError(
            f"nonbonded={nonbonded!r} is not available for a {kind} layout"
        ) from None


def _match_templates(ff, asu, require_all: bool = False):
    """Template name of each residue, and the residues no template matches.

    A single-atom residue named like a single-atom template takes that template: OpenMM
    matches ions by element alone and cannot choose between, say, ``FE`` and ``FE2``.
    The other residues are matched by OpenMM on bonds and elements.

    Parameters
    ----------
    ff : openmm.app.ForceField
    asu : AsuTopology
    require_all : bool
        Raise unless every residue matches.

    Returns
    -------
    names : dict
        ``{residue: template name}`` for the matched residues of ``asu``.
    unmatched : list of int
        Residues of ``asu`` without a template.
    """
    known = getattr(ff, "_templates", {})
    sizes = np.bincount(asu.molecule_of, minlength=asu.n_molecules)
    names = {}
    for r in np.flatnonzero(np.diff(asu.residue_start) == 1):
        name = str(asu.residue_name[r])
        template = known.get(name)
        molecule = asu.molecule_of[asu.residue_start[r]]
        if template is not None and len(template.atoms) == 1 and sizes[molecule] == 1:
            names[int(r)] = name
    present = np.ones((1, asu.n_molecules), dtype=bool)
    present[0, asu.molecule_of[asu.residue_start[list(names)]]] = False
    topology, _, origin = asu.to_openmm(present)
    try:
        unmatched = [
            int(origin[res.index]) for res in ff.getUnmatchedResidues(topology)
        ]
        if unmatched and not require_all:
            return names, unmatched
        for k, template in enumerate(ff.getMatchingTemplates(topology)):
            names[int(origin[k])] = template.name
    except Exception as exc:  # OpenMM raises a bare Exception for ambiguous matches
        lacking = dict(list(missing_hydrogens(asu, range(asu.n_residues)).items())[:10])
        detail = (
            f" Residues lacking hydrogens their dictionary names: {lacking}."
            if lacking
            else ""
        )
        raise ValueError(
            "The model is not AMBER-compatible; prepare missing atoms, terminal groups "
            f"and protonation in TorchRef.{detail} OpenMM: {exc}"
        ) from exc
    return names, []
