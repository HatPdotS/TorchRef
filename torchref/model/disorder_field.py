"""Node-field parametrization of the atomic displacement parameters.

A disorder field stores disorder parameters on a small set of **nodes** and gives each
atom a distance-weighted mean of the nodes near it, so the parameter count scales with
node count rather than atom count.

This is :class:`~torchref.model.parameter_wrappers.OccupancyTensor`'s collapse-and-expand
with a soft, distance-derived expansion in place of a fixed integer assignment: storage
is ``(K, 2)`` per node, ``forward()`` returns one B per atom. Two index spaces therefore
meet in this class, and callers must not mix them --- masks handed to
:meth:`~DisorderFieldTensor.update_refinable_mask` are in ATOM space, while
``refinable_mask`` and :meth:`get_refinable_count` are in NODE space.

A node's position is anchored, not free: it is the centroid of the atoms in its anchor
cluster, plus an optional refinable offset. Anchoring keeps a node inside the molecule
and moving with the model, so the offset says only where it sits *relative* to the atoms
it serves and cannot wander off into solvent.
"""

import math
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from torchref.config import get_float_dtype, get_int_dtype, normalize_device
from torchref.model.parameter_wrappers import (
    MixedTensor,
    chol_param_count,
    psd_to_raw,
    raw6_to_u6,
    raw_to_cholesky,
    u6_to_matrix,
    u6_to_raw6,
)
from torchref.utils.utils import ModuleReference

__all__ = [
    "DisorderFieldTensor",
    "NodePayload",
    "IsotropicPayload",
    "AnisotropicPayload",
    "ModeCovariancePayload",
    "MODE_SETS",
    "PAYLOAD_CODES",
    "payload_code",
    "payload_from_code",
    "farthest_point_anchors",
    "density_anchor_rows",
    "build_neighbor_list",
]


def farthest_point_anchors(xyz: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Pick ``n_nodes`` well-spread anchor atoms, then relax them onto local density.

    Greedy farthest-point selection seeded from the atom nearest the centroid, followed
    by Lloyd iterations that move each anchor to the atom closest to its cluster mean.
    Farthest-point alone favours extremities; the Lloyd pass pulls the anchors back onto
    where atoms actually are, which is what a disorder field wants.

    Deterministic end to end --- no RNG --- so two processes produce the same anchors.
    Restraint row order was hash-seed dependent in this package once; node placement is
    not going to be.

    Parameters
    ----------
    xyz : torch.Tensor
        ``(N, 3)`` atom coordinates.
    n_nodes : int
        Number of anchors to pick. Clamped to ``N``.

    Returns
    -------
    torch.Tensor
        ``(K,)`` int64 atom indices, sorted ascending.
    """
    n_atoms = int(xyz.shape[0])
    n_nodes = max(1, min(int(n_nodes), n_atoms))

    centroid = xyz.mean(dim=0, keepdim=True)
    first = int(torch.cdist(centroid, xyz).argmin())

    chosen = [first]
    d2_nearest = ((xyz - xyz[first]) ** 2).sum(-1)
    for _ in range(n_nodes - 1):
        nxt = int(d2_nearest.argmax())
        chosen.append(nxt)
        d2_nearest = torch.minimum(d2_nearest, ((xyz - xyz[nxt]) ** 2).sum(-1))

    anchors = torch.tensor(chosen, dtype=torch.int64, device=xyz.device)  # dtype-ok: anchor atom indices; torch indexing requires int64

    # Lloyd relaxation, snapping to real atoms so an anchor is always an atom index.
    for _ in range(10):
        assign = torch.cdist(xyz, xyz[anchors]).argmin(dim=1)
        moved = anchors.clone()
        for j in range(anchors.shape[0]):
            members = (assign == j).nonzero(as_tuple=True)[0]
            if members.numel() == 0:
                continue
            mean = xyz[members].mean(dim=0, keepdim=True)
            moved[j] = members[int(torch.cdist(mean, xyz[members]).argmin())]
        moved = torch.unique(moved)
        if moved.shape[0] == anchors.shape[0] and bool((moved == anchors).all()):
            break
        anchors = moved

    return torch.sort(anchors).values


def density_anchor_rows(xyz: torch.Tensor, n_nodes: int):
    """Anchor each node on its whole density cluster rather than on one atom.

    :func:`farthest_point_anchors` snaps every anchor onto an atom, which puts each node
    exactly *on* an atom -- so narrowing its kernel isolates the atom it is already
    standing on, and the node ends up owning a single atom outright. Anchoring on the
    cluster instead places the node at the cluster centroid, generally between atoms, so
    there is no atom for it to fall onto.

    Returns
    -------
    tuple of torch.Tensor
        ``(atom index per entry, node index per entry)``, flat and ragged, suitable for
        ``DisorderFieldTensor(anchor_rows=...)``. Every node keeps at least its seed
        atom, so no node is left with an empty neighbourhood.
    """
    seeds = farthest_point_anchors(xyz, n_nodes)
    assign = torch.cdist(xyz, xyz[seeds]).argmin(dim=1)
    atom_idx = torch.arange(xyz.shape[0], dtype=torch.int64, device=xyz.device)  # dtype-ok: arange atom indices; index requires int64

    # A seed whose cluster somehow came out empty still needs a position.
    present = torch.bincount(assign, minlength=seeds.shape[0]) > 0
    if not bool(present.all()):
        missing = (~present).nonzero(as_tuple=True)[0]
        atom_idx = torch.cat([atom_idx, seeds[missing]])
        assign = torch.cat([assign, missing])

    order = torch.argsort(assign)
    return atom_idx[order], assign[order]


def build_neighbor_list(
    xyz: torch.Tensor, node_pos: torch.Tensor, k: int
) -> torch.Tensor:
    """The ``k`` nearest nodes to each atom.

    A dense ``cdist`` plus ``topk``. Node counts are small by construction --- the whole
    point of the representation --- so ``(N, K)`` stays cheap and a spatial cell list
    (``topology.nonbonded.build_cell_list``) would only add machinery. Revisit if node
    counts ever approach atom counts.

    Parameters
    ----------
    xyz : torch.Tensor
        ``(N, 3)`` atom coordinates.
    node_pos : torch.Tensor
        ``(K, 3)`` node positions.
    k : int
        Candidates per atom. Clamped to ``K``.

    Returns
    -------
    torch.Tensor
        ``(N, k)`` int64 node indices, nearest first.
    """
    k = max(1, min(int(k), int(node_pos.shape[0])))
    d = torch.cdist(xyz, node_pos)
    return d.topk(k, dim=1, largest=False).indices.contiguous()


def _wrap_accessor(xyz_fn):
    """Hold a coordinate accessor without registering it as a submodule.

    ``model.xyz`` is itself an ``nn.Module``, so a plain assignment would enrol it in
    this wrapper's module tree and drag it into ``state_dict``, ``.to()`` and
    ``deepcopy``. :class:`~torchref.utils.utils.ModuleReference` exists for exactly that
    and is what the device-conformance walker already knows how to follow. A bare
    callable needs no wrapping.
    """
    if isinstance(xyz_fn, nn.Module):
        return ModuleReference(xyz_fn)
    return xyz_fn


# ----------------------------------------------------------------------------------
# Payload strategies. A payload says what one node carries and how that becomes a
# per-atom quantity; it knows nothing about where nodes are or how weights arise.
#
# Deliberately stateless. Node parameters live in one flat leaf on the field itself,
# because ``Model.parameters_of_types`` reads a single ``refinable_params`` per type,
# and a strategy owning parameters would break that. Each strategy declares only how
# many columns of that leaf it interprets.
# ----------------------------------------------------------------------------------


class NodePayload:
    """What a node carries, and how it becomes a per-atom ADP.

    Attributes
    ----------
    width : int
        Columns of node storage this payload interprets.
    out_width : int
        Components of the per-atom output: 1 for an isotropic B, 6 for a U tensor.
    """

    width: int = 1
    out_width: int = 1

    def contributions(self, payload, xyz, node_pos, neighbor_list):
        """``(n_atoms, k, out_width)``: what each candidate node offers each atom.

        ``xyz`` and ``node_pos`` are passed even though the payloads here ignore them,
        because a payload with an r-dependence (TLS: constant, linear and quadratic in
        the displacement from the node) needs them, and giving it the arguments now
        means adding one later touches no shared code.
        """
        raise NotImplementedError

    def fit(self, target, w_dense, epsilon, xyz, node_pos, neighbor_list):
        """``(K, width)`` payload whose field reproduces ``target`` as closely as it can.

        Takes the same geometric context as :meth:`contributions` and for the same
        reason: an r-dependent payload cannot build its modes without it. ``w_dense`` is
        the ``(n_atoms, K)`` weight matrix at the seeded kernel widths, which is what
        makes the payload-only problem linear.
        """
        raise NotImplementedError

    def log_magnitude(self, payload):
        """``(K,)`` log of each node's ADP magnitude, for a magnitude restraint.

        Lets a restraint price node values without branching on payload type.
        """
        raise NotImplementedError


class IsotropicPayload(NodePayload):
    """One isotropic B per node, stored as ``log B`` so it stays positive.

    The per-atom B is a convex combination of positive node values, so it is positive
    without a clamp.
    """

    width = 1
    out_width = 1

    def contributions(self, payload, xyz, node_pos, neighbor_list):
        return torch.exp(payload[:, 0])[neighbor_list].unsqueeze(-1)

    def fit(self, target, w_dense, epsilon, xyz, node_pos, neighbor_list):
        b = _ridged_solve(w_dense, target.unsqueeze(-1)).squeeze(-1)
        return torch.log(b.clamp(min=epsilon)).unsqueeze(-1)

    def log_magnitude(self, payload):
        return payload[:, 0]


class AnisotropicPayload(NodePayload):
    """A full U tensor per node, positive-definite by construction.

    Stored as the six free parameters of a Cholesky factor, so ``U = L L^T`` is PD for
    any parameter value -- the same device
    :class:`~torchref.model.parameter_wrappers.CholeskyMixedTensor` uses for per-atom
    ADPs, and for the same reason: an indefinite U makes the anisotropic B-matrix
    singular and the structure-factor FFT returns NaN.

    Positive-definiteness survives the combination for free: the per-atom U is a convex
    combination of PD matrices. Averaging in U space is what buys that -- averaging the
    Cholesky parameters instead would be a different object with no such guarantee.
    """

    width = 6
    out_width = 6

    def __init__(self, epsilon: float = 1e-3):
        # Floor on the Cholesky diagonal, which bounds the smallest eigenvalue of U
        # from below. Same default and same meaning as the per-atom wrapper.
        self.epsilon = float(epsilon)

    def contributions(self, payload, xyz, node_pos, neighbor_list):
        return raw6_to_u6(payload, self.epsilon)[neighbor_list]

    def fit(self, target, w_dense, epsilon, xyz, node_pos, neighbor_list):
        """Fit six U components at once, then re-encode as Cholesky parameters.

        The per-atom U is linear in each component independently, so this is the same
        ridged solve as the isotropic case with a six-column right-hand side. The
        least-squares result is not constrained to be PD, which is why it goes back
        through ``u6_to_raw6`` -- that projects onto PD by clamping eigenvalues.
        """
        if target.ndim == 1:  # a B target: lift to the equivalent isotropic U
            u_iso = target / (8.0 * math.pi**2)
            zero = torch.zeros_like(u_iso)
            target = torch.stack([u_iso, u_iso, u_iso, zero, zero, zero], dim=1)
        return u6_to_raw6(_ridged_solve(w_dense, target), self.epsilon)

    def log_magnitude(self, payload):
        u6 = raw6_to_u6(payload, self.epsilon)
        b_eq = (8.0 * math.pi**2 / 3.0) * (u6[:, 0] + u6[:, 1] + u6[:, 2])
        return torch.log(b_eq.clamp(min=1e-6))


# ----------------------------------------------------------------------------------
# Displacement-mode generators. A gradient mode is a constant 3x3 matrix G acting on the
# displacement r from the node, giving the displacement field psi(r) = G r. Rotation,
# dilation and deviatoric strain together span every linear displacement field, and
# splitting them that way is what lets a mode set stop partway.
# ----------------------------------------------------------------------------------

_SQ2 = math.sqrt(2.0)
_SQ3 = math.sqrt(3.0)
_SQ6 = math.sqrt(6.0)

# Rotations are NOT normalised: psi_i(r) = e_i x r exactly, so that the rigid mode set
# reproduces the textbook TLS formula with no stray factor. The others are Frobenius
# normalised, which is a conditioning choice and nothing more.
_GENERATORS = {
    "rotation": [
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],   # e1 x r
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],   # e2 x r
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],   # e3 x r
    ],
    "dilation": [
        [[1 / _SQ3, 0.0, 0.0], [0.0, 1 / _SQ3, 0.0], [0.0, 0.0, 1 / _SQ3]],
    ],
    "deviatoric": [
        [[1 / _SQ2, 0.0, 0.0], [0.0, -1 / _SQ2, 0.0], [0.0, 0.0, 0.0]],
        [[1 / _SQ6, 0.0, 0.0], [0.0, 1 / _SQ6, 0.0], [0.0, 0.0, -2 / _SQ6]],
        [[0.0, 1 / _SQ2, 0.0], [1 / _SQ2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1 / _SQ2], [0.0, 0.0, 0.0], [1 / _SQ2, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1 / _SQ2], [0.0, 1 / _SQ2, 0.0]],
    ],
}

#: Named mode sets, in order of expressiveness. Three translations are always present;
#: each entry lists the gradient modes added on top.
MODE_SETS = {
    "constant": (),
    "rigid": ("rotation",),
    "rigid_dilation": ("rotation", "dilation"),
    "affine": ("rotation", "dilation", "deviatoric"),
}


class ModeCovariancePayload(NodePayload):
    """A node carries the covariance of its displacement modes; TLS is one mode set.

    Instead of storing an ADP and averaging it, store the *displacement field* the node
    represents and take its covariance. With ``q`` modes ``Psi(r) = [psi_1(r) ... psi_q(r)]``
    the node's disorder is ``u(r) = Psi(r) c`` for a random coefficient vector ``c``, and
    the ADP an atom at displacement ``r`` receives is::

        U(r) = Psi(r) Sigma Psi(r)^T,   Sigma = <c c^T>,   Sigma = L L^T

    Three properties follow, and they are the whole reason for the form:

    * **Positive-semidefinite at every r, unconditionally**, because
      ``U = (Psi L)(Psi L)^T``. An arbitrary polynomial in ``r`` carries no such
      guarantee and goes indefinite somewhere --- and "somewhere" is the edge of the
      node's region, exactly where the softmax weights have not yet decayed.
    * **Spatial variation becomes intra-node and smooth by construction.** A constant-U
      node can only express variation by having neighbours, so detail costs nodes, and
      every added node is another kernel that can collapse onto a single atom. Here one
      node's U already varies across its whole region, and it cannot spike.
    * ``U(r)`` is **linear in Sigma**, so fitting stays a linear problem.

    Mode sets, from :data:`MODE_SETS`, with ``q(q+1)/2`` parameters per node:

    ==================  ===  =======  =========================================
    set                 q    params   model
    ==================  ===  =======  =========================================
    ``constant``        3    6        one U per node; same model as
                                      :class:`AnisotropicPayload`
    ``rigid``           6    21       **exactly TLS** (20 determinable; ``tr S``
                                      is the one flat direction)
    ``rigid_dilation``  7    28       TLS plus uniform breathing
    ``affine``          12   78       full linear displacement field: TLS plus
                                      shear and extension
    ==================  ===  =======  =========================================

    With the rigid set this reproduces ``U(r) = T + A S + S^T A^T + A L A^T`` identically,
    ``A`` being the matrix whose columns are ``e_i x r``: the classical TLS expression is
    what ``Psi Sigma Psi^T`` expands to when the modes are three translations and three
    rotations. Releasing the antisymmetry of the gradient -- the ``dilation`` and
    ``deviatoric`` rungs -- gives domains that breathe and shear as well as rotate.

    Displacements are divided by the node layout's own length scale (median
    nearest-neighbour node distance, detached) before the modes are built. That is pure
    conditioning: without it the gradient modes carry a factor of the domain size against
    the translations, and the curvature ratio between them runs to several hundred. It
    is detached and derived from the layout for the reason
    :class:`~torchref.refinement.targets.adp.NodeSmoothnessTarget` uses the same
    quantity: the length scale is a property of where the nodes are, not something the
    optimiser should tune.

    Memory scales as ``n_atoms * k * q^2`` for the gathered node factors, so this payload
    is meant for the small-``K`` regime it was designed for (a handful to a few dozen
    expressive nodes, with ``k_neighbors`` set to ``K``). At ``q = 12`` and
    ``k = 8`` that is ~90 MB for 20k atoms; a large ``K`` *and* a large ``k`` together
    is what to avoid.

    Parameters
    ----------
    mode_set : str, optional
        Key of :data:`MODE_SETS`. Default ``"rigid"``, i.e. TLS.
    epsilon : float, optional
        Floor on the Cholesky diagonal of ``Sigma``, which bounds its smallest
        eigenvalue. Also the value the non-translation modes start at, so a freshly
        fitted field begins as the equivalent constant-U field.
    """

    out_width = 6

    def __init__(self, mode_set: str = "rigid", epsilon: float = 1e-3):
        if mode_set not in MODE_SETS:
            raise ValueError(
                f"Unknown mode set {mode_set!r}. Available: {sorted(MODE_SETS)}."
            )
        self.mode_set = mode_set
        self.epsilon = float(epsilon)
        self._gradient_names = MODE_SETS[mode_set]
        self.q = 3 + sum(len(_GENERATORS[n]) for n in self._gradient_names)
        self.width = chol_param_count(self.q)
        self._generator_cache = {}

    def __repr__(self):
        return (
            f"ModeCovariancePayload({self.mode_set!r}, q={self.q}, "
            f"params={self.width})"
        )

    # ------------------------------------------------------------------
    # Modes.
    # ------------------------------------------------------------------

    def _generators(self, dtype, device):
        """``(q - 3, 3, 3)`` gradient generators, cached per dtype and device."""
        key = (dtype, str(device))
        G = self._generator_cache.get(key)
        if G is None:
            rows = [m for n in self._gradient_names for m in _GENERATORS[n]]
            G = torch.tensor(rows, dtype=dtype, device=device).reshape(-1, 3, 3)
            self._generator_cache[key] = G
        return G

    @staticmethod
    def _length_scale(node_pos):
        """Median nearest-neighbour node distance, as a plain float.

        Deliberately outside the graph. A median's derivative is supported on whichever
        single node pair sits at the median, which is an artifact of the layout rather
        than a direction worth following, and leaving it connected would also let the
        optimiser rescale its own modes by spreading the nodes apart. Node position
        keeps its real gradient through the displacement ``r``.
        """
        with torch.no_grad():
            if node_pos.shape[0] < 2:
                return 1.0
            d = torch.cdist(node_pos, node_pos)
            d.fill_diagonal_(float("inf"))
            return max(float(d.min(dim=1).values.median()), 1e-3)

    def modes(self, r):
        """``(..., 3, q)`` displacement modes at (already scaled) displacement ``r``."""
        eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(
            *r.shape[:-1], 3, 3
        )
        if not self._gradient_names:
            return eye
        G = self._generators(r.dtype, r.device)
        grad = torch.einsum("sij,...j->...is", G, r)
        return torch.cat([eye, grad], dim=-1)

    def sigma(self, payload):
        """``(K, q, q)`` mode covariance of each node, positive-definite."""
        L = raw_to_cholesky(payload, self.q, self.epsilon)
        return L @ L.transpose(-1, -2)

    # ------------------------------------------------------------------
    # NodePayload interface.
    # ------------------------------------------------------------------

    def contributions(self, payload, xyz, node_pos, neighbor_list):
        r = (xyz.unsqueeze(1) - node_pos[neighbor_list]) / self._length_scale(node_pos)
        Psi = self.modes(r)                                     # (N, k, 3, q)
        L = raw_to_cholesky(payload, self.q, self.epsilon)      # (K, q, q)
        A = Psi @ L[neighbor_list]                              # (N, k, 3, q)
        U = A @ A.transpose(-1, -2)                             # (N, k, 3, 3)
        return torch.stack(
            [U[..., 0, 0], U[..., 1, 1], U[..., 2, 2],
             U[..., 0, 1], U[..., 0, 2], U[..., 1, 2]],
            dim=-1,
        )

    def fit(self, target, w_dense, epsilon, xyz, node_pos, neighbor_list):
        """Seed the translation block from the constant-U solve, floor the rest.

        The full joint solve is available in principle -- ``U(r)`` is linear in
        ``Sigma``, so it is one least-squares problem in ``K * q(q+1)/2`` unknowns -- but
        it is not what is wanted here. Seeding only the translations makes the field
        start as the equivalent constant-U field, which is a state whose R-factor is
        already known, so entering this parametrisation cannot make the model worse and
        refinement can only move away from a sane point. It also sidesteps the joint
        solve's normal equations, which stop being cheap well before ``K`` does.
        """
        if target.ndim == 1:  # a B target: lift to the equivalent isotropic U
            u_iso = target / (8.0 * math.pi**2)
            zero = torch.zeros_like(u_iso)
            target = torch.stack([u_iso, u_iso, u_iso, zero, zero, zero], dim=1)
        u6 = _ridged_solve(w_dense, target)                      # (K, 6)
        sigma = u6.new_zeros(u6.shape[0], self.q, self.q)
        sigma[:, :3, :3] = u6_to_matrix(u6)
        # psd_to_raw clamps every eigenvalue to epsilon^2, so the gradient modes come
        # out at the floor rather than at zero -- non-degenerate, and negligible against
        # a real U.
        return psd_to_raw(sigma, self.epsilon)

    def log_magnitude(self, payload):
        """Log ``B_eq`` of ``U(0)``: the translation block, which is the node's own ADP.

        Evaluated at the node rather than averaged over its region, so the number means
        the same thing for every mode set and a magnitude restraint can price nodes
        without knowing which one is in use.
        """
        T = self.sigma(payload)[:, :3, :3]
        b_eq = (8.0 * math.pi**2 / 3.0) * (T[:, 0, 0] + T[:, 1, 1] + T[:, 2, 2])
        return torch.log(b_eq.clamp(min=1e-6))


def _ridged_solve(w_dense, target):
    """Least squares ``min ||W x - target||`` through the ridged normal equations.

    Non-finite target rows are dropped rather than carried in: deposited models have
    NaN ADPs, and one NaN row propagates through the normal equations and takes every
    node with it. Those atoms still receive a fitted value on output.
    """
    finite = torch.isfinite(target).all(dim=-1)
    if not bool(finite.all()):
        w_dense, target = w_dense[finite], target[finite]
        if target.shape[0] == 0:
            return torch.zeros(
                w_dense.shape[1], target.shape[-1],
                dtype=w_dense.dtype, device=w_dense.device,
            )
    gram = w_dense.T @ w_dense
    ridge = 1e-6 * torch.diagonal(gram).mean().clamp(min=1e-30)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + ridge * eye, w_dense.T @ target)


#: Stable integer code per payload, so a saved field can rebuild the one it had.
#: ``state_dict`` holds tensors only, and the payload is a plain object that never
#: reaches it, so without this a restore has to guess -- and guessing wrong is not a
#: clean failure: a mode payload restored as a constant-U one has the wrong storage
#: width and only shows up as a shape mismatch, or worse, silently different ADPs.
#:
#: **Append, never renumber.** A saved state dict holds the number.
PAYLOAD_CODES = {
    "isotropic": 0,
    "anisotropic": 1,
    "modes:constant": 2,
    "modes:rigid": 3,
    "modes:rigid_dilation": 4,
    "modes:affine": 5,
}


def payload_code(payload: "NodePayload") -> int:
    """Code identifying ``payload`` well enough to rebuild it."""
    if isinstance(payload, ModeCovariancePayload):
        key = f"modes:{payload.mode_set}"
    elif isinstance(payload, AnisotropicPayload):
        key = "anisotropic"
    else:
        key = "isotropic"
    if key not in PAYLOAD_CODES:
        raise ValueError(
            f"Payload {key!r} has no code in PAYLOAD_CODES, so a field carrying it "
            "cannot be saved and restored. Add one (appending, never renumbering)."
        )
    return PAYLOAD_CODES[key]


def payload_from_code(code: int, epsilon: float = 1e-3) -> "NodePayload":
    """Rebuild the payload a saved ``code`` names."""
    names = {v: k for k, v in PAYLOAD_CODES.items()}
    key = names.get(int(code))
    if key is None:
        raise ValueError(
            f"Unknown payload code {code!r}. It was written by a newer TorchRef than "
            f"this one, which knows {sorted(PAYLOAD_CODES)}."
        )
    if key == "isotropic":
        return IsotropicPayload()
    if key == "anisotropic":
        return AnisotropicPayload(epsilon=epsilon)
    return ModeCovariancePayload(key.split(":", 1)[1], epsilon=epsilon)


class DisorderFieldTensor(MixedTensor):
    """Per-atom ADPs from a small set of nodes, each atom a weighted mean of its k nearest.

    Storage is ``(K, 2)``: ``[log B, log sigma]`` per node. ``forward()`` returns
    ``(n_atoms,)`` isotropic B, so this drops into the ``model.adp`` slot and every
    consumer of ``adp()`` keeps working unchanged.

    The weight of node ``j`` at atom ``i`` is ``softmax_j(-d_ij^2 / 2 sigma_j^2)`` over
    that atom's candidate list, so weights are non-negative and sum to one and B is a
    convex combination of positive node values --- positive for free, with no clamping.

    Coordinates come from an accessor injected at construction rather than being passed
    per call, which keeps ``forward()`` argument-free. That makes the inherited forward
    cache incorrect on its own, since :class:`~torchref.utils.caching.CachedForwardMixin`
    fingerprints parameters, buffers and call *arguments* --- and a borrowed accessor's
    output is none of those. :meth:`_fingerprint_state` closes that by folding the
    accessor's output into the key.

    Parameters
    ----------
    initial_values : torch.Tensor, optional
        ``(n_atoms,)`` isotropic B to fit the field to. Omit for an empty shell ready
        for ``load_state_dict``.
    xyz_fn : callable, optional
        Returns the current ``(n_atoms, 3)`` coordinates. Typically ``model.xyz``. Held
        by reference and deliberately invisible to ``state_dict``, device traversal and
        ``copy``; re-attach with :meth:`set_xyz_fn` after a state-dict load.
    n_nodes : int, optional
        Number of nodes. Default 32.
    k_neighbors : int, optional
        Candidate nodes per atom. Default 12. Doubles as the skin margin that makes a
        slightly stale candidate list harmless, so prefer generous over tight.
    anchor_rows : tuple of torch.Tensor, optional
        ``(flat atom indices, node index per entry)`` defining each node's anchor
        neighbourhood. Omit to anchor every node at a single atom, which is what a model
        without a topology gets.
    node_values : torch.Tensor, optional
        ``(K, 2)`` storage to adopt directly instead of fitting to ``initial_values``.
        Used by :meth:`copy`.
    refinable_mask : torch.Tensor, optional
        Boolean mask. Interpreted in ATOM space unless ``mask_in_node_space``.
    mask_in_node_space : bool, optional
        Treat ``refinable_mask`` as already collapsed to ``(K,)``. Default False.
    requires_grad : bool, optional
        Whether node parameters carry gradients. Default True.
    dtype, device : optional
        Floating dtype and device.
    name : str, optional
        Wrapper name. Defaults to ``"adp"`` so ``Model`` consumers find it.
    epsilon : float, optional
        Floor on ``sigma`` and on fitted node B, in the same units as each. Default 1e-3.
    """

    def __init__(
        self,
        initial_values: Optional[torch.Tensor] = None,
        xyz_fn: Optional[Callable[[], torch.Tensor]] = None,
        n_nodes: int = 32,
        k_neighbors: int = 12,
        refine_positions: bool = False,
        payload: Optional["NodePayload"] = None,
        anchor_rows: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        node_values: Optional[torch.Tensor] = None,
        refinable_mask: Optional[torch.Tensor] = None,
        mask_in_node_space: bool = False,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = "adp",
        epsilon: float = 1e-3,
    ):
        self.epsilon = epsilon
        self._k_neighbors = int(k_neighbors)
        self._refine_positions = bool(refine_positions)
        # Storage columns are [payload | log sigma | offset]. Payload first keeps the
        # isotropic layout unchanged, so an existing state dict still loads.
        self._payload = payload if payload is not None else IsotropicPayload()
        object.__setattr__(self, "_xyz_fn", _wrap_accessor(xyz_fn))

        if initial_values is None and node_values is None:
            device = normalize_device(device)
            dtype = dtype if dtype is not None else get_float_dtype()
            super().__init__(None, None, requires_grad, dtype, device, name)
            self._full_shape = 0
            self.register_buffer("neighbor_list", None)
            self.register_buffer("anchor_atom", None)
            self.register_buffer("anchor_node", None)
            # device=device so the empty path lands on the requested device,
            # matching the populated path below (it used to omit it and land on CPU).
            self.register_buffer(
                "payload_code",
                torch.tensor(
                    payload_code(self._payload),
                    dtype=get_int_dtype(),
                    device=device,
                ),
            )
            return

        if xyz_fn is None:
            raise ValueError(
                "DisorderFieldTensor needs xyz_fn to place its nodes; pass the model's "
                "coordinate wrapper (e.g. model.xyz)."
            )

        xyz = xyz_fn().detach()
        if dtype is None:
            dtype = initial_values.dtype if initial_values is not None else xyz.dtype
        if device is None:
            device = xyz.device
        xyz = xyz.to(dtype=dtype, device=device)

        n_atoms = int(xyz.shape[0])

        if anchor_rows is None:
            anchor_atom = farthest_point_anchors(xyz, n_nodes)
            anchor_node = torch.arange(
                anchor_atom.shape[0], dtype=torch.int64, device=device  # dtype-ok: arange anchor indices; index requires int64
            )
        else:
            anchor_atom, anchor_node = anchor_rows
            anchor_atom = anchor_atom.to(device=device, dtype=torch.int64)  # dtype-ok: anchor_atom indices cast; index requires int64
            anchor_node = anchor_node.to(device=device, dtype=torch.int64)  # dtype-ok: anchor_node indices cast; index requires int64

        n_k = int(anchor_node.max()) + 1
        node_pos = self._segment_mean(xyz, anchor_atom, anchor_node, n_k)
        neighbor_list = build_neighbor_list(xyz, node_pos, self._k_neighbors)

        if node_values is None:
            node_values = self._fit_nodes(
                initial_values.to(dtype=dtype, device=device),
                xyz,
                node_pos,
                neighbor_list,
            )
        node_values = node_values.to(dtype=dtype, device=device)

        if refinable_mask is None:
            node_mask = torch.ones(n_k, dtype=torch.bool, device=device)
        elif mask_in_node_space:
            node_mask = refinable_mask.to(device=device, dtype=torch.bool)
        else:
            node_mask = self._collapse_mask(
                refinable_mask.to(device=device, dtype=torch.bool), neighbor_list, n_k
            )

        # ``register_buffer`` needs ``nn.Module.__init__`` to have run, which happens
        # inside this call, so every buffer below is registered after it.
        super().__init__(
            initial_values=node_values,
            refinable_mask=node_mask,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name,
        )
        self._full_shape = n_atoms
        self.register_buffer("anchor_atom", anchor_atom)
        self.register_buffer("anchor_node", anchor_node)
        self.register_buffer("neighbor_list", neighbor_list)
        # Which payload this field carries, so a restore rebuilds it rather than
        # inferring it from the storage width.
        self.register_buffer(
            "payload_code",
            torch.tensor(
                payload_code(self._payload), dtype=get_int_dtype(), device=device
            ),
        )

    # ------------------------------------------------------------------
    # Construction helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_mean(
        xyz: torch.Tensor,
        anchor_atom: torch.Tensor,
        anchor_node: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Mean coordinate of each node's anchor atoms, ``(K, 3)``.

        Differentiable in ``xyz``, which is what makes a node move with the model.
        """
        acc = xyz.new_zeros(n_nodes, 3)
        acc = acc.index_add(0, anchor_node, xyz[anchor_atom])
        counts = torch.zeros(n_nodes, dtype=xyz.dtype, device=xyz.device)
        counts = counts.index_add(
            0, anchor_node, torch.ones_like(anchor_node, dtype=xyz.dtype)
        )
        return acc / counts.clamp(min=1.0).unsqueeze(-1)

    @staticmethod
    def _weights(
        xyz: torch.Tensor,
        node_pos: torch.Tensor,
        neighbor_list: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax weights over each atom's candidate nodes, ``(n_atoms, k)``.

        Normalised across the candidates, so rows sum to one and no atom can end up
        without support even when every node is far away.
        """
        cand = node_pos[neighbor_list]
        d2 = ((xyz.unsqueeze(1) - cand) ** 2).sum(-1)
        sigma2 = torch.exp(2.0 * log_sigma)[neighbor_list]
        return torch.softmax(-d2 / (2.0 * sigma2), dim=1)

    def _fit_nodes(
        self,
        target: torch.Tensor,
        xyz: torch.Tensor,
        node_pos: torch.Tensor,
        neighbor_list: torch.Tensor,
    ) -> torch.Tensor:
        """Node storage whose field reproduces ``target`` as closely as it can.

        Kernel width is seeded at half the median nearest-neighbour node spacing, which
        makes it a property of the node layout rather than a tuned constant. With the
        weights fixed at that seed the payload is linear in the target, so the payload
        half is a closed-form solve rather than an optimisation loop -- delegated,
        because what "linear in the target" means differs between a scalar B and a
        six-component U.

        Returns
        -------
        torch.Tensor
            ``(K, payload.width + 1 + 3*refine_positions)`` node storage.
        """
        n_k = int(node_pos.shape[0])
        if n_k > 1:
            dnode = torch.cdist(node_pos, node_pos)
            dnode.fill_diagonal_(float("inf"))
            spacing = float(dnode.min(dim=1).values.median())
        else:
            spacing = float(xyz.std()) * 2.0
        sigma0 = max(spacing / 2.0, 10.0 * self.epsilon)
        log_sigma = torch.full(
            (n_k,), math.log(sigma0), dtype=xyz.dtype, device=xyz.device
        )

        w_sparse = self._weights(xyz, node_pos, neighbor_list, log_sigma)
        w_dense = torch.zeros(xyz.shape[0], n_k, dtype=xyz.dtype, device=xyz.device)
        w_dense.scatter_(1, neighbor_list, w_sparse)

        payload = self._payload.fit(
            target, w_dense, self.epsilon, xyz, node_pos, neighbor_list
        )

        columns = [payload, log_sigma.unsqueeze(-1)]
        if self._refine_positions:
            columns.append(torch.zeros_like(node_pos))
        return torch.cat(columns, dim=1)

    @staticmethod
    def _collapse_mask(
        atom_mask: torch.Tensor, neighbor_list: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        """Atom-space mask to node space: a node is refinable if any atom it serves is."""
        acc = torch.zeros(n_nodes, dtype=torch.bool, device=atom_mask.device)
        served = neighbor_list[atom_mask]
        if served.numel():
            acc[served.reshape(-1)] = True
        return acc

    # ------------------------------------------------------------------
    # Public surface.
    # ------------------------------------------------------------------

    @property
    def payload(self) -> "NodePayload":
        """What each node carries and how it becomes a per-atom ADP."""
        return self._payload

    @property
    def out_width(self) -> int:
        """Components of the per-atom output: 1 for isotropic B, 6 for a U tensor."""
        return self._payload.out_width

    def _split(self, raw):
        """Storage columns as ``(payload, log sigma, offset or None)``."""
        w = self._payload.width
        offset = raw[:, w + 1 : w + 4] if self._refine_positions else None
        return raw[:, :w], raw[:, w], offset

    def log_magnitude(self, raw=None) -> torch.Tensor:
        """``(K,)`` log ADP magnitude per node, whatever the payload.

        Lets a magnitude restraint price node values without knowing the layout.
        """
        if raw is None:
            raw = super().forward()
        return self._payload.log_magnitude(self._split(raw)[0])

    @property
    def shape(self):
        """Shape of the FULL per-atom tensor, not the node storage."""
        return (self._full_shape,)

    @property
    def node_shape(self):
        """Shape of the node storage."""
        return tuple(self.fixed_values.shape)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.fixed_values.shape[0])

    def set_xyz_fn(self, xyz_fn: Callable[[], torch.Tensor]) -> None:
        """Attach the coordinate accessor.

        Needed after ``load_state_dict``, which cannot carry a callable.
        """
        object.__setattr__(self, "_xyz_fn", _wrap_accessor(xyz_fn))

    @property
    def refines_positions(self) -> bool:
        """Whether node positions carry a refinable offset."""
        return self._refine_positions

    def node_positions(self, xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Node positions, ``(K, 3)``: anchored centroid plus any refinable offset."""
        if xyz is None:
            xyz = self._xyz_fn()
        return self._node_positions_from(xyz, super().forward())

    def _node_positions_from(self, xyz, raw):
        """Anchor centroid, displaced by the refinable offset when there is one.

        Keeping the centroid as the base rather than storing absolute coordinates means
        a node still travels with the model under xyz refinement; the offset only says
        where it sits *relative* to the atoms it belongs to.
        """
        base = self._segment_mean(
            xyz, self.anchor_atom, self.anchor_node, raw.shape[0]
        )
        offset = self._split(raw)[2]
        return base if offset is None else base + offset

    def weights(self, xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-atom weights over candidate nodes, ``(n_atoms, k)``. Rows sum to one."""
        if xyz is None:
            xyz = self._xyz_fn()
        raw = super().forward()
        return self._weights(
            xyz,
            self._node_positions_from(xyz, raw),
            self.neighbor_list,
            self._split(raw)[1],
        )

    def node_load(self, xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Total weight each node carries across all atoms, ``(K,)``.

        Not obtainable by summing :meth:`weights` over atoms: that returns
        ``(n_atoms, k)`` over each atom's CANDIDATES, so a column is a candidate *slot*
        shared by different nodes for different atoms, and summing it gives a length-k
        vector with no meaning. The weights have to be scattered back into node space
        through ``neighbor_list`` first, which is what this does.

        Load is what identifies a node that has stopped doing useful work: a node that
        narrows onto a handful of atoms carries almost none.
        """
        W = self.weights(xyz)
        load = W.new_zeros(self.n_nodes)
        return load.index_add(0, self.neighbor_list.reshape(-1), W.reshape(-1))

    def smallest_candidate_weight(self, xyz: Optional[torch.Tensor] = None) -> float:
        """Largest per-atom minimum candidate weight --- the list-adequacy invariant.

        Each atom sees only its ``k`` candidate nodes. While the weakest of those
        candidates carries negligible weight, the list brackets the real neighbourhood
        and a node drifting in or out of it cannot move any ADP appreciably. When this
        rises, the list no longer brackets it and should be rebuilt with
        :meth:`rebuild_neighbor_list` or a larger ``k``.
        """
        with torch.no_grad():
            return float(self.weights(xyz).min(dim=1).values.max())

    def rebuild_neighbor_list(
        self, xyz: Optional[torch.Tensor] = None, k_neighbors: Optional[int] = None
    ) -> None:
        """Recompute which nodes each atom sees, at the current coordinates.

        The candidate list is the slowly-varying combinatorial half of the field and is
        never refreshed implicitly: membership is piecewise constant in position, so a
        rebuild is a discrete jump and belongs at a point the caller chooses.
        """
        if xyz is None:
            xyz = self._xyz_fn()
        xyz = xyz.detach()
        if k_neighbors is not None:
            self._k_neighbors = int(k_neighbors)
        self.neighbor_list = build_neighbor_list(
            xyz, self.node_positions(xyz).detach(), self._k_neighbors
        )
        self.reset_forward_cache()

    def evaluate(self, xyz: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Per-atom B from explicit coordinates and node storage, ``(n_atoms,)``.

        The field's arithmetic, with no accessor and no cache in the way, so it can be
        differentiated and checked on its own. ``forward()`` is this plus the plumbing
        that fetches both arguments.

        Parameters
        ----------
        xyz : torch.Tensor
            ``(n_atoms, 3)`` coordinates.
        raw : torch.Tensor
            ``(K, 2)`` node storage, ``[log b, log sigma]``.
        """
        payload, log_sigma, _ = self._split(raw)
        node_pos = self._node_positions_from(xyz, raw)
        W = self._weights(xyz, node_pos, self.neighbor_list, log_sigma)
        contrib = self._payload.contributions(
            payload, xyz, node_pos, self.neighbor_list
        )
        out = (W.unsqueeze(-1) * contrib).sum(dim=1)
        # A scalar payload reports per-atom B as (N,), not (N, 1): that is the shape
        # every consumer of ``adp()`` expects.
        return out.squeeze(-1) if self._payload.out_width == 1 else out

    def forward(self) -> torch.Tensor:
        """Per-atom isotropic B, ``(n_atoms,)``.

        A convex combination of positive node values, so strictly positive without a
        clamp. Translation-invariant by construction: node positions are centroids of
        atom coordinates, so a rigid shift of the model moves the nodes with it and
        leaves every distance, and therefore every weight, unchanged.
        """
        return self.evaluate(self._xyz_fn(), super().forward())

    def node_values(self) -> torch.Tensor:
        """The assembled node storage ``(K, 2)`` in raw ``[log b, log sigma]`` space."""
        return super().forward()

    def _fingerprint_state(self):
        """Fold the accessor's coordinates into the forward-cache key.

        Without this the cache would be keyed on parameters and buffers alone and would
        serve a per-atom B computed at coordinates that have since moved --- the
        coordinates reach ``forward()`` through the accessor, not through an argument,
        so the mixin cannot see them by itself.
        """
        base = super()._fingerprint_state()
        if self._xyz_fn is None:
            return base
        xyz = self._xyz_fn()
        return base + ((xyz.data_ptr(), xyz._version),)

    def _set_values(self, key, value: torch.Tensor) -> None:
        """Rejected: a node field cannot represent arbitrary per-atom values.

        Assigning per-atom ADPs would silently be a projection onto the field rather
        than the write the caller asked for. Use :meth:`refit` to move the field toward
        a per-atom target, or switch the model back to a per-atom representation.
        """
        raise NotImplementedError(
            "DisorderFieldTensor stores K nodes, not per-atom values, so per-atom "
            "assignment is not representable. Use refit() to fit the field to a "
            "per-atom target, or Model.set_adp_mode('isotropic') to leave field mode."
        )

    def refit(self, target_b: torch.Tensor) -> None:
        """Re-fit the node values to a per-atom target, in place.

        The target is per-atom B for a scalar payload, or per-atom U6 for a tensor one;
        an anisotropic payload also accepts a B target and lifts it to ``U_iso * I``.

        Replaces ``refinable_params``, so any optimizer state held for it is stale.
        """
        xyz = self._xyz_fn().detach()
        target_b = target_b.to(dtype=self.dtype, device=self.device)
        node_values = self._fit_nodes(
            target_b, xyz, self.node_positions(xyz).detach(), self.neighbor_list
        )
        self.fixed_values = node_values.clone().detach()
        refinable = node_values[self.refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(
            refinable, requires_grad=self.refinable_params.requires_grad
        )
        self._build_index_cache()
        self.reset_forward_cache()

    def update_refinable_mask(
        self, new_mask: torch.Tensor, in_node_space: bool = False
    ):
        """Repartition refinable/fixed nodes, keeping the raw node values.

        Parameters
        ----------
        new_mask : torch.Tensor
            Boolean mask, ``(n_atoms,)`` in atom space or ``(K,)`` when
            ``in_node_space``. An atom-space mask collapses with OR: a node is refinable
            if any atom it serves is.
        in_node_space : bool, optional
            Whether ``new_mask`` is already in node space. Default False.
        """
        new_mask = new_mask.to(device=self.device, dtype=torch.bool)
        if in_node_space:
            if new_mask.shape[0] != self.n_nodes:
                raise ValueError(
                    f"Node-space mask must have shape ({self.n_nodes},), "
                    f"got {tuple(new_mask.shape)}"
                )
            node_mask = new_mask
        else:
            if new_mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"Atom-space mask must have shape ({self._full_shape},), "
                    f"got {tuple(new_mask.shape)}"
                )
            node_mask = self._collapse_mask(
                new_mask, self.neighbor_list, self.n_nodes
            )

        current = self.fixed_values.clone()
        if self.refinable_mask is not None and bool(self.refinable_mask.any()):
            current[self.refinable_mask] = self.refinable_params.data

        self.fixed_values = current.clone().detach()
        if bool(node_mask.any()):
            self.refinable_params = nn.Parameter(
                current[node_mask].clone().detach(),
                requires_grad=self.refinable_params.requires_grad,
            )
        else:
            self.refinable_params = nn.Parameter(
                torch.empty(0, 2, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )
        self.refinable_mask = node_mask
        self.fixed_mask = ~node_mask
        self._build_index_cache()
        self.reset_forward_cache()

    def copy(self) -> "DisorderFieldTensor":
        """Independent copy sharing no parameter storage.

        The coordinate accessor is carried by REFERENCE, never deep-copied: it holds a
        cache that can contain a graph-attached tensor, which ``deepcopy`` refuses to
        walk.
        """
        accessor = self._xyz_fn
        if isinstance(accessor, ModuleReference):
            accessor = accessor.module
        new = DisorderFieldTensor(
            initial_values=None,
            xyz_fn=accessor,
            k_neighbors=self._k_neighbors,
            refine_positions=self._refine_positions,
            payload=self._payload,
            anchor_rows=(self.anchor_atom.clone(), self.anchor_node.clone()),
            node_values=self.node_values().detach().clone(),
            refinable_mask=self.refinable_mask.clone(),
            mask_in_node_space=True,
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self._name,
            epsilon=self.epsilon,
        )
        new.neighbor_list = self.neighbor_list.clone()
        new._full_shape = self._full_shape
        return new

    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        return (
            f"DisorderFieldTensor({name_str}atoms={self._full_shape}, "
            f"nodes={self.n_nodes}, k={self._k_neighbors}, "
            f"payload={type(self._payload).__name__}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()})"
        )
