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

A node's position is *derived*, not refined: it is the centroid of the atoms within
``anchor_radius`` bonds of its anchor atom. That keeps a node inside the molecule,
confined to one connected fragment, and moving with the model, and it leaves the
optimiser no free coordinate to wander with.
"""

import math
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from torchref.config import get_float_dtype, normalize_device
from torchref.model.parameter_wrappers import MixedTensor
from torchref.utils.utils import ModuleReference

__all__ = ["DisorderFieldTensor", "farthest_point_anchors", "build_neighbor_list"]


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

    anchors = torch.tensor(chosen, dtype=torch.int64, device=xyz.device)

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
    atom_idx = torch.arange(xyz.shape[0], dtype=torch.int64, device=xyz.device)

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
        object.__setattr__(self, "_xyz_fn", _wrap_accessor(xyz_fn))

        if initial_values is None and node_values is None:
            device = normalize_device(device)
            dtype = dtype if dtype is not None else get_float_dtype()
            super().__init__(None, None, requires_grad, dtype, device, name)
            self._full_shape = 0
            self.register_buffer("neighbor_list", None)
            self.register_buffer("anchor_atom", None)
            self.register_buffer("anchor_node", None)
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
                anchor_atom.shape[0], dtype=torch.int64, device=device
            )
        else:
            anchor_atom, anchor_node = anchor_rows
            anchor_atom = anchor_atom.to(device=device, dtype=torch.int64)
            anchor_node = anchor_node.to(device=device, dtype=torch.int64)

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
        target_b: torch.Tensor,
        xyz: torch.Tensor,
        node_pos: torch.Tensor,
        neighbor_list: torch.Tensor,
    ) -> torch.Tensor:
        """Least-squares node values reproducing ``target_b`` as closely as possible.

        ``sigma`` is seeded at half the median nearest-neighbour node spacing, then the
        node values are the ridged solution of ``W b = target_b``. Linear in ``b``, so
        this is a closed-form solve rather than an optimisation loop.

        Returns
        -------
        torch.Tensor
            ``(K, 2)`` storage, ``[log b, log sigma]``.
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

        W_sparse = self._weights(xyz, node_pos, neighbor_list, log_sigma)
        W = torch.zeros(xyz.shape[0], n_k, dtype=xyz.dtype, device=xyz.device)
        W.scatter_(1, neighbor_list, W_sparse)

        # Non-finite targets are dropped from the solve rather than carried into it:
        # a single NaN row would propagate through the normal equations and take
        # every node with it. Those atoms still receive a fitted value on output.
        finite = torch.isfinite(target_b)
        if not bool(finite.all()):
            W, target_b = W[finite], target_b[finite]
            if target_b.numel() == 0:
                flat = torch.stack([torch.zeros_like(log_sigma), log_sigma], dim=1)
                if self._refine_positions:
                    flat = torch.cat([flat, torch.zeros_like(node_pos)], dim=1)
                return flat

        gram = W.T @ W
        ridge = 1e-6 * torch.diagonal(gram).mean().clamp(min=1e-30)
        eye = torch.eye(n_k, dtype=W.dtype, device=W.device)
        b = torch.linalg.solve(gram + ridge * eye, W.T @ target_b)

        b = b.clamp(min=self.epsilon)
        node_values = torch.stack([torch.log(b), log_sigma], dim=1)
        if self._refine_positions:
            node_values = torch.cat(
                [node_values, torch.zeros_like(node_pos)], dim=1
            )
        return node_values

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
        return base + raw[:, 2:5] if self._refine_positions else base

    def weights(self, xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-atom weights over candidate nodes, ``(n_atoms, k)``. Rows sum to one."""
        if xyz is None:
            xyz = self._xyz_fn()
        raw = super().forward()
        return self._weights(
            xyz, self._node_positions_from(xyz, raw), self.neighbor_list, raw[:, 1]
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
        log_b, log_sigma = raw[:, 0], raw[:, 1]
        node_pos = self._node_positions_from(xyz, raw)
        W = self._weights(xyz, node_pos, self.neighbor_list, log_sigma)
        return (W * torch.exp(log_b)[self.neighbor_list]).sum(dim=1)

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
        """Re-fit the node values to a per-atom B target, in place.

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
            f"nodes={self.n_nodes}, k={self._k_neighbors}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()})"
        )
