"""
Core chain closure mathematics for junction residues.

Uses the standard NeRF (Natural Extension Reference Frame) formula for
backbone forward kinematics. Slave DOFs are solved by Newton's method
with IFT-based backward gradients.

NeRF torsion assignments for placing backbone atoms:
- Place N(i):  torsion = psi(i-1)  = dihedral(N(i-1), CA(i-1), C(i-1), N(i))
- Place CA(i): torsion = omega(i)  = dihedral(CA(i-1), C(i-1), N(i), CA(i))
- Place C(i):  torsion = phi(i)    = dihedral(C(i-1), N(i), CA(i), C(i))

For K junction residues, the slave DOFs are phi(0..K-1) and psi(0..K-1)
(2K total). omega values are fixed. psi(-1) (= psi of the last pre-junction
residue) is also fixed. The residual matches the computed end-of-junction
position against the known post-junction position.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _nerf_place(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    bond_length: torch.Tensor,
    bond_angle: torch.Tensor,
    torsion: torch.Tensor,
) -> torch.Tensor:
    """
    Place an atom using the NeRF formula.

    new_pos = p3 + dx*bc + dy*m - dz*n
    where bc = (p3-p2)/|p3-p2|, n = cross(p2-p1, bc)/|...|, m = cross(n, bc)

    Parameters
    ----------
    p1, p2, p3 : torch.Tensor
        Three ancestor positions, shape (J, 3).
    bond_length : torch.Tensor
        |new - p3|, shape (J,).
    bond_angle : torch.Tensor
        Angle at p3 between p2-p3-new, shape (J,).
    torsion : torch.Tensor
        Dihedral p1-p2-p3-new, shape (J,).

    Returns
    -------
    torch.Tensor
        New atom position, shape (J, 3).
    """
    bc = p3 - p2
    bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)

    ab = p2 - p1
    n = torch.linalg.cross(ab, bc)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-10)

    m = torch.linalg.cross(n, bc)

    theta = torch.pi - bond_angle
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    dx = bond_length * cos_t
    dy = bond_length * sin_t * torch.cos(torsion)
    dz = bond_length * sin_t * torch.sin(torsion)

    return (
        p3
        + dx.unsqueeze(-1) * bc
        + dy.unsqueeze(-1) * m
        - dz.unsqueeze(-1) * n
    )


def backbone_fk_junction(
    p1_start: torch.Tensor,
    p2_start: torch.Tensor,
    p3_start: torch.Tensor,
    phi_psi: torch.Tensor,
    nerf_bond_lengths: torch.Tensor,
    nerf_bond_angles: torch.Tensor,
    omega: torch.Tensor,
    psi_prev: torch.Tensor,
    post_bond_length: Optional[torch.Tensor] = None,
    post_bond_angle: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward kinematics through junction backbone using NeRF.

    Optionally extends the FK by one atom (N_post) using psi(K-1), which
    enables the closure to target the post-junction N position dynamically.

    Parameters
    ----------
    p1_start, p2_start, p3_start : torch.Tensor
        N, CA, C of last pre-junction residue, each shape (3,) or (J, 3).
    phi_psi : torch.Tensor
        Slave DOFs: [phi_0, psi_0, phi_1, psi_1, ...],
        shape (2*K,) or (J, 2*K) where K = junction_size.
    nerf_bond_lengths : torch.Tensor
        Bond lengths in NeRF order: [C_prev-N_0, N_0-CA_0, CA_0-C_0, ...],
        shape (3*K,) or (J, 3*K).
    nerf_bond_angles : torch.Tensor
        Bond angles at pivot atoms in NeRF order:
        [angle_at_C_prev_for_N, angle_at_N_for_CA, angle_at_CA_for_C, ...],
        shape (3*K,) or (J, 3*K).
    omega : torch.Tensor
        Fixed omega angles, shape (K,) or (J, K).
    psi_prev : torch.Tensor
        Fixed psi of pre-junction residue (used to place first N),
        shape () or (J,).
    post_bond_length : torch.Tensor, optional
        C(K-1) -> N_post bond length, shape (J,). If provided together with
        post_bond_angle, the FK extends one more atom using psi(K-1).
    post_bond_angle : torch.Tensor, optional
        Angle at C(K-1) for N_post placement, shape (J,).

    Returns
    -------
    end_p1, end_p2, end_p3 : torch.Tensor
        N, CA, C of last junction residue.
    backbone_xyz : torch.Tensor
        All junction backbone positions, shape (3*K, 3) or (J, 3*K, 3).
    end_point : torch.Tensor
        Closure target point: N_post if post params given, else C(K-1).
        Shape (3,) or (J, 3).
    """
    batched = p1_start.dim() == 2
    if not batched:
        p1_start = p1_start.unsqueeze(0)
        p2_start = p2_start.unsqueeze(0)
        p3_start = p3_start.unsqueeze(0)
        phi_psi = phi_psi.unsqueeze(0)
        nerf_bond_lengths = nerf_bond_lengths.unsqueeze(0)
        nerf_bond_angles = nerf_bond_angles.unsqueeze(0)
        omega = omega.unsqueeze(0)
        psi_prev = psi_prev.unsqueeze(0)
        if post_bond_length is not None:
            post_bond_length = post_bond_length.unsqueeze(0)
        if post_bond_angle is not None:
            post_bond_angle = post_bond_angle.unsqueeze(0)

    n_junction = omega.shape[1]
    backbone_positions = []

    # Trailing 3 atoms
    pp1, pp2, pp3 = p1_start, p2_start, p3_start

    for i in range(n_junction):
        phi_i = phi_psi[:, 2 * i]
        psi_i = phi_psi[:, 2 * i + 1]
        omg_i = omega[:, i]

        bl_N  = nerf_bond_lengths[:, 3 * i]
        bl_CA = nerf_bond_lengths[:, 3 * i + 1]
        bl_C  = nerf_bond_lengths[:, 3 * i + 2]

        ba_N  = nerf_bond_angles[:, 3 * i]      # angle at C_prev for N placement
        ba_CA = nerf_bond_angles[:, 3 * i + 1]   # angle at N for CA placement
        ba_C  = nerf_bond_angles[:, 3 * i + 2]   # angle at CA for C placement

        # Torsion for placing N(i): psi(i-1)
        if i == 0:
            torsion_N = psi_prev
        else:
            torsion_N = phi_psi[:, 2 * (i - 1) + 1]  # psi(i-1)

        # Place N(i) from (pp1, pp2, pp3) using torsion=psi(i-1)
        N_pos = _nerf_place(pp1, pp2, pp3, bl_N, ba_N, torsion_N)
        backbone_positions.append(N_pos)

        # Place CA(i) from (pp2, pp3, N) using torsion=omega(i)
        CA_pos = _nerf_place(pp2, pp3, N_pos, bl_CA, ba_CA, omg_i)
        backbone_positions.append(CA_pos)

        # Place C(i) from (pp3, N, CA) using torsion=phi(i)
        C_pos = _nerf_place(pp3, N_pos, CA_pos, bl_C, ba_C, phi_i)
        backbone_positions.append(C_pos)

        pp1, pp2, pp3 = N_pos, CA_pos, C_pos

    backbone_xyz = torch.stack(backbone_positions, dim=1)

    # Extend FK: place N_post using psi(K-1) for dynamic closure targeting.
    # This makes psi(K-1) an active DOF (it didn't affect C(K-1) before).
    if post_bond_length is not None and post_bond_angle is not None:
        torsion_N_post = phi_psi[:, 2 * (n_junction - 1) + 1]  # psi(K-1)
        end_point = _nerf_place(
            pp1, pp2, pp3, post_bond_length, post_bond_angle, torsion_N_post,
        )
    else:
        end_point = pp3  # C(K-1)

    if not batched:
        pp1 = pp1.squeeze(0)
        pp2 = pp2.squeeze(0)
        pp3 = pp3.squeeze(0)
        backbone_xyz = backbone_xyz.squeeze(0)
        end_point = end_point.squeeze(0)

    return pp1, pp2, pp3, backbone_xyz, end_point


def closure_residual(
    end_p3: torch.Tensor,
    target_p3: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 3D position closure residual.

    Parameters
    ----------
    end_p3 : torch.Tensor
        Computed C position of last junction residue, shape (3,) or (J, 3).
    target_p3 : torch.Tensor
        Target C position, shape (3,) or (J, 3).

    Returns
    -------
    torch.Tensor
        3D position residual, shape (3,) or (J, 3).
    """
    return end_p3 - target_p3


class JunctionClosure(torch.autograd.Function):
    """
    Custom autograd for chain closure using the Implicit Function Theorem.

    Forward: Newton solve for junction phi/psi.
    Backward: IFT adjoint for exact gradients.
    """

    @staticmethod
    def forward(
        ctx,
        phi_psi_init: torch.Tensor,
        p1_start: torch.Tensor,
        p2_start: torch.Tensor,
        p3_start: torch.Tensor,
        target_p3: torch.Tensor,
        nerf_bond_lengths: torch.Tensor,
        nerf_bond_angles: torch.Tensor,
        omega: torch.Tensor,
        psi_prev: torch.Tensor,
        post_bond_length: torch.Tensor,
        post_bond_angle: torch.Tensor,
        max_iter: int = 20,
        tol: float = 1e-4,
        tikhonov_eps: float = 1e-6,
    ) -> torch.Tensor:
        phi_psi = phi_psi_init.detach().clone()
        n_dof = phi_psi.shape[-1]
        n_junc = phi_psi.shape[0]
        pbl_det = post_bond_length.detach() if post_bond_length is not None else None
        pba_det = post_bond_angle.detach() if post_bond_angle is not None else None

        # Track per-junction best results
        best_phi_psi = phi_psi.clone()
        best_res_per_junc = torch.full((n_junc,), float("inf"),
                                       device=phi_psi.device, dtype=phi_psi.dtype)

        for iteration in range(max_iter):
            with torch.enable_grad():
                phi_psi_var = phi_psi.detach().requires_grad_(True)

                _, _, _, _, end_point = backbone_fk_junction(
                    p1_start.detach(), p2_start.detach(), p3_start.detach(),
                    phi_psi_var,
                    nerf_bond_lengths.detach(), nerf_bond_angles.detach(),
                    omega.detach(), psi_prev.detach(),
                    pbl_det, pba_det,
                )
                res = closure_residual(end_point, target_p3.detach())  # (J, 3)
                per_junc_norm = torch.linalg.norm(res, dim=-1)  # (J,)

                # Update per-junction best
                improved = per_junc_norm < best_res_per_junc
                if improved.any():
                    best_phi_psi[improved] = phi_psi[improved].clone()
                    best_res_per_junc[improved] = per_junc_norm[improved].clone()

                res_norm = per_junc_norm.max().item()
                if res_norm < tol:
                    break

                # Jacobian dF/d(phi_psi): (J, 3, n_dof)
                jacobian = torch.zeros(
                    n_junc, 3, n_dof,
                    device=phi_psi.device, dtype=phi_psi.dtype,
                )
                for k in range(3):
                    grad_out = torch.zeros_like(res)
                    grad_out[:, k] = 1.0
                    g = torch.autograd.grad(
                        res, phi_psi_var, grad_outputs=grad_out,
                        retain_graph=True, create_graph=False,
                    )[0]
                    jacobian[:, k, :] = g

            # Gauss-Newton step: (J^T J + eps I) dx = J^T F
            JtJ = torch.bmm(jacobian.detach().transpose(-1, -2), jacobian.detach())
            eye = torch.eye(n_dof, device=JtJ.device, dtype=JtJ.dtype).unsqueeze(0)
            JtJ_reg = JtJ + tikhonov_eps * eye
            Jt_res = torch.bmm(
                jacobian.detach().transpose(-1, -2),
                res.detach().unsqueeze(-1),
            ).squeeze(-1)
            try:
                delta = torch.linalg.solve(JtJ_reg, Jt_res)
            except torch._C._LinAlgError:
                # Fallback: pseudoinverse via lstsq
                delta = torch.linalg.lstsq(JtJ_reg, Jt_res.unsqueeze(-1)).solution.squeeze(-1)

            # Backtracking line search per junction
            alpha = torch.ones(n_junc, 1, device=phi_psi.device, dtype=phi_psi.dtype)
            for _ in range(6):
                phi_psi_trial = phi_psi - alpha * delta
                phi_psi_trial = torch.atan2(
                    torch.sin(phi_psi_trial), torch.cos(phi_psi_trial)
                )
                _, _, _, _, ep_trial = backbone_fk_junction(
                    p1_start.detach(), p2_start.detach(), p3_start.detach(),
                    phi_psi_trial,
                    nerf_bond_lengths.detach(), nerf_bond_angles.detach(),
                    omega.detach(), psi_prev.detach(),
                    pbl_det, pba_det,
                )
                trial_norms = torch.linalg.norm(
                    ep_trial - target_p3.detach(), dim=-1
                )
                # Halve alpha for junctions that didn't improve
                no_improve = trial_norms >= per_junc_norm.detach()
                if no_improve.any():
                    alpha[no_improve] *= 0.5
                if (~no_improve).all():
                    break
            phi_psi = phi_psi_trial

            # Restore best for converged junctions (don't disturb them)
            converged = best_res_per_junc < tol
            if converged.any():
                phi_psi[converged] = best_phi_psi[converged]

        # Random restarts for junctions that haven't converged
        stuck = best_res_per_junc > tol
        if stuck.any():
            n_restarts = 5
            for restart in range(n_restarts):
                # Random initial phi_psi for stuck junctions
                phi_psi_rand = phi_psi.clone()
                phi_psi_rand[stuck] = (
                    torch.randn_like(phi_psi_rand[stuck]) * 0.5
                )
                # Run a shorter solve
                for _ in range(max_iter // 2):
                    with torch.enable_grad():
                        pv = phi_psi_rand.detach().requires_grad_(True)
                        _, _, _, _, ep = backbone_fk_junction(
                            p1_start.detach(), p2_start.detach(),
                            p3_start.detach(), pv,
                            nerf_bond_lengths.detach(),
                            nerf_bond_angles.detach(),
                            omega.detach(), psi_prev.detach(),
                            pbl_det, pba_det,
                        )
                        r = closure_residual(ep, target_p3.detach())
                        rn = torch.linalg.norm(r, dim=-1)

                        improved_r = rn < best_res_per_junc
                        if improved_r.any():
                            best_phi_psi[improved_r] = phi_psi_rand[improved_r].clone()
                            best_res_per_junc[improved_r] = rn[improved_r].clone()

                        if rn.max().item() < tol:
                            break

                        jac = torch.zeros(n_junc, 3, n_dof,
                                          device=pv.device, dtype=pv.dtype)
                        for k in range(3):
                            go = torch.zeros_like(r)
                            go[:, k] = 1.0
                            g = torch.autograd.grad(r, pv, grad_outputs=go,
                                                    retain_graph=True,
                                                    create_graph=False)[0]
                            jac[:, k, :] = g

                    JtJ_r = torch.bmm(jac.detach().transpose(-1, -2), jac.detach())
                    JtJ_r = JtJ_r + tikhonov_eps * eye
                    Jt_r = torch.bmm(jac.detach().transpose(-1, -2),
                                     r.detach().unsqueeze(-1)).squeeze(-1)
                    try:
                        d = torch.linalg.solve(JtJ_r, Jt_r)
                    except torch._C._LinAlgError:
                        d = torch.linalg.lstsq(JtJ_r, Jt_r.unsqueeze(-1)).solution.squeeze(-1)
                    phi_psi_rand = torch.atan2(
                        torch.sin(phi_psi_rand - d),
                        torch.cos(phi_psi_rand - d),
                    )

                # Update stuck mask
                stuck = best_res_per_junc > tol
                if not stuck.any():
                    break

        # Use the best results
        phi_psi = best_phi_psi
        res_norm = best_res_per_junc.max().item()

        if res_norm > tol:
            logger.warning(
                f"Junction closure: residual={res_norm:.4e} after solve "
                f"({stuck.sum().item()}/{n_junc} junctions unconverged)"
            )

        ctx.has_post_params = post_bond_length is not None
        # save_for_backward can't store None — use dummy if absent
        if post_bond_length is None:
            post_bond_length = torch.zeros(phi_psi.shape[0], device=phi_psi.device, dtype=phi_psi.dtype)
        if post_bond_angle is None:
            post_bond_angle = torch.zeros(phi_psi.shape[0], device=phi_psi.device, dtype=phi_psi.dtype)
        ctx.save_for_backward(
            phi_psi, p1_start, p2_start, p3_start, target_p3,
            nerf_bond_lengths, nerf_bond_angles, omega, psi_prev,
            post_bond_length, post_bond_angle,
        )
        ctx.tikhonov_eps = tikhonov_eps
        ctx.n_dof = n_dof

        return phi_psi

    @staticmethod
    def backward(ctx, grad_output):
        """
        IFT-based backward pass for the implicit closure constraint.

        Given F(phi_psi*, theta) = 0, the IFT gives:
          d(phi_psi*)/d(theta) = -[dF/d(phi_psi)]^+ @ dF/d(theta)
        where ^+ denotes the pseudoinverse.

        The VJP is:
          grad_theta = grad_output^T @ d(phi_psi*)/d(theta)
                     = -lam_3d^T @ dF/d(theta)
        where lam_3d = (J_phi @ J_phi^T)^{-1} @ J_phi @ grad_output (adjoint).
        """
        (phi_psi, p1_start, p2_start, p3_start, target_p3,
         nerf_bond_lengths, nerf_bond_angles, omega, psi_prev,
         post_bond_length, post_bond_angle) = ctx.saved_tensors
        tikhonov_eps = ctx.tikhonov_eps
        n_dof = ctx.n_dof
        has_post = ctx.has_post_params

        with torch.enable_grad():
            # Recompute residual with grad tracking on all inputs
            phi_psi_var = phi_psi.detach().requires_grad_(True)
            p1_var = p1_start.detach().requires_grad_(True)
            p2_var = p2_start.detach().requires_grad_(True)
            p3_var = p3_start.detach().requires_grad_(True)
            target_var = target_p3.detach().requires_grad_(True)
            bl_var = nerf_bond_lengths.detach().requires_grad_(True)
            ba_var = nerf_bond_angles.detach().requires_grad_(True)
            omg_var = omega.detach().requires_grad_(True)
            psi_prev_var = psi_prev.detach().requires_grad_(True)

            if has_post:
                post_bl_var = post_bond_length.detach().requires_grad_(True)
                post_ba_var = post_bond_angle.detach().requires_grad_(True)
            else:
                post_bl_var = None
                post_ba_var = None

            _, _, _, _, end_point = backbone_fk_junction(
                p1_var, p2_var, p3_var, phi_psi_var,
                bl_var, ba_var, omg_var, psi_prev_var,
                post_bl_var, post_ba_var,
            )
            res = closure_residual(end_point, target_var)  # (J, 3)

            # Jacobian dF/d(phi_psi): shape (J, 3, n_dof)
            J_batch = res.shape[0]
            J_phi = torch.zeros(J_batch, 3, n_dof, device=res.device, dtype=res.dtype)
            for k in range(3):
                grad_out = torch.zeros_like(res)
                grad_out[:, k] = 1.0
                g = torch.autograd.grad(
                    res, phi_psi_var, grad_outputs=grad_out,
                    retain_graph=True, create_graph=False,
                )[0]
                J_phi[:, k, :] = g

            # Solve for adjoint vector lam_3d: (J_phi @ J_phi^T + eps*I) lam_3d = J_phi @ grad_output
            JJt = torch.bmm(J_phi, J_phi.transpose(-1, -2))  # (J, 3, 3)
            eye3 = torch.eye(3, device=JJt.device, dtype=JJt.dtype).unsqueeze(0)
            JJt_reg = JJt + tikhonov_eps * eye3
            J_grad = torch.bmm(J_phi, grad_output.unsqueeze(-1)).squeeze(-1)  # (J, 3)
            lam_3d = torch.linalg.solve(JJt_reg, J_grad)  # (J, 3)

            # VJP: grad_theta = -d(lam_3d . res)/d(theta)
            pseudo_loss = (lam_3d.detach() * res).sum()
            theta_vars = [
                p1_var, p2_var, p3_var, target_var,
                bl_var, ba_var, omg_var, psi_prev_var,
            ]
            if has_post:
                theta_vars.extend([post_bl_var, post_ba_var])
            grads = torch.autograd.grad(
                pseudo_loss, theta_vars,
                allow_unused=True, create_graph=False,
            )

        def neg(g):
            return -g if g is not None else None

        return (
            None,  # phi_psi_init
            neg(grads[0]),  # p1_start
            neg(grads[1]),  # p2_start
            neg(grads[2]),  # p3_start
            neg(grads[3]),  # target_p3
            neg(grads[4]),  # nerf_bond_lengths
            neg(grads[5]),  # nerf_bond_angles
            neg(grads[6]),  # omega
            neg(grads[7]),  # psi_prev
            neg(grads[8]) if has_post else None,  # post_bond_length
            neg(grads[9]) if has_post else None,  # post_bond_angle
            None, None, None,  # max_iter, tol, tikhonov_eps
        )


class JunctionSolver(nn.Module):
    """
    Manages warm-start buffers and solves junction closures.

    Parameters
    ----------
    n_junctions : int
        Number of junctions.
    junction_size : int
        Number of residues per junction.
    initial_phi_psi : torch.Tensor
        Initial guess for phi/psi, shape (J, 2*K).
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        n_junctions: int,
        junction_size: int,
        initial_phi_psi: torch.Tensor,
        max_iter: int = 20,
        tol: float = 1e-4,
        tikhonov_eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_junctions = n_junctions
        self.junction_size = junction_size
        self.n_dof = 2 * junction_size
        self.max_iter = max_iter
        self.tol = tol
        self.tikhonov_eps = tikhonov_eps

        self.register_buffer(
            "warm_start",
            initial_phi_psi.detach().clone().to(dtype=dtype, device=device),
        )

    def forward(
        self,
        p1_start: torch.Tensor,
        p2_start: torch.Tensor,
        p3_start: torch.Tensor,
        target_p3: torch.Tensor,
        nerf_bond_lengths: torch.Tensor,
        nerf_bond_angles: torch.Tensor,
        omega: torch.Tensor,
        psi_prev: torch.Tensor,
        post_bond_length: Optional[torch.Tensor] = None,
        post_bond_angle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve junction closures and return backbone positions.

        Returns
        -------
        phi_psi : torch.Tensor
            Solved phi/psi, shape (J, 2*K).
        backbone_xyz : torch.Tensor
            Junction backbone positions, shape (J, 3*K, 3).
        """
        if self.n_junctions == 0:
            dev, dt = p1_start.device, p1_start.dtype
            return (
                torch.zeros(0, self.n_dof, device=dev, dtype=dt),
                torch.zeros(0, 3 * self.junction_size, 3, device=dev, dtype=dt),
            )

        phi_psi = JunctionClosure.apply(
            self.warm_start,
            p1_start, p2_start, p3_start, target_p3,
            nerf_bond_lengths, nerf_bond_angles, omega, psi_prev,
            post_bond_length, post_bond_angle,
            self.max_iter, self.tol, self.tikhonov_eps,
        )

        with torch.no_grad():
            self.warm_start.copy_(phi_psi.detach())

        _, _, _, backbone_xyz, _ = backbone_fk_junction(
            p1_start, p2_start, p3_start, phi_psi,
            nerf_bond_lengths, nerf_bond_angles, omega, psi_prev,
        )

        return phi_psi, backbone_xyz

    @property
    def closure_residuals(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_residuals", None)
