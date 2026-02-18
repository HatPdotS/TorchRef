"""
Tests for the chain closure submodule: backbone_utils.py and closure.py.

Tests cover:
- Backbone atom identification and torsion computation
- Secondary structure estimation
- Junction placement planning
- NeRF atom placement formula
- Backbone forward kinematics
- Closure residual computation
- Newton solver convergence
- IFT backward gradients (JunctionClosure)
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from torchref.base.chain_closure.backbone_utils import (
    AA_NAMES,
    BACKBONE_ATOMS,
    _find_best_junction_start,
    _torsion_angle,
    compute_backbone_torsions,
    estimate_secondary_structure,
    get_chain_residues,
    get_junction_backbone_indices,
    identify_backbone_atoms,
    plan_junction_placement,
)
from torchref.base.chain_closure.closure import (
    JunctionClosure,
    JunctionSolver,
    _nerf_place,
    backbone_fk_junction,
    closure_residual,
)


# =============================================================================
# Fixtures
# =============================================================================


def _make_ideal_helix(n_residues: int, start_idx: int = 0):
    """Build an ideal alpha-helix backbone (N, CA, C per residue).

    Returns (xyz, pdb_df) where xyz is shape (3*n_residues, 3) and
    pdb_df has columns: index, chainid, resseq, name, resname.
    """
    # Standard backbone bond lengths (A) and angles (rad)
    bl_CN = 1.329   # C(i-1) - N(i)
    bl_NCA = 1.458  # N - CA
    bl_CAC = 1.525  # CA - C

    ang_CN  = math.radians(116.2)  # CA(i-1)-C(i-1)-N(i)
    ang_NCA = math.radians(121.7)  # C(i-1)-N(i)-CA(i)
    ang_CAC = math.radians(111.2)  # N(i)-CA(i)-C(i)

    # Ideal helix angles
    phi = math.radians(-57.8)
    psi = math.radians(-47.0)
    omega = math.radians(180.0)

    # Seed atoms: place 3 atoms of residue 0
    positions = []
    positions.append(torch.tensor([0.0, 0.0, 0.0]))         # N(0)
    positions.append(torch.tensor([bl_NCA, 0.0, 0.0]))      # CA(0)

    # C(0) via angle
    theta = math.pi - ang_CAC
    cx = bl_NCA + bl_CAC * math.cos(theta)
    cy = bl_CAC * math.sin(theta)
    positions.append(torch.tensor([cx, cy, 0.0]))            # C(0)

    for i in range(1, n_residues):
        p1, p2, p3 = positions[-3], positions[-2], positions[-1]

        # Place N(i): torsion=psi(i-1)
        N_pos = _nerf_single(p1, p2, p3, bl_CN, ang_CN, psi)
        positions.append(N_pos)

        # Place CA(i): torsion=omega(i)
        CA_pos = _nerf_single(p2, p3, N_pos, bl_NCA, ang_NCA, omega)
        positions.append(CA_pos)

        # Place C(i): torsion=phi(i)
        C_pos = _nerf_single(p3, N_pos, CA_pos, bl_CAC, ang_CAC, phi)
        positions.append(C_pos)

    xyz = torch.stack(positions)

    # Build PDB DataFrame
    records = []
    for i in range(n_residues):
        base_idx = start_idx + i * 3
        for j, name in enumerate(["N", "CA", "C"]):
            records.append({
                "index": base_idx + j,
                "chainid": "A",
                "resseq": i + 1,
                "name": f" {name} " if len(name) < 3 else name,
                "resname": "ALA",
            })

    pdb = pd.DataFrame(records)
    return xyz, pdb


def _nerf_single(p1, p2, p3, bond_length, bond_angle, torsion):
    """Single-atom NeRF placement (non-batched)."""
    bc = p3 - p2
    bc = bc / torch.linalg.norm(bc).clamp(min=1e-10)
    ab = p2 - p1
    n = torch.linalg.cross(ab, bc)
    n = n / torch.linalg.norm(n).clamp(min=1e-10)
    m = torch.linalg.cross(n, bc)
    theta = math.pi - bond_angle
    dx = bond_length * math.cos(theta)
    dy = bond_length * math.sin(theta) * math.cos(torsion)
    dz = bond_length * math.sin(theta) * math.sin(torsion)
    return p3 + dx * bc + dy * m - dz * n


@pytest.fixture
def small_helix():
    """A 10-residue alpha helix."""
    return _make_ideal_helix(10)


@pytest.fixture
def large_helix():
    """A 50-residue alpha helix (enough for multiple junctions)."""
    return _make_ideal_helix(50)


# =============================================================================
# backbone_utils tests
# =============================================================================


class TestIdentifyBackboneAtoms:
    def test_complete_residues(self, small_helix):
        xyz, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        assert len(backbone_map) == 10
        for (chain, resseq), atoms in backbone_map.items():
            assert "N" in atoms
            assert "CA" in atoms
            assert "C" in atoms
            assert chain == "A"

    def test_non_protein_residues_excluded(self, small_helix):
        _, pdb = small_helix
        # Add a non-protein residue
        extra = pd.DataFrame([{
            "index": 30, "chainid": "A", "resseq": 100,
            "name": " O  ", "resname": "HOH",
        }])
        pdb_ext = pd.concat([pdb, extra], ignore_index=True)
        backbone_map = identify_backbone_atoms(pdb_ext)
        assert ("A", 100) not in backbone_map


class TestGetChainResidues:
    def test_single_chain(self, small_helix):
        _, pdb = small_helix
        chain_residues = get_chain_residues(pdb)
        assert "A" in chain_residues
        assert len(chain_residues["A"]) == 10
        # Should be sorted
        resseqs = [r[1] for r in chain_residues["A"]]
        assert resseqs == list(range(1, 11))


class TestComputeBackboneTorsions:
    def test_helix_phi_psi(self, small_helix):
        xyz, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)
        torsions = compute_backbone_torsions(xyz, backbone_map, chain_residues)
        assert len(torsions) == 10

        # Interior residues should have valid phi/psi close to helix
        for resseq in range(2, 10):  # residues 2-9 have both phi and psi
            t = torsions[("A", resseq)]
            assert not np.isnan(t["phi"])
            assert not np.isnan(t["psi"])
            # Helix: phi ~ -57.8 deg, psi ~ -47.0 deg (within some tolerance)
            phi_deg = np.degrees(t["phi"])
            psi_deg = np.degrees(t["psi"])
            assert -80 < phi_deg < -30, f"phi={phi_deg} for resseq {resseq}"
            assert -70 < psi_deg < -20, f"psi={psi_deg} for resseq {resseq}"

    def test_terminal_nan(self, small_helix):
        xyz, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)
        torsions = compute_backbone_torsions(xyz, backbone_map, chain_residues)
        # First residue: no phi, no omega
        assert np.isnan(torsions[("A", 1)]["phi"])
        assert np.isnan(torsions[("A", 1)]["omega"])
        # Last residue: no psi
        assert np.isnan(torsions[("A", 10)]["psi"])


class TestEstimateSecondaryStructure:
    def test_helix_assignment(self, small_helix):
        xyz, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)
        torsions = compute_backbone_torsions(xyz, backbone_map, chain_residues)
        ss = estimate_secondary_structure(torsions)

        # Interior residues should be classified as helix
        for resseq in range(2, 10):
            assert ss[("A", resseq)] == "H", f"resseq {resseq} not classified as H"


class TestPlanJunctionPlacement:
    def test_short_chain_no_junctions(self, small_helix):
        _, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)

        segments, junctions = plan_junction_placement(
            chain_residues, backbone_map,
            n_aa_per_segment=18, junction_size=3,
        )
        # 10 residues < 18 + 3 + 1, so single segment, no junctions
        assert len(segments) == 1
        assert len(junctions) == 0
        assert len(segments[0]) == 10

    def test_junctions_created(self, large_helix):
        _, pdb = large_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)

        segments, junctions = plan_junction_placement(
            chain_residues, backbone_map,
            n_aa_per_segment=18, junction_size=3,
            prefer_loops=False,
        )
        assert len(junctions) >= 1
        assert len(segments) == len(junctions) + 1

    def test_no_overlap(self, large_helix):
        _, pdb = large_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)

        segments, junctions = plan_junction_placement(
            chain_residues, backbone_map,
            n_aa_per_segment=18, junction_size=3,
            prefer_loops=False,
        )
        # All residues should appear exactly once
        all_residues = []
        for seg in segments:
            all_residues.extend(seg)
        for junc in junctions:
            all_residues.extend(junc)

        assert len(all_residues) == len(set(all_residues)), "Overlap detected"

    def test_junction_size(self, large_helix):
        _, pdb = large_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)

        segments, junctions = plan_junction_placement(
            chain_residues, backbone_map,
            n_aa_per_segment=18, junction_size=3,
            prefer_loops=False,
        )
        for junc in junctions:
            assert len(junc) == 3


class TestGetJunctionBackboneIndices:
    def test_returns_all_backbone(self, small_helix):
        _, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        junction_residues = [("A", 3), ("A", 4), ("A", 5)]
        indices = get_junction_backbone_indices(junction_residues, backbone_map)
        assert len(indices) == 3
        for d in indices:
            assert "N" in d
            assert "CA" in d
            assert "C" in d

    def test_missing_residue_raises(self, small_helix):
        _, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        junction_residues = [("A", 3), ("A", 999)]
        with pytest.raises(ValueError, match="lacks backbone atoms"):
            get_junction_backbone_indices(junction_residues, backbone_map)


# =============================================================================
# closure.py tests
# =============================================================================


class TestNerfPlace:
    def test_bond_length_preserved(self):
        """Placed atom should be at correct distance from p3."""
        p1 = torch.tensor([[0.0, 0.0, 0.0]])
        p2 = torch.tensor([[1.5, 0.0, 0.0]])
        p3 = torch.tensor([[2.5, 1.0, 0.0]])
        bl = torch.tensor([1.33])
        ba = torch.tensor([2.0])  # ~114 deg
        torsion = torch.tensor([0.5])

        new_pos = _nerf_place(p1, p2, p3, bl, ba, torsion)
        dist = torch.linalg.norm(new_pos - p3, dim=-1)
        assert torch.allclose(dist, bl, atol=1e-5), f"dist={dist.item()}, expected={bl.item()}"

    def test_bond_angle_preserved(self):
        """Bond angle at p3 should match input."""
        p1 = torch.tensor([[0.0, 0.0, 0.0]])
        p2 = torch.tensor([[1.5, 0.0, 0.0]])
        p3 = torch.tensor([[2.5, 1.0, 0.0]])  # Non-collinear with p1-p2
        bl = torch.tensor([1.5])
        ba = torch.tensor([2.094])  # 120 deg in radians
        torsion = torch.tensor([0.0])

        new_pos = _nerf_place(p1, p2, p3, bl, ba, torsion)
        v1 = p2 - p3
        v2 = new_pos - p3
        cos_a = torch.sum(v1 * v2, dim=-1) / (
            torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1)
        )
        recovered_angle = torch.acos(cos_a.clamp(-1, 1))
        assert torch.allclose(recovered_angle, ba, atol=1e-4)

    def test_batch_consistency(self):
        """Batched call should match individual calls."""
        J = 5
        p1 = torch.randn(J, 3)
        p2 = p1 + torch.randn(J, 3) * 0.5 + 1.0
        p3 = p2 + torch.randn(J, 3) * 0.5 + 1.0
        bl = torch.ones(J) * 1.5
        ba = torch.ones(J) * 2.0
        torsion = torch.randn(J)

        batched = _nerf_place(p1, p2, p3, bl, ba, torsion)
        for i in range(J):
            single = _nerf_place(
                p1[i:i+1], p2[i:i+1], p3[i:i+1],
                bl[i:i+1], ba[i:i+1], torsion[i:i+1],
            )
            assert torch.allclose(batched[i], single[0], atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through NeRF placement."""
        p3 = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
        p1 = torch.tensor([[0.0, 0.0, 0.0]])
        p2 = torch.tensor([[1.5, 0.0, 0.0]])
        bl = torch.tensor([1.33])
        ba = torch.tensor([2.0])
        torsion = torch.tensor([0.5], requires_grad=True)

        new_pos = _nerf_place(p1, p2, p3, bl, ba, torsion)
        loss = new_pos.sum()
        loss.backward()
        assert p3.grad is not None
        assert torsion.grad is not None
        assert torch.isfinite(p3.grad).all()
        assert torch.isfinite(torsion.grad).all()


class TestBackboneFKJunction:
    def test_coordinate_recovery_single_junction(self, small_helix):
        """FK with correct phi/psi should recover original backbone positions."""
        xyz, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)
        torsions = compute_backbone_torsions(xyz, backbone_map, chain_residues)

        # Use residues 4, 5, 6 as junction (0-indexed: resseq 4-6)
        junction_resseqs = [4, 5, 6]
        K = len(junction_resseqs)

        # Pre-junction residue: resseq 3
        pre_bb = backbone_map[("A", 3)]
        p1 = xyz[pre_bb["N"]]
        p2 = xyz[pre_bb["CA"]]
        p3 = xyz[pre_bb["C"]]

        # Extract phi/psi for junction
        phi_psi = []
        for rs in junction_resseqs:
            t = torsions[("A", rs)]
            phi_psi.append(t["phi"])
            phi_psi.append(t["psi"])
        phi_psi = torch.tensor(phi_psi)

        # Extract bond lengths, angles, omega
        bl_list = []
        ba_list = []
        omega_list = []
        for i, rs in enumerate(junction_resseqs):
            bb = backbone_map[("A", rs)]
            n_pos = xyz[bb["N"]]
            ca_pos = xyz[bb["CA"]]
            c_pos = xyz[bb["C"]]

            if i == 0:
                prev_bb = backbone_map[("A", 3)]
            else:
                prev_bb = backbone_map[("A", junction_resseqs[i-1])]
            prev_n = xyz[prev_bb["N"]]
            prev_ca = xyz[prev_bb["CA"]]
            prev_c = xyz[prev_bb["C"]]

            bl_list.extend([
                torch.linalg.norm(n_pos - prev_c).item(),
                torch.linalg.norm(ca_pos - n_pos).item(),
                torch.linalg.norm(c_pos - ca_pos).item(),
            ])

            # NeRF angles at pivot
            ba_list.extend([
                _compute_angle_val(prev_ca, prev_c, n_pos),
                _compute_angle_val(prev_c, n_pos, ca_pos),
                _compute_angle_val(n_pos, ca_pos, c_pos),
            ])

            # Omega
            omega_list.append(
                _compute_torsion_val(prev_ca, prev_c, n_pos, ca_pos)
            )

        nerf_bl = torch.tensor(bl_list)
        nerf_ba = torch.tensor(ba_list)
        omega = torch.tensor(omega_list)

        # psi_prev
        psi_prev = torch.tensor(torsions[("A", 3)]["psi"])

        # Run FK
        end_p1, end_p2, end_p3, backbone_xyz, _ = backbone_fk_junction(
            p1, p2, p3, phi_psi, nerf_bl, nerf_ba, omega, psi_prev,
        )

        # Check that backbone_xyz matches original positions
        for i, rs in enumerate(junction_resseqs):
            bb = backbone_map[("A", rs)]
            expected_N = xyz[bb["N"]]
            expected_CA = xyz[bb["CA"]]
            expected_C = xyz[bb["C"]]

            computed_N = backbone_xyz[3*i]
            computed_CA = backbone_xyz[3*i + 1]
            computed_C = backbone_xyz[3*i + 2]

            assert torch.allclose(computed_N, expected_N, atol=1e-3), \
                f"N mismatch at res {rs}: diff={torch.linalg.norm(computed_N - expected_N):.6f}"
            assert torch.allclose(computed_CA, expected_CA, atol=1e-3), \
                f"CA mismatch at res {rs}: diff={torch.linalg.norm(computed_CA - expected_CA):.6f}"
            assert torch.allclose(computed_C, expected_C, atol=1e-3), \
                f"C mismatch at res {rs}: diff={torch.linalg.norm(computed_C - expected_C):.6f}"

    def test_batched_vs_unbatched(self):
        """Batched FK should match unbatched."""
        K = 3
        p1 = torch.randn(3)
        p2 = p1 + torch.tensor([1.5, 0.0, 0.0])
        p3 = p2 + torch.tensor([0.5, 1.2, 0.0])
        phi_psi = torch.randn(2 * K)
        bl = torch.ones(3 * K) * 1.5
        ba = torch.ones(3 * K) * 2.0
        omega = torch.ones(K) * math.pi
        psi_prev = torch.tensor(0.0)

        # Unbatched
        ep1_u, ep2_u, ep3_u, bxyz_u, _ = backbone_fk_junction(
            p1, p2, p3, phi_psi, bl, ba, omega, psi_prev,
        )

        # Batched
        ep1_b, ep2_b, ep3_b, bxyz_b, _ = backbone_fk_junction(
            p1.unsqueeze(0), p2.unsqueeze(0), p3.unsqueeze(0),
            phi_psi.unsqueeze(0), bl.unsqueeze(0), ba.unsqueeze(0),
            omega.unsqueeze(0), psi_prev.unsqueeze(0),
        )

        assert torch.allclose(ep3_u, ep3_b[0], atol=1e-5)
        assert torch.allclose(bxyz_u, bxyz_b[0], atol=1e-5)


class TestClosureResidual:
    def test_zero_at_target(self):
        target = torch.tensor([[1.0, 2.0, 3.0]])
        res = closure_residual(target, target)
        assert torch.allclose(res, torch.zeros_like(res), atol=1e-10)

    def test_nonzero_off_target(self):
        end = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[4.0, 5.0, 6.0]])
        res = closure_residual(end, target)
        assert torch.allclose(res, end - target)


class TestJunctionClosure:
    """Test the custom autograd Function for Newton-based closure."""

    @pytest.fixture
    def closure_data(self, small_helix):
        """Set up a junction closure problem from the helix fixture."""
        xyz, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)
        torsions = compute_backbone_torsions(xyz, backbone_map, chain_residues)

        junction_resseqs = [4, 5, 6]
        K = len(junction_resseqs)

        pre_bb = backbone_map[("A", 3)]
        p1 = xyz[pre_bb["N"]].clone()
        p2 = xyz[pre_bb["CA"]].clone()
        p3 = xyz[pre_bb["C"]].clone()

        phi_psi = []
        bl_list, ba_list, omega_list = [], [], []

        for i, rs in enumerate(junction_resseqs):
            bb = backbone_map[("A", rs)]
            n_pos = xyz[bb["N"]]
            ca_pos = xyz[bb["CA"]]
            c_pos = xyz[bb["C"]]

            if i == 0:
                prev_bb = backbone_map[("A", 3)]
            else:
                prev_bb = backbone_map[("A", junction_resseqs[i-1])]
            prev_n = xyz[prev_bb["N"]]
            prev_ca = xyz[prev_bb["CA"]]
            prev_c = xyz[prev_bb["C"]]

            t = torsions[("A", rs)]
            phi_psi.extend([t["phi"], t["psi"]])

            bl_list.extend([
                torch.linalg.norm(n_pos - prev_c).item(),
                torch.linalg.norm(ca_pos - n_pos).item(),
                torch.linalg.norm(c_pos - ca_pos).item(),
            ])
            ba_list.extend([
                _compute_angle_val(prev_ca, prev_c, n_pos),
                _compute_angle_val(prev_c, n_pos, ca_pos),
                _compute_angle_val(n_pos, ca_pos, c_pos),
            ])
            omega_list.append(
                _compute_torsion_val(prev_ca, prev_c, n_pos, ca_pos)
            )

        # Target: N of post-junction residue (residue 7)
        last_bb = backbone_map[("A", junction_resseqs[-1])]
        post_bb = backbone_map[("A", 7)]
        target_n = xyz[post_bb["N"]].clone()

        # Post-junction geometry: C(last_junc) -> N(post)
        last_c = xyz[last_bb["C"]]
        last_ca = xyz[last_bb["CA"]]
        post_n = xyz[post_bb["N"]]
        post_bl = torch.linalg.norm(post_n - last_c)
        post_ba = _compute_angle_val(last_ca, last_c, post_n)

        psi_prev = torch.tensor(torsions[("A", 3)]["psi"])

        return {
            "phi_psi_init": torch.tensor(phi_psi).unsqueeze(0),
            "p1_start": p1.unsqueeze(0),
            "p2_start": p2.unsqueeze(0),
            "p3_start": p3.unsqueeze(0),
            "target_p3": target_n.unsqueeze(0),
            "nerf_bond_lengths": torch.tensor(bl_list).unsqueeze(0),
            "nerf_bond_angles": torch.tensor(ba_list).unsqueeze(0),
            "omega": torch.tensor(omega_list).unsqueeze(0),
            "psi_prev": psi_prev.unsqueeze(0),
            "post_bond_length": torch.tensor([post_bl]),
            "post_bond_angle": torch.tensor([post_ba]),
        }

    def test_solver_convergence(self, closure_data):
        """Newton solver should converge to near-zero residual."""
        d = closure_data
        # Perturb initial guess to make solver work
        perturbed_init = d["phi_psi_init"] + 0.1 * torch.randn_like(d["phi_psi_init"])

        phi_psi = JunctionClosure.apply(
            perturbed_init,
            d["p1_start"], d["p2_start"], d["p3_start"],
            d["target_p3"], d["nerf_bond_lengths"], d["nerf_bond_angles"],
            d["omega"], d["psi_prev"],
            d["post_bond_length"], d["post_bond_angle"],
            20, 1e-6, 1e-6,
        )

        # Check residual
        _, _, _, _, end_point = backbone_fk_junction(
            d["p1_start"], d["p2_start"], d["p3_start"],
            phi_psi, d["nerf_bond_lengths"], d["nerf_bond_angles"],
            d["omega"], d["psi_prev"],
            d["post_bond_length"], d["post_bond_angle"],
        )
        res = closure_residual(end_point, d["target_p3"])
        res_norm = torch.linalg.norm(res, dim=-1).max().item()
        assert res_norm < 0.01, f"Residual too large: {res_norm}"

    def test_backward_produces_gradients(self, closure_data):
        """Backward pass should produce non-NaN gradients for start points."""
        d = closure_data
        p3 = d["p3_start"].clone().requires_grad_(True)

        phi_psi = JunctionClosure.apply(
            d["phi_psi_init"],
            d["p1_start"], d["p2_start"], p3,
            d["target_p3"], d["nerf_bond_lengths"], d["nerf_bond_angles"],
            d["omega"], d["psi_prev"],
            d["post_bond_length"], d["post_bond_angle"],
            20, 1e-6, 1e-6,
        )
        loss = phi_psi.sum()
        loss.backward()
        assert p3.grad is not None, "No gradient for p3_start"
        assert torch.isfinite(p3.grad).all(), f"Non-finite gradient: {p3.grad}"


class TestJunctionSolver:
    def test_solver_forward(self, small_helix):
        """JunctionSolver produces backbone xyz and phi/psi."""
        xyz, pdb = small_helix
        backbone_map = identify_backbone_atoms(pdb)
        chain_residues = get_chain_residues(pdb)
        torsions = compute_backbone_torsions(xyz, backbone_map, chain_residues)

        K = 3
        init_pp = torch.zeros(1, 2*K)
        solver = JunctionSolver(
            n_junctions=1, junction_size=K,
            initial_phi_psi=init_pp,
            max_iter=20, tol=1e-6,
        )

        p1 = torch.randn(1, 3)
        p2 = p1 + torch.randn(1, 3)
        p3 = p2 + torch.randn(1, 3)
        target = p3 + torch.randn(1, 3) * 0.1 + 2.0
        bl = torch.ones(1, 3*K) * 1.5
        ba = torch.ones(1, 3*K) * 2.0
        omega = torch.ones(1, K) * math.pi
        psi_prev = torch.zeros(1)

        phi_psi, backbone_xyz = solver(
            p1, p2, p3, target, bl, ba, omega, psi_prev,
        )

        assert phi_psi.shape == (1, 2*K)
        assert backbone_xyz.shape == (1, 3*K, 3)

    def test_zero_junctions(self):
        """Solver with zero junctions returns empty tensors."""
        solver = JunctionSolver(
            n_junctions=0, junction_size=3,
            initial_phi_psi=torch.zeros(0, 6),
        )
        p1 = torch.randn(0, 3)
        pp, bxyz = solver(p1, p1, p1, p1,
                          torch.zeros(0, 9), torch.zeros(0, 9),
                          torch.zeros(0, 3), torch.zeros(0))
        assert pp.shape[0] == 0
        assert bxyz.shape[0] == 0

    def test_warm_start_updated(self, small_helix):
        """Warm start buffer should be updated after each forward call."""
        K = 3
        init_pp = torch.zeros(1, 2*K)
        solver = JunctionSolver(
            n_junctions=1, junction_size=K,
            initial_phi_psi=init_pp,
            max_iter=5, tol=1e-2,
        )
        old_ws = solver.warm_start.clone()

        p1 = torch.randn(1, 3)
        p2 = p1 + torch.tensor([[1.5, 0.0, 0.0]])
        p3 = p2 + torch.tensor([[0.5, 1.2, 0.0]])
        target = p3 + torch.tensor([[3.0, 2.0, 1.0]])
        bl = torch.ones(1, 3*K) * 1.5
        ba = torch.ones(1, 3*K) * 2.0
        omega = torch.ones(1, K) * math.pi
        psi_prev = torch.zeros(1)

        solver(p1, p2, p3, target, bl, ba, omega, psi_prev)
        assert not torch.allclose(solver.warm_start, old_ws), \
            "Warm start not updated"


# =============================================================================
# Helpers
# =============================================================================

def _compute_angle_val(p1, p2, p3):
    """Compute angle at p2 between p1-p2-p3, return float."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = torch.dot(v1, v2) / (
        torch.linalg.norm(v1) * torch.linalg.norm(v2) + 1e-10
    )
    return torch.acos(cos_a.clamp(-1, 1)).item()


def _compute_torsion_val(p1, p2, p3, p4):
    """Compute torsion angle, return float."""
    return _torsion_angle(p1, p2, p3, p4).item()
