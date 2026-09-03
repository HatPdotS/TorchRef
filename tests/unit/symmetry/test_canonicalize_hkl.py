"""Tests for canonicalize_hkl and ReflectionData.canonicalize."""

import numpy as np
import pytest
import torch

from torchref.symmetry import SpaceGroup


# The HKL verbs live on the space group now. These adapters keep the assertions
# below -- which pin the phase-sign contract -- expressed in terms of the space
# group specifications the cases are parametrised over.
def canonicalize_hkl(hkl, sg, include_friedel=True, device=None):
    return SpaceGroup(sg).canonicalize_hkl(
        hkl, include_friedel=include_friedel, device=device
    )


def expand_hkl(hkl, sg, include_friedel=True, remove_absences=True, device=None):
    return SpaceGroup(sg).expand_hkl(
        hkl,
        include_friedel=include_friedel,
        remove_absences=remove_absences,
        device=device,
    )


def reduce_hkl(hkl, sg, include_friedel=True, device=None):
    return SpaceGroup(sg).reduce_hkl(
        hkl, include_friedel=include_friedel, device=device
    )


# ---------------------------------------------------------------------------
# canonicalize_hkl unit tests
# ---------------------------------------------------------------------------


class TestCanonicalizeHkl:
    """Tests for the standalone canonicalize_hkl function."""

    @pytest.mark.parametrize(
        "sg",
        ["P1", "P21", "P212121", "C2", "P4"],
    )
    def test_idempotency(self, sg):
        """canonicalize(canonicalize(hkl)) == canonicalize(hkl)."""
        hkl = torch.tensor(
            [[1, 2, 3], [-1, 0, 2], [0, 3, -1], [2, -1, 4]],
            dtype=torch.int32,
        )
        can1, ps1, ff1, si1 = canonicalize_hkl(hkl, sg)

        # Second pass on already-canonical output
        can2, ps2, ff2, si2 = canonicalize_hkl(can1, sg)

        torch.testing.assert_close(can2, can1)
        # Phase shifts and friedel flags should all be zero / False
        assert torch.allclose(ps2, torch.zeros_like(ps2), atol=1e-5)
        assert not ff2.any(), "No Friedel flags expected on already-canonical HKL"

    @pytest.mark.parametrize(
        "sg",
        ["P21", "P212121", "C2"],
    )
    def test_equivalents_converge(self, sg):
        """All symmetry equivalents of a reflection map to the same canonical HKL."""
        # Use (1,1,5) which satisfies centering conditions for C2
        hkl_asu = torch.tensor([[1, 1, 5]], dtype=torch.int32)
        hkl_p1, _, _ = expand_hkl(hkl_asu, sg, include_friedel=True)

        if len(hkl_p1) == 0:
            pytest.skip(f"expand_hkl returned empty for {sg}")

        can, _, _, _ = canonicalize_hkl(hkl_p1, sg, include_friedel=True)

        # All canonical HKLs should be identical
        for i in range(1, len(can)):
            torch.testing.assert_close(
                can[i], can[0], msg=f"Row {i} differs from row 0"
            )

    @pytest.mark.parametrize(
        "sg",
        ["P21", "P212121", "C2", "P4"],
    )
    def test_phase_roundtrip(self, sg):
        """F * exp(i*phi) is preserved after canonicalization.

        We construct physically correct phases for symmetry equivalents
        (Friedel mates are conjugated, sym-ops get translation phases),
        then verify canonicalization produces a single consistent SF.
        """
        import gemmi

        # Reference SF at canonical h
        F_ref = 10.0
        phi_ref = 0.7
        sf_ref = F_ref * torch.exp(torch.tensor(1j * phi_ref, dtype=torch.complex64))

        # Get symmetry operations
        sg_obj = SpaceGroup(sg, dtype=torch.float32, device=torch.device("cpu"))
        recip_mats = sg_obj.matrices.transpose(-2, -1)
        translations = sg_obj.translations
        n_ops = len(recip_mats)

        # Canonical hkl — use (1,1,5) which satisfies C2 centering
        h_can = torch.tensor([1, 1, 5], dtype=torch.float32)

        # Generate all equivalents with physically correct phases
        hkl_list = []
        phi_list = []
        for i in range(n_ops):
            R = recip_mats[i]
            t = translations[i]
            h_sym = torch.round(R @ h_can).to(torch.int32)
            # Phase at h_sym: phi(h_sym) = phi_ref - 2*pi*h_can . t
            ps = 2.0 * np.pi * torch.dot(h_can, t).item()
            hkl_list.append(h_sym)
            phi_list.append(phi_ref - ps)

            # Friedel mate
            hkl_list.append(-h_sym)
            # F(-h) = F*(h) => phi(-h) = -phi(h)
            phi_list.append(-(phi_ref - ps))

        hkl_all = torch.stack(hkl_list)
        phi_all = torch.tensor(phi_list, dtype=torch.float32)

        # Remove exact duplicates
        seen = {}
        unique_idx = []
        for i, h in enumerate(hkl_all):
            key = tuple(h.tolist())
            if key not in seen:
                seen[key] = i
                unique_idx.append(i)
        hkl_uniq = hkl_all[unique_idx]
        phi_uniq = phi_all[unique_idx]
        F_uniq = torch.full((len(hkl_uniq),), F_ref)

        if len(hkl_uniq) == 0:
            pytest.skip(f"No unique equivalents for {sg}")

        # Canonicalize
        can_hkl, can_ps, can_ff, can_si = canonicalize_hkl(hkl_uniq, sg)

        # Apply correction
        phi_sorted = phi_uniq[can_si]
        F_sorted = F_uniq[can_si]
        phi_can = torch.where(can_ff, -phi_sorted, phi_sorted) + can_ps

        # Reconstruct complex SF
        sf_can = F_sorted * torch.exp(1j * phi_can.to(torch.complex64))

        # All should agree
        for i in range(1, len(sf_can)):
            torch.testing.assert_close(
                sf_can[i].real, sf_can[0].real, atol=1e-3, rtol=1e-3,
            )
            torch.testing.assert_close(
                sf_can[i].imag, sf_can[0].imag, atol=1e-3, rtol=1e-3,
            )

    def test_friedel_pair(self):
        """(1,1,1) and (-1,-1,-1) map to the same canonical form."""
        hkl = torch.tensor([[1, 1, 1], [-1, -1, -1]], dtype=torch.int32)
        can, _, _, _ = canonicalize_hkl(hkl, "P1", include_friedel=True)
        torch.testing.assert_close(can[0], can[1])

    def test_friedel_flagged_correctly(self):
        """For P1 with (1,1,1) and (-1,-1,-1), exactly one has friedel=True."""
        hkl = torch.tensor([[1, 1, 1], [-1, -1, -1]], dtype=torch.int32)
        _, _, ff, _ = canonicalize_hkl(hkl, "P1", include_friedel=True)
        # One should be identity, one should be Friedel
        assert ff.sum().item() == 1

    def test_sort_order(self):
        """Output is lexicographically sorted by (h, k, l)."""
        hkl = torch.tensor(
            [[3, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]],
            dtype=torch.int32,
        )
        can, _, _, _ = canonicalize_hkl(hkl, "P1", include_friedel=False)
        # Verify sorted order
        for i in range(len(can) - 1):
            a = can[i].tolist()
            b = can[i + 1].tolist()
            assert a <= b, f"Not sorted: {a} > {b}"

    @pytest.mark.parametrize("sg", ["P21", "P212121"])
    def test_systematic_absences(self, sg):
        """Systematic absences are mapped to canonical form (not rejected)."""
        import gemmi

        sg_gemmi = gemmi.SpaceGroup(sg)
        ops = sg_gemmi.operations()

        # Find a reflection that is systematically absent
        absent_hkl = None
        for h in range(5):
            for k in range(5):
                for l in range(5):
                    if h == k == l == 0:
                        continue
                    if ops.is_systematically_absent([h, k, l]):
                        absent_hkl = [h, k, l]
                        break
                if absent_hkl:
                    break
            if absent_hkl:
                break

        if absent_hkl is None:
            pytest.skip(f"No systematic absence found for {sg} in range")

        hkl = torch.tensor([absent_hkl], dtype=torch.int32)
        can, ps, ff, si = canonicalize_hkl(hkl, sg)
        # Should still return a valid canonical HKL (not crash)
        assert can.shape == (1, 3)

    def test_returns_correct_shapes(self):
        """Verify output shapes match input."""
        n = 50
        hkl = torch.randint(-5, 6, (n, 3), dtype=torch.int32)
        # Remove (0,0,0)
        mask = (hkl != 0).any(dim=1)
        hkl = hkl[mask]
        n = len(hkl)

        can, ps, ff, si = canonicalize_hkl(hkl, "P212121")
        assert can.shape == (n, 3)
        assert ps.shape == (n,)
        assert ff.shape == (n,)
        assert si.shape == (n,)
        assert can.dtype == torch.int32
        assert ps.dtype == torch.float32
        assert ff.dtype == torch.bool
        assert si.dtype == torch.int64

    def test_sort_indices_is_valid_permutation(self):
        """sort_indices should be a valid permutation of range(N)."""
        hkl = torch.tensor(
            [[3, 2, 1], [1, 0, 0], [-1, -1, -1], [2, 1, 3]],
            dtype=torch.int32,
        )
        _, _, _, si = canonicalize_hkl(hkl, "P21")
        assert set(si.tolist()) == set(range(len(hkl)))

    def test_unmappable_reflection_raises(self):
        """include_friedel=False on a Friedel-half reflection must fail loudly.

        The CCP4 reciprocal ASU is Laue-based, so without Friedel mates the
        "minus" half of reciprocal space has no pure-rotation representative.
        Such rows were previously left as uninitialized ``np.empty`` memory
        (silent garbage Miller indices); they must now raise instead.
        """
        hkl = torch.tensor([[-1, 0, 0]], dtype=torch.int32)
        with pytest.raises(ValueError, match="could not map"):
            canonicalize_hkl(hkl, "P1", include_friedel=False)

    def test_empty_input(self):
        """Empty input should return empty tensors without error."""
        hkl = torch.empty((0, 3), dtype=torch.int32)
        can, ps, ff, si = canonicalize_hkl(hkl, "P21")
        assert can.shape == (0, 3)
        assert ps.shape == (0,)
        assert ff.shape == (0,)
        assert si.shape == (0,)
