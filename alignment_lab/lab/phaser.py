"""Phaser oracle adapter.

Phaser is the reference the FRF is measured against, so this wraps invoking it
and reading its peaks back in our conventions. Previously these helpers lived
inside a pytest module and six scripts imported them from there.

Three details are load-bearing and easy to lose:

* **Convention.** ``R_ours = R_phaser.T``. Calibrated empirically in P1, where
  ``n_ops == 1`` leaves no orbit ambiguity to hide a transpose error.
* **``PEAKS ROT SELECT ALL``** with clustering off. Phaser otherwise merges
  symmetry equivalents before we can rank them. Expect ~80-92k samples with a
  median nearest-neighbour spacing under 1 degree, so "the nearest sample is
  within 1 degree" is not evidence of anything on its own.
* **Phaser exits 0 on fatal input errors.** The return code is not a success
  test; an empty peak list is the real signal. Keyword files also need
  **absolute** paths.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

_SOLU_TRIAL_RE = re.compile(
    r"SOLU\s+TRIAL\s+ENSEMBLE\s+\S+\s+"
    r"EULER\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+"
    r"RF\s+([-+\d.eE]+)\s+RFZ\s+([-+\d.eE]+)",
    re.IGNORECASE,
)


@dataclass
class PhaserPeak:
    """One Phaser FRF peak. Euler angles in **degrees**, Edmonds ZYZ."""

    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    rf: float
    rfz: float


def write_frf_keywords(
    work: Path, *, mtz_path: Path, model_pdb: Path,
    f_label: str = "FP", sigf_label: str = "SIGFP", root: str = "phaser_frf",
) -> Path:
    """Write an MR_FRF keyword file. Paths are resolved to absolute.

    Parameters
    ----------
    work : Path
        Working directory; created if absent.
    mtz_path, model_pdb : Path
        Inputs. Relative paths are resolved -- Phaser fails on relative ones.
    f_label, sigf_label : str, optional
        MTZ column labels.
    root : str, optional
        Phaser output root.

    Returns
    -------
    Path
        The keyword file.
    """
    work.mkdir(parents=True, exist_ok=True)
    kw = work / f"{root}.kw"
    kw.write_text(
        f"TITLE FRF rotation ranking\n"
        f"MODE MR_FRF\n"
        f"HKLIN {Path(mtz_path).resolve()}\n"
        f"LABIN F={f_label} SIGF={sigf_label}\n"
        f"ENSEMBLE search PDB {Path(model_pdb).resolve()} IDENT 1.0\n"
        f"COMPOSITION BY AVERAGE\n"
        f"SEARCH ENSEMBLE search\n"
        f"PEAKS ROT SELECT ALL\n"
        f"PEAKS ROT CLUSTER OFF\n"
        f"PEAKS ROT LEVEL 0\n"
        f"ROOT {root}\n"
    )
    return kw


def run_phaser(work: Path, kw_path: Path, timeout_s: int = 5400) -> Tuple[int, float]:
    """Run ``phenix.phaser`` on a keyword file.

    Returns
    -------
    tuple
        ``(returncode, seconds)``; ``-1`` on timeout. **A zero return code does
        not mean success** -- check that :func:`parse_rlist` found peaks.
    """
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["phenix.phaser"], cwd=str(work), input=kw_path.read_text(),
            capture_output=True, text=True, timeout=timeout_s,
        )
        (work / "phaser.stdout").write_text(proc.stdout or "")
        (work / "phaser.stderr").write_text(proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    return rc, time.time() - t0


def parse_rlist(path: Path) -> List[PhaserPeak]:
    """Parse ``SOLU TRIAL`` lines from a Phaser ``.rlist``.

    Returns an empty list when the file is absent, which is also what a failed
    run looks like -- see the note on exit codes in the module docstring.
    """
    path = Path(path)
    if not path.exists():
        return []
    peaks: List[PhaserPeak] = []
    for line in path.read_text().splitlines():
        if "SOLU TRIAL" not in line.upper():
            continue
        m = _SOLU_TRIAL_RE.search(line)
        if m is None:
            continue
        a, b, g, rf, rfz = (float(x) for x in m.groups())
        peaks.append(PhaserPeak(a, b, g, rf, rfz))
    peaks.sort(key=lambda p: p.rfz, reverse=True)
    return peaks


def euler_deg_to_matrices(peaks: Sequence[PhaserPeak]) -> torch.Tensor:
    """Stack Phaser peaks as rotation matrices in **Phaser's** frame.

    Edmonds ZYZ active rotation ``R = Rz(alpha) Ry(beta) Rz(gamma)``. Apply
    :func:`to_our_frame` before comparing against our orbit.

    Returns
    -------
    torch.Tensor
        ``(n, 3, 3)`` float64. Empty ``(0, 3, 3)`` for an empty input.
    """
    if not peaks:
        return torch.zeros((0, 3, 3), dtype=torch.float64)
    a = torch.tensor([p.alpha_deg for p in peaks], dtype=torch.float64).deg2rad()
    b = torch.tensor([p.beta_deg for p in peaks], dtype=torch.float64).deg2rad()
    g = torch.tensor([p.gamma_deg for p in peaks], dtype=torch.float64).deg2rad()
    ca, sa, cb, sb, cg, sg = a.cos(), a.sin(), b.cos(), b.sin(), g.cos(), g.sin()
    return torch.stack([
        torch.stack([ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb], dim=-1),
        torch.stack([sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb], dim=-1),
        torch.stack([-sb * cg, sb * sg, cb], dim=-1),
    ], dim=-2)


def to_our_frame(R_phaser: torch.Tensor) -> torch.Tensor:
    """Convert Phaser-frame rotations to ours: ``R_ours = R_phaser.T``.

    Calibrated in P1 (``n_ops == 1``), where no symmetry orbit can mask a
    transposition. Do not re-derive this per structure: with ~80-92k samples,
    both conventions match *something* within a degree.
    """
    return R_phaser.transpose(-1, -2)


def phaser_truth_rank(
    peaks: Sequence[PhaserPeak],
    R_true: torch.Tensor,
    symops: torch.Tensor,
    *,
    reciprocal_basis: Optional[torch.Tensor] = None,
    frame: str = "cart",
    side: str = "left",
    thr_deg: float = 5.0,
) -> Tuple[int, float]:
    """Rank of the true orientation in Phaser's own peak list.

    Uses the same orbit machinery as our engine, after mapping Phaser's frame
    onto ours, so the two ranks are directly comparable.

    Returns
    -------
    tuple
        ``(rank, best_angle_deg)``; rank ``-1`` if unmatched.
    """
    from .truth import angle_to_orbit, symmetry_orbit

    if not peaks:
        return -1, float("inf")
    orbit = symmetry_orbit(
        R_true, symops, side=side, frame=frame, reciprocal_basis=reciprocal_basis,
    )
    R_ours = to_our_frame(euler_deg_to_matrices(peaks))
    rank, best = -1, float("inf")
    for i in range(R_ours.shape[0]):
        ang = angle_to_orbit(R_ours[i], orbit)
        if ang < best:
            best = ang
        if ang <= thr_deg and rank < 0:
            rank = i
    return rank, best
