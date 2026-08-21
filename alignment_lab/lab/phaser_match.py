"""Reproduce Phaser's FRF parameter chain exactly, and read back its map.

Every number the rotation function depends on -- spherical-harmonic bandwidth,
the resolution actually expanded, and the SO(3) sampling step -- is *derived* by
Phaser from one quantity: ``mean_radius()``. This module implements that chain
verbatim from the 1.20 source so our engine can be pinned to the same values,
and parses the patched binary's log/dump so the derivation can be checked
against what Phaser actually did rather than trusted.

Source anchors (PHENIX 1.20-4459, ``modules/phaser/codebase/phaser``):

* ``lib/xyz_weight.cc:178`` ``mean_radius()`` -- the mean of the three
  principal-axis **semi-extents of the bounding box**, NOT the mean atomic
  distance from the centroid. These differ by ~25-30% on a protein.
* ``run/runMR_FRF.cc:406-410`` bandwidth::

      sphereOuter = 2 * mean_radius
      LMAX = ceil(2*pi*sphereOuter / HiRes)   # round UP to even
      LMAX = min(LMAX, DEF_CLMN_LMAX = 100)

* ``run/runMR_FRF.cc:411-419`` resolution -- coarsened **only** when the cap
  binds::

      LMAX_RESO = (LMAX == 100) ? 2*pi*sphereOuter/LMAX : HiRes

* ``run/runMR_FRF.cc:469-474`` sampling -- likewise keyed on the cap::

      SAMP_RESO = (LMAX == 100) ? LMAX_RESO : HiRes
      sampling  = 2 * degrees(atan(SAMP_RESO / (4 * mean_radius)))

The three are one coupled system: when the bandwidth saturates, the resolution
and the angular step coarsen together so the expansion is never asked to carry
detail it cannot represent.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

#: ``DEF_CLMN_LMAX`` from ``phaser_src/defaults:19``.
PHASER_LMAX_CAP = 100

#: ``DEF_CLMN_SPHE`` from ``phaser_src/defaults:17``. Zero means "use
#: ``2 * mean_radius``" rather than an explicit sphere radius.
PHASER_SPHERE_DEFAULT = 0.0

#: Built by the recipe in the ``phaser-instrumented-build`` memo. Honours
#: ``$PHASER_FRF_DUMP`` and is otherwise stock.
PATCHED_PHASER = (
    Path(__file__).resolve().parents[2] / "phaser_src" / "build" / "phaser_patched"
)


def phaser_mean_radius(model) -> float:
    """``xyz_weight::mean_radius()`` -- mean principal-axis semi-extent.

    Rotates the coordinates onto the principal axes of their covariance, takes
    the bounding-box extent along each axis, halves it, and averages the three.

    This is emphatically *not* ``mean(|xyz - centroid|)``; on 1DAW the two give
    26.2 A and 19.5 A. Since ``LMAX`` and the sampling step are both derived
    from it, using the wrong one detunes the whole rotation function.

    Parameters
    ----------
    model : ModelFT
        Search model.

    Returns
    -------
    float
        Mean radius in Angstrom.
    """
    xyz = model.xyz().to(torch.float64)
    centred = xyz - xyz.mean(dim=0)
    cov = (centred.T @ centred) / centred.shape[0]
    _, axes = torch.linalg.eigh(cov)
    projected = centred @ axes
    extent = projected.max(dim=0).values - projected.min(dim=0).values
    return float((extent / 2.0).mean().item())


@dataclass
class PhaserFRFParams:
    """The derived FRF parameters for one case.

    Attributes
    ----------
    mean_radius_A : float
        ``mean_radius()``.
    hires_A : float
        ``mr.HiRes()`` -- the high-resolution limit of the selected data.
    lmax : int
        Maximum ``l`` of the expansion (even, capped at 100).
    lmax_reso_A : float
        Resolution the expansion actually runs at.
    samp_reso_A : float
        Resolution feeding the sampling formula.
    sampling_deg : float
        SO(3) grid step in degrees.
    capped : bool
        Whether ``lmax`` hit ``PHASER_LMAX_CAP`` -- when True the resolution and
        sampling are both coarsened, when False both stay at ``hires_A``.
    """

    mean_radius_A: float
    hires_A: float
    lmax: int
    lmax_reso_A: float
    samp_reso_A: float
    sampling_deg: float
    capped: bool

    def as_row(self) -> dict:
        """Flatten for a CSV result row, prefixed ``phaser_``."""
        return {f"phaser_{k}": v for k, v in asdict(self).items()}


def phaser_frf_params(
    mean_radius_A: float,
    hires_A: float,
    *,
    lmax_cap: int = PHASER_LMAX_CAP,
    use_rotate_lmax_reso: bool = True,
) -> PhaserFRFParams:
    """Run Phaser's bandwidth/resolution/sampling chain.

    Parameters
    ----------
    mean_radius_A : float
        From :func:`phaser_mean_radius`.
    hires_A : float
        High-resolution limit of the data being expanded.
    lmax_cap : int, optional
        ``DEF_CLMN_LMAX``. Default 100. Lower it to emulate our historical
        ``lmax_cap`` settings *with Phaser's coupling intact* -- note that
        coarsening then engages, exactly as it does in Phaser at 100.
    use_rotate_lmax_reso : bool, optional
        ``input.USE_ROTATE_LMAX_RESO``. Default True (Phaser's default).

    Returns
    -------
    PhaserFRFParams
    """
    sphere_outer = 2.0 * float(mean_radius_A)
    lmax = int(math.ceil(2.0 * math.pi * sphere_outer / float(hires_A)))
    if lmax % 2 != 0:
        lmax += 1
    lmax = min(lmax, int(lmax_cap))

    capped = lmax == int(lmax_cap) and use_rotate_lmax_reso
    if capped:
        lmax_reso = 2.0 * math.pi * sphere_outer / lmax
        samp_reso = lmax_reso
    else:
        lmax_reso = float(hires_A)
        samp_reso = float(hires_A)

    sampling = 2.0 * math.degrees(math.atan(samp_reso / (4.0 * float(mean_radius_A))))
    return PhaserFRFParams(
        mean_radius_A=float(mean_radius_A),
        hires_A=float(hires_A),
        lmax=lmax,
        lmax_reso_A=lmax_reso,
        samp_reso_A=samp_reso,
        sampling_deg=sampling,
        capped=capped,
    )


# ---------------------------------------------------------------------------
# Running the patched binary
# ---------------------------------------------------------------------------

#: ``RFACTOR USE OFF`` is compulsory for any diagnostic that uses a model
#: already close to its answer: Phaser otherwise computes the R-factor of the
#: ensemble at the origin, decides the structure is solved, and emits a single
#: identity peak with ``RF*0`` -- skipping the rotation search entirely while
#: still reporting ``EXIT STATUS: SUCCESS``.
_KEYWORD_TEMPLATE = """TITLE {title}
MODE MR_FRF
HKLIN {mtz}
LABIN F={f_label} SIGF={sigf_label}
ENSEMBLE search PDB {pdb} IDENT 1.0
COMPOSITION BY AVERAGE
SEARCH ENSEMBLE search
RFACTOR USE OFF
PEAKS ROT SELECT NUMBER
PEAKS ROT CUTOFF {n_peaks}
PEAKS ROT CLUSTER OFF
OUTPUT LEVEL VERBOSE
ROOT {root}
"""


def write_keywords(
    work: Path,
    *,
    mtz_path: Path,
    model_pdb: Path,
    n_peaks: int = 20,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    f_label: str = "FP",
    sigf_label: str = "SIGFP",
    root: str = "phaser_frf",
    title: str = "FRF map dump",
) -> Path:
    """Write an MR_FRF keyword file for the patched binary.

    ``d_min``/``d_max`` are omitted by default so Phaser uses the full data
    range and its own coupling picks the expansion resolution -- that is the
    configuration our engine should be matched against.

    Note the peak keyword takes two cards: ``PEAKS ROT SELECT NUMBER`` sets the
    *mode* and ``PEAKS ROT CUTOFF n`` the count. ``SELECT NUMBER n`` is a syntax
    error. Keeping the count small matters: with ``SELECT ALL`` Phaser rescores
    every sample point (150k+ for a mid-size case), which dwarfs the search.
    """
    work.mkdir(parents=True, exist_ok=True)
    text = _KEYWORD_TEMPLATE.format(
        title=title,
        mtz=Path(mtz_path).resolve(),
        pdb=Path(model_pdb).resolve(),
        f_label=f_label,
        sigf_label=sigf_label,
        n_peaks=int(n_peaks),
        root=root,
    )
    if d_min is not None and d_max is not None:
        text = text.replace(
            "RFACTOR USE OFF", f"RESOLUTION {d_min} {d_max}\nRFACTOR USE OFF",
        )
    kw = work / f"{root}.kw"
    kw.write_text(text)
    return kw


def run_patched_phaser(
    work: Path,
    kw_path: Path,
    *,
    dump_path: Optional[Path] = None,
    binary: Path = PATCHED_PHASER,
    timeout_s: int = 5400,
) -> Tuple[int, float, Path]:
    """Run the instrumented binary, dumping the FRF sample list.

    Returns
    -------
    tuple
        ``(returncode, seconds, log_path)``. **The return code is not a success
        test** -- Phaser exits 0 on fatal keyword errors. Check that the log
        contains a ``TORCHREF:`` line and that the dump parses.
    """
    if not Path(binary).exists():
        raise FileNotFoundError(
            f"patched phaser binary not found at {binary}; build it with the "
            "recipe in the phaser-instrumented-build memo"
        )
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / (kw_path.stem + ".log")
    env = dict(os.environ)
    if dump_path is not None:
        env["PHASER_FRF_DUMP"] = str(Path(dump_path).resolve())

    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(binary)], cwd=str(work), input=kw_path.read_text(),
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        log_path.write_text("TIMEOUT\n")
        rc = -1
    return rc, time.time() - t0, log_path


_LOG_PATTERNS = {
    "lmax": re.compile(r"maximum l value\s+(\d+)"),
    "sampling_deg": re.compile(r"Sampling:\s+([0-9.]+)\s+degrees"),
    "mean_radius_A": re.compile(r"^\s*[0-9.]+\s+([0-9.]+)\s+\d+\s+-?[0-9.]+", re.M),
    "lmax_reso_A": re.compile(r"Elmn with resolution\s+([0-9.]+)"),
    "n_samples": re.compile(r"TORCHREF: wrote (\d+) FRF sample points"),
    "selected_hi": re.compile(
        r"Resolution of Selected Data \(Number\):\s+([0-9.]+)\s+([0-9.]+)\s+\((\d+)\)"
    ),
}


def parse_phaser_log(log_path: Path) -> dict:
    """Pull the parameters Phaser actually used out of a VERBOSE log.

    These are the ground truth for the derivation in :func:`phaser_frf_params`;
    the comparison harness asserts the two agree rather than assuming they do.

    Returns
    -------
    dict
        Keys present only when found: ``lmax``, ``sampling_deg``,
        ``mean_radius_A``, ``lmax_reso_A``, ``n_samples``, ``selected_d_min``,
        ``selected_d_max``, ``selected_n_refl``, ``all_data_to_limit``,
        ``rotation_search_skipped``.
    """
    text = Path(log_path).read_text()
    out: dict = {}
    for key in ("lmax", "n_samples"):
        m = _LOG_PATTERNS[key].search(text)
        if m:
            out[key] = int(m.group(1))
    for key in ("sampling_deg", "lmax_reso_A", "mean_radius_A"):
        m = _LOG_PATTERNS[key].search(text)
        if m:
            out[key] = float(m.group(1))
    m = _LOG_PATTERNS["selected_hi"].search(text)
    if m:
        out["selected_d_min"] = float(m.group(1))
        out["selected_d_max"] = float(m.group(2))
        out["selected_n_refl"] = int(m.group(3))
    out["all_data_to_limit"] = "Elmn with all data to resolution limit" in text
    # The R-factor short-circuit. Present => the search never ran.
    out["rotation_search_skipped"] = "SOLU SET  RF*0" in text
    return out


def load_phaser_map(
    dump_path: Path, *, dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read a ``PHASER_FRF_DUMP`` CSV into our angle convention.

    Angles are returned **exactly as stored**, with no sign change. Phaser
    writes ``euler = (-360*alpha_frac, beta_deg, -360*gamma_frac)``
    (``FastRot.cc:153``), and those stored values are already the Euler angles
    of the grid rotation under ``R = Rz(alpha)Ry(beta)Rz(gamma)`` -- the same
    Edmonds convention we use. Calibrated against a known grid/output pair on
    1DAW this reproduces Phaser's own reported peak to **0.074 deg**.

    Two traps here, both of which cost real time:

    * Negating alpha/gamma to "convert to our convention" is WRONG; they need
      no conversion.
    * The ``FastRot.cc`` comment at the end of ``get_FRF`` claims Phaser assumes
      ``Rz(gamma)Ry(beta)Rz(alpha)``. That does not describe these stored
      values; taking it literally puts the truth peak ~60-130 deg away.

    The stored angles are in the search model's **principal frame**. Use
    ``ROT = axisrot @ R_grid @ PR`` (from the ``.frame`` sidecar) to reach the
    PDB frame our engine works in.

    Returns
    -------
    tuple
        ``(angles_deg, values)`` -- ``(N, 3)`` of (alpha, beta, gamma) as stored,
        and ``(N,)`` rotation-function values, in Phaser's sample order.
    """
    import numpy as np

    raw = np.loadtxt(str(dump_path), delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw[None, :]
    angles = torch.from_numpy(raw[:, 1:4].copy()).to(dtype)
    values = torch.from_numpy(raw[:, 4].copy()).to(dtype)
    return angles, values


def load_phaser_frame(dump_path: Path, *, dtype: torch.dtype = torch.float64) -> dict:
    """Read the ``.frame`` sidecar written next to a map dump.

    Returns ``PR``, ``axisrot`` (3x3 tensors), ``high_order_axis`` (int) and
    ``stats`` (Phaser's own mean/sigma/max/min over the raw sample list, useful
    for checking an externally computed sigma rather than trusting it).
    """
    import numpy as np

    path = Path(str(dump_path) + ".frame")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; the binary must carry the runMR_FRF.cc patch that "
            "dumps PR/axisrot, not only the FastRot.cc map patch"
        )
    out: dict = {}
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split(",")
        vals = [float(x) for x in parts[1:10]]
        if parts[0] in ("PR", "axisrot"):
            out[parts[0]] = torch.tensor(np.array(vals).reshape(3, 3)).to(dtype)
        elif parts[0] == "high_order_axis":
            out["high_order_axis"] = int(vals[0])
        elif parts[0] == "stats_mean_sigma_max_min":
            out["stats"] = dict(zip(("mean", "sigma", "max", "min"), vals[:4]))
    return out


def phaser_sampling_from_dump(angles_deg: torch.Tensor) -> float:
    """Recover the *exact* SO(3) sampling step from a dumped sample list.

    Phaser logs the sampling through ``dtos(SAMPLING,5,2)`` -- two decimals --
    which is far too coarse to rebuild the grid: on 1DAW the rounded 6.23 deg
    gives 53270 sample points against the true 54430. The beta values in the
    dump are full-precision and uniformly spaced, so their step is the exact
    figure.

    Parameters
    ----------
    angles_deg : torch.Tensor, shape (N, 3)
        As returned by :func:`load_phaser_map`.

    Returns
    -------
    float
        Sampling step in degrees.
    """
    betas = torch.unique(angles_deg[:, 1])
    if betas.numel() < 2:
        raise ValueError("need at least two beta sections to infer the step")
    steps = betas[1:] - betas[:-1]
    return float(steps.median())


def phaser_mean_radius_from_sampling(sampling_deg: float, samp_reso_A: float) -> float:
    """Invert Phaser's sampling formula for ``mean_radius()``.

    ``sampling = 2*deg(atan(SAMP_RESO/(4*r)))`` inverts to
    ``r = SAMP_RESO / (4*tan(sampling/2))``. Combined with
    :func:`phaser_sampling_from_dump` this yields Phaser's own radius to full
    precision, which is the reference our :func:`phaser_mean_radius`
    reimplementation should be judged against (it currently runs ~4% high).
    """
    half = math.radians(float(sampling_deg) / 2.0)
    return float(samp_reso_A) / (4.0 * math.tan(half))
