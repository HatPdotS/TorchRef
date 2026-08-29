"""Timing, memory and node-calibration primitives for the benchmark harness.

Three measurements, three different hazards:

**Time.** Wall clock on a shared cluster is a measurement of the cluster, not of
the code. Two engines timed on different nodes, or on the same node under
different neighbours, have been seen to differ by more than the effect being
looked for. So every timing row carries :func:`calibration_seconds` -- a fixed
workload run in the same process -- and the node's identity. Compare normalised
times, or compare only within a node.

**Memory.** Peak RSS is a high-water mark: it never falls, so several
measurements in one process all report the largest. :class:`PeakMemory` samples
``/proc/self/statm`` on a thread so each window gets its own peak, and reports
the delta over the value at window entry. Two caveats, both reported rather than
hidden: a sampler can miss a spike shorter than its interval, and glibc does not
always return freed pages, so a later window in the same process can look
cheaper than it is. For absolute numbers, run one measurement per process and
read ``VmHWM``.

**Stages.** A stage that cannot be resolved is *reported*, not skipped: an absent
row and a free stage look identical in a table.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

#: ``(module path, attribute)`` for each stage worth timing separately, coarse
#: to fine.
#:
#: The module named here is where the call is *resolved*, not where the function
#: is defined. ``frf.api`` binds ``bessel_sh_expand`` and friends into its own
#: namespace with ``from .data_mr import ...`` at import time, so replacing
#: ``data_mr.bessel_sh_expand`` leaves api's reference untouched and the stage
#: silently registers zero calls -- which reads as "that stage is free". Getting
#: this wrong left 85% of the runtime unattributed.
FRF_STAGES: Tuple[Tuple[str, str], ...] = (
    ("torchref.experimental.alignment.frf.dense_calc", "dense_calc_via_box"),
    # Patched on its DEFINING module, not on `api`: it is reached through
    # `FrenchWilsonE._compute`, which imports it inside the method body, so the
    # lookup happens at call time and `api` never holds a reference at all.
    ("torchref.experimental.alignment.frf.french_wilson",
     "french_wilson_preprocess"),
    ("torchref.experimental.alignment.frf.api", "bessel_sh_expand"),
    ("torchref.experimental.alignment.frf.api", "cross_correlate_xi"),
    ("torchref.experimental.alignment.frf.api", "evaluate_rotation_function"),
    ("torchref.experimental.alignment.frf.api", "find_rotation_peaks"),
    ("torchref.experimental.alignment.frf.sitelist_ang",
     "wigner_contraction_per_beta"),
    ("torchref.experimental.alignment.frf.sitelist_ang",
     "build_dense_map_per_beta"),
    ("torchref.experimental.alignment.frf.data_mr", "spherical_bessel_table"),
    # Added once the named stages stopped accounting for the run: after the obs
    # chain moved to the unique set, French-Wilson fell from 29.9% to 3.6% and
    # the unattributed remainder became the second largest item at 29%. These are
    # the rest of the per-reflection work, patched where each call RESOLVES --
    # `api` imports the preprocessing names at module top, `rotation_search`
    # imports `apply_overall_anisotropy` from `sh` at module top, and
    # `fit_relative_wilson_b` is imported inside `search_peaks` so it has to be
    # patched on the defining module.
    # No `wilson_normalise` row any more. The observed-side normalisation now
    # arrives as the `e_convention` CLASS and is called through a parameter, so
    # there is no module attribute to patch -- and a row that registers zero
    # calls is worse than no row, because it reads as "that stage is free".
    ("torchref.experimental.alignment.frf.api", "eterm_sigma_a"),
    ("torchref.experimental.alignment.frf.api", "build_lerf1_intensity"),
    ("torchref.experimental.alignment.frf.api", "apply_shell_variance_weights"),
    ("torchref.experimental.alignment.frf.api", "detect_zsymm"),
    ("torchref.experimental.alignment.frf.preprocessing",
     "fit_relative_wilson_b"),
    ("torchref.experimental.alignment.rotation_search",
     "apply_overall_anisotropy"),
)

#: Which stage each nested stage sits inside. A parent's time *includes* its
#: children, so summing the raw rows double-counts -- the harness subtracts to
#: report exclusive time as well, because otherwise the table invites optimising
#: the wrong thing.
FRF_STAGE_PARENTS = {
    "build_dense_map_per_beta": "evaluate_rotation_function",
    "wigner_contraction_per_beta": "build_dense_map_per_beta",
    "spherical_bessel_table": "bessel_sh_expand",
}


def exclusive_times(totals):
    """Per-stage time with nested children subtracted out.

    Parameters
    ----------
    totals : mapping
        Stage name -> inclusive seconds, as :func:`stage_timers` yields.

    Returns
    -------
    dict
        Stage name -> exclusive seconds. These sum without double counting.
    """
    excl = dict(totals)
    for child, parent in FRF_STAGE_PARENTS.items():
        if child in totals and parent in excl:
            excl[parent] = excl[parent] - totals[child]
    return excl


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_bytes() -> int:
    """Current resident set size, in bytes."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return 0


def vm_hwm_bytes() -> int:
    """Process peak resident set size since start, in bytes. 0 if unavailable.

    Monotonic over the life of the process, so it answers "how much did this
    process ever need", not "how much did this call need".
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class PeakMemory:
    """Sample RSS on a thread and report the peak inside a window.

    Parameters
    ----------
    interval_s : float, optional
        Sampling period. Default 0.02 s. Anything shorter than this that the
        code allocates and frees again is invisible; ``missed_window_risk``
        records the interval so a reader can judge that.
    """

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak = 0
        self._samples = 0

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            r = rss_bytes()
            self._samples += 1
            if r > self._peak:
                self._peak = r

    @contextmanager
    def window(self):
        """Measure the peak RSS over the body, as a delta and an absolute."""
        baseline = rss_bytes()
        self._peak = baseline
        self._samples = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        out: Dict[str, float] = {}
        try:
            yield out
        finally:
            self._stop.set()
            self._thread.join(timeout=5.0)
            peak = max(self._peak, rss_bytes())
            out.update(
                rss_baseline_mb=round(baseline / 1e6, 1),
                rss_peak_mb=round(peak / 1e6, 1),
                rss_delta_mb=round((peak - baseline) / 1e6, 1),
                rss_samples=self._samples,
                rss_sample_interval_s=self.interval_s,
                vm_hwm_mb=round(vm_hwm_bytes() / 1e6, 1),
            )


@contextmanager
def stage_timers(stages: Sequence[Tuple[str, str]] = FRF_STAGES):
    """Time each stage in ``stages`` for the duration of the body.

    Yields ``(totals, counts, unresolved)``. Every resolved stage is registered
    at zero, so a stage that was instrumented but never called still shows up
    with 0 calls -- distinguishable from one that is simply fast.
    """
    from contextlib import ExitStack

    from .frf import patched

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    unresolved: List[str] = []

    def make(orig, key):
        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                totals[key] += time.perf_counter() - t0
                counts[key] += 1
        return timed

    with ExitStack() as stack:
        for mod_path, attr in stages:
            try:
                mod = importlib.import_module(mod_path)
                original = getattr(mod, attr)
            except (ImportError, AttributeError):
                unresolved.append(f"{mod_path.rsplit('.', 1)[-1]}.{attr}")
                continue
            totals.setdefault(attr, 0.0)
            counts.setdefault(attr, 0)
            stack.enter_context(patched(mod, attr, make(original, attr)))
        yield totals, counts, unresolved


def calibration_seconds(repeats: int = 3) -> float:
    """Time a fixed workload, to normalise wall clock across nodes.

    Exercises the two kernels the rotation function spends its time in: a
    complex einsum contraction and a batched FFT, both float64. Fixed shapes and
    a fixed seed, so the only thing that varies is the machine and its
    neighbours. Returns the best of ``repeats`` -- the least contended sample.
    """
    import torch

    g = torch.Generator().manual_seed(0)
    a = torch.randn(64, 96, 96, dtype=torch.float64, generator=g)
    b = torch.randn(64, 96, 96, dtype=torch.float64, generator=g)
    x = torch.complex(a, b)
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        torch.einsum("nij,njk->nik", x, x)
        torch.fft.ifft2(x)
        best = min(best, time.perf_counter() - t0)
    return best


def host_info() -> Dict[str, object]:
    """Node identity and thread configuration, for every benchmark row."""
    import platform

    import torch

    model = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "host": platform.node(),
        "cpu_model": model,
        "torch_threads": torch.get_num_threads(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_exclusive": os.environ.get("SLURM_JOB_NUM_NODES", ""),
    }


__all__ = ["FRF_STAGES", "FRF_STAGE_PARENTS", "PeakMemory",
           "calibration_seconds", "exclusive_times", "host_info",
           "rss_bytes", "stage_timers", "vm_hwm_bytes"]
