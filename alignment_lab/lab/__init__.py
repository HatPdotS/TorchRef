"""Shared library for the alignment lab.

Every diagnostic imports its primitives from here rather than re-deriving them.
The scripts this replaces carried ~37 copies of the rotation generator (in two
mutually incompatible variants), ~30 copies of the benchmark list, ~28 copies of
the CSV writer and ~20 copies of the rank-of-truth computation, several of which
disagreed with each other. One definition each, so two runs are comparable.
"""

from .benchmark import (
    BENCH_PDBS,
    PDB_STEMS,
    REPO_ROOT,
    case_paths,
    load_case,
    rotated_case,
)
from .truth import (
    orbit_rank,
    random_rotation,
    seed_for,
    symmetry_orbit,
)
from .aniso import (
    ARMS as ANISO_ARMS,
    aniso_arm,
    fit_aniso_log_space,
    tensor_report,
)
from .frf import (FRFConfig, FRFResult, merge_peak_lists, patched,
                  run_frf)
from .rescore import ENGINES, RescoreResult, paired_ranks, run_rescore
from .profile import (FRF_STAGES, PeakMemory, calibration_seconds,
                      host_info, stage_timers)
from .results import ResultWriter, append_row, provenance

__all__ = [
    "BENCH_PDBS",
    "PDB_STEMS",
    "REPO_ROOT",
    "case_paths",
    "load_case",
    "rotated_case",
    "orbit_rank",
    "random_rotation",
    "seed_for",
    "symmetry_orbit",
    "ANISO_ARMS",
    "aniso_arm",
    "fit_aniso_log_space",
    "tensor_report",
    "FRFConfig",
    "FRFResult",
    "merge_peak_lists",
    "patched",
    "run_frf",
    "ENGINES",
    "RescoreResult",
    "paired_ranks",
    "run_rescore",
    "FRF_STAGES",
    "PeakMemory",
    "calibration_seconds",
    "host_info",
    "stage_timers",
    "ResultWriter",
    "append_row",
    "provenance",
]
