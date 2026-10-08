"""

Bootstrap module to configure threading before importing heavy libraries.

torchref/__init__.py calls ``configure_threading`` itself before importing torch, and
it overwrites OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS. To choose the
thread count, set TORCHREF_NUM_THREADS before importing torchref.

"""

import os


def detect_available_cpus(max_if_not_slurm=4) -> int:
    """Detect actual available CPUs, respecting SLURM and CPU affinity."""

    # 1. Check SLURM first (most reliable on HPC)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)

    # 2. Check CPU affinity
    try:
        return min(len(os.sched_getaffinity(0)), 4)
    except (AttributeError, OSError):
        pass

    # 3. Fallback to os.cpu_count() but cap it sensibly
    return min(os.cpu_count() or 1, 4)


def configure_threading(num_threads: int = None, pin_threads=False) -> int:
    """Configure all threading libraries. Call BEFORE importing torch/numpy."""

    if num_threads is None:
        num_threads = detect_available_cpus()

    n = str(num_threads)

    os.environ["OMP_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n

    if pin_threads:
        # Optional: enable thread pinning
        os.environ["OMP_PROC_BIND"] = "TRUE"
        os.environ["OMP_PLACES"] = "cores"

    return num_threads
