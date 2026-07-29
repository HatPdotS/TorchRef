"""Shared ``load_inline`` harness for the CPU C++ kernels.

Extracted from ``scatter.py`` so the fused sphere splat (``sphere_splat.py``) does
not carry a second copy of it. Everything here is cluster-deployment plumbing that
must not drift between the two extensions:

* **POSIX ``lockf`` locking** instead of PyTorch's ``FileBaton``. FileBaton is a
  file-existence lock, so a process killed mid-compile leaves it behind and blocks
  every future import. ``fcntl.lockf`` record locks are enforced by the filesystem
  (so they work across NFS/GPFS nodes) and released by the kernel on process death,
  even SIGKILL.
* **Per-microarchitecture build directory**, keyed on the CPU model. Without it a
  ``-march=native`` binary built on one cluster node raises Illegal Instruction on
  a node with a different CPU (e.g. AMD vs Intel).
* **ninja on PATH** — pip-installed ninja lives next to ``sys.executable``, which
  is not on PATH on compute nodes.
* **GCC >= 9** via ``/opt/rh/gcc-toolset-*``, required by PyTorch C++ extensions.
* **No ``-fopenmp`` on macOS** — Apple Clang rejects it (no bundled OpenMP
  runtime), so kernels must provide a ``std::thread`` fallback under
  ``#ifndef _OPENMP``.

Failures are returned, never raised: every caller degrades to a slower pure-torch
path, so a missing compiler is a performance problem and not an outage.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional, Tuple

from torch.utils.cpp_extension import load_inline


def cpu_tag() -> str:
    """A tag identifying this CPU's microarchitecture, for the build directory."""
    import platform

    tag = platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    # e.g. "EPYC_7443P" or "Xeon_Gold_6248"
                    return line.split(":")[1].strip().replace(" ", "_")
    except OSError:
        pass
    return tag


def build_extension(
    name: str,
    cpp_source: str,
    extra_cflags: Optional[list] = None,
) -> Tuple[object, Optional[Tuple[str, str]]]:
    """Compile ``cpp_source`` into an extension module.

    Parameters
    ----------
    name : str
        Extension name; also the ``PYBIND11_MODULE`` name the source must use.
    cpp_source : str
        Complete translation unit, including its own ``PYBIND11_MODULE`` block.
    extra_cflags : list of str, optional
        Appended after the shared ``-O3 -march=native``. ``-fopenmp`` is added
        automatically on every platform except macOS.

    Returns
    -------
    (module, error)
        ``(module, None)`` on success, ``(None, (message, traceback))`` on any
        failure. Never raises.
    """
    try:
        import fcntl
    except ImportError:
        # Non-POSIX (Windows): no record locks, so don't attempt a build at all.
        return None, ("fcntl unavailable (non-POSIX platform)", "")

    try:
        # pip-installed ninja sits next to the interpreter, not on PATH.
        bin_dir = os.path.dirname(sys.executable)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")
        # PyTorch C++ extensions need GCC >= 9.
        for toolset in ("14", "13", "12"):
            gcc = f"/opt/rh/gcc-toolset-{toolset}/root/usr/bin/g++"
            if os.path.isfile(gcc):
                os.environ["CXX"] = gcc
                os.environ["CC"] = gcc.replace("g++", "gcc")
                break

        build_dir = os.path.join(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"),
            ),
            f"{name}_{cpu_tag()}",
        )
        os.makedirs(build_dir, exist_ok=True)

        cflags = ["-O3", "-march=native"] + list(extra_cflags or [])
        ldflags: list = []
        if sys.platform != "darwin":  # Apple Clang has no bundled OpenMP runtime
            cflags.append("-fopenmp")
            ldflags.append("-fopenmp")

        lock_fd = os.open(
            os.path.join(build_dir, "compile.lock"), os.O_CREAT | os.O_RDWR
        )
        try:
            fcntl.lockf(lock_fd, fcntl.LOCK_EX)
            # Clear a stale PyTorch FileBaton lock left by a killed process.
            try:
                os.unlink(os.path.join(build_dir, "lock"))
            except FileNotFoundError:
                pass
            module = load_inline(
                name=name,
                cpp_sources=[cpp_source],
                extra_cflags=cflags,
                extra_ldflags=ldflags,
                build_directory=build_dir,
                verbose=False,
            )
        finally:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    except Exception as e:  # noqa: BLE001 - any build failure degrades gracefully
        return None, (f"{type(e).__name__}: {e}", traceback.format_exc())

    return module, None
