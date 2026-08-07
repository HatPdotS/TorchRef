"""Shared ``load_inline`` harness for the CPU C++ kernels.

Everything here is cluster-deployment plumbing that must not drift between the two
extensions:

* **POSIX ``lockf`` locking**, not PyTorch's ``FileBaton``. FileBaton is a
  file-existence lock, so a process killed mid-compile leaves it behind and blocks every
  future import; ``fcntl.lockf`` record locks are enforced by the filesystem (so they
  work across NFS/GPFS nodes) and released by the kernel even on SIGKILL.
* **Per-microarchitecture build directory**, keyed on CPU model -- without it a
  ``-march=native`` binary built on one node raises Illegal Instruction on a node with a
  different CPU.
* **ninja on PATH** -- pip-installed ninja lives next to ``sys.executable``, which is not
  on PATH on compute nodes.
* **A C++20-capable compiler**, required by PyTorch C++ extensions. The default
  ``c++`` on an HPC image is often far older than the newer GCCs installed beside it
  (SLES 15 ships GCC 7.5 as ``c++`` with ``g++-13`` in the same directory), so the
  candidates are probed for ``-std=c++20`` rather than assumed.
* **No ``-fopenmp`` on macOS** -- Apple Clang rejects it, so kernels must provide a
  ``std::thread`` fallback under ``#ifndef _OPENMP``.

Failures are returned, never raised: every caller degrades to a slower pure-torch path,
so a missing compiler is a performance problem and not an outage.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from typing import Optional, Tuple

from torch.utils.cpp_extension import load_inline

# Newest first: the first one that accepts -std=c++20 wins.
_CXX_CANDIDATES = (
    [f"/opt/rh/gcc-toolset-{v}/root/usr/bin/g++" for v in ("14", "13", "12")]
    + [f"g++-{v}" for v in ("14", "13", "12")]
    + ["g++", "c++"]
)


def find_cxx() -> Optional[str]:
    """Absolute path to a compiler that accepts ``-std=c++20``, or ``None``.

    A pre-set ``CXX`` is honoured first and is *not* validated -- if the user pointed us
    at a compiler, that choice stands. Otherwise the candidates are tried in order and
    each is actually invoked, because a name like ``g++`` says nothing about its vintage:
    on SLES 15 it is GCC 7.5, which rejects ``-std=c++20`` outright.
    """
    preset = os.environ.get("CXX")
    if preset:
        return preset
    for cand in _CXX_CANDIDATES:
        path = shutil.which(cand)
        if path is None:
            continue
        try:
            subprocess.run(
                [path, "-std=c++20", "-E", "-x", "c++", os.devnull],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return path
    return None


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
    """Compile ``cpp_source`` into an extension module. **Never raises.**

    Parameters
    ----------
    name : str
        Extension name; also the ``PYBIND11_MODULE`` name the source must use.
    cpp_source : str
        Complete translation unit, including its own ``PYBIND11_MODULE`` block.
    extra_cflags : list of str, optional
        Appended after the shared ``-O3 -march=native``. ``-fopenmp`` is added automatically
        everywhere except macOS.

    Returns
    -------
    (module, error)
        ``(module, None)`` on success, ``(None, (message, traceback))`` on any failure.
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
        # PyTorch C++ extensions are C++20; the default c++ may be far older.
        if sys.platform != "darwin":
            cxx = find_cxx()
            if cxx is None:
                return None, ("no C++20-capable compiler found", "")
            os.environ["CXX"] = cxx
            cc = cxx.replace("g++", "gcc")
            if cc != cxx and os.path.isfile(cc):
                os.environ["CC"] = cc

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
