"""Console-script entry points for the TorchRef command-line tools.

This module sits outside the ``torchref`` package on purpose: importing anything under
``torchref`` imports torch, and the process-wide settings made here have to be in place
before torch loads its OpenMP runtime. Each entry point applies them and then runs
``torchref.cli.<module>.main``. A library import of ``torchref`` changes none of them.

``python -m torchref.cli.<module>`` bypasses this module and with it the settings.
"""

import os
from importlib import import_module


def configure_process() -> None:
    """Apply the process-wide settings for a TorchRef CLI run.

    Must run before torch is imported, or it has no effect. A variable the user has
    already set is left alone.

    ``OMP_WAIT_POLICY=PASSIVE``: by default torch's idle OpenMP workers spin for several
    milliseconds after every parallel op, on the same cores the CPU kernels'
    own thread pool (``torchref-kernels``) then needs, so each kernel launched shortly
    after a torch op runs at a fraction of its speed. Passive workers sleep at once.
    """
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def _entry(module: str):
    def main():
        configure_process()
        return import_module(f"torchref.cli.{module}").main()

    main.__qualname__ = main.__name__ = module
    main.__doc__ = f"Run ``torchref.cli.{module}.main`` with :func:`configure_process`."
    return main


refine = _entry("refine")
collection_difference_refine = _entry("collection_difference_refine")
simulate_noisy_data = _entry("simulate_noisy_data")
mtz2map = _entry("mtz2map")
validate_ded = _entry("validate_ded")
difference_map = _entry("difference_map")
add_metadata = _entry("add_metadata")
strip_altlocs = _entry("strip_altlocs")
uniform_rfree = _entry("uniform_rfree")
