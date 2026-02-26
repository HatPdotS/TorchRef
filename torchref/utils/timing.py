"""CLI timing utility — register once at the start of main()."""

import atexit
import time


def register_timing():
    """Register an atexit handler that prints wall and CPU time on exit.

    Call at the top of ``main()`` in CLI scripts::

        from torchref.utils.timing import register_timing

        def main():
            register_timing()
            ...
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    def _report():
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        print(f"\nTiming: {wall:.1f}s wall, {cpu:.1f}s CPU")

    atexit.register(_report)
