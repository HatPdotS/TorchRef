"""Run the difference-map CLI through its phased-difference-map alias."""

from torchref.cli.difference_map import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
