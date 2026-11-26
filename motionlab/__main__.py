"""Module entry point for `python -m motionlab`."""

from __future__ import annotations

import sys

from motionlab.cli import main

if __name__ == "__main__":
    sys.exit(main())
