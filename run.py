"""Entry point: `python3 run.py <command>`.

Exists so the project runs with no PYTHONPATH and no install step. Without it you have
to remember `PYTHONPATH=src python3 -m ragpipe.cli ...`, which is the single most
common way to get started wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragpipe.cli import main  # noqa: E402  (path setup must run first)

if __name__ == "__main__":
    raise SystemExit(main())
