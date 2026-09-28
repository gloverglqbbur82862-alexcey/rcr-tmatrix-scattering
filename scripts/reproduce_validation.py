"""Run the frozen MSTM validation cases.

This wrapper delegates to the validation runner and keeps paths relative to
the repository root. A local MSTM executable is required for --run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "validation" / "run_mstm_validation.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="also execute MSTM")
    parser.add_argument("--mstm-bin", help="path to a locally built MSTM executable")
    args = parser.parse_args()
    command = [sys.executable, str(RUNNER)]
    if args.run:
        command.append("--run")
    if args.mstm_bin:
        command.extend(["--mstm-bin", args.mstm_bin])
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
