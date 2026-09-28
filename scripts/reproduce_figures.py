"""Export paper tables and generate figures from application results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "experiments" / "application"
RESULTS = APP / "results" / "seed0-9"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    args = parser.parse_args()
    export = [sys.executable, str(APP / "export_tables.py"),
              "--results", str(RESULTS), "--out", str(ROOT / "data" / "results"),
              "--seeds", args.seeds]
    plot = [sys.executable, str(APP / "plot_figures.py"),
            "--results", str(RESULTS), "--out", str(ROOT / "figures" / "generated")]
    return_code = subprocess.call(export, cwd=ROOT)
    if return_code:
        return return_code
    return subprocess.call(plot, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
