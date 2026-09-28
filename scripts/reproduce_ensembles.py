"""Run the paper's BCCA/BPCA application ensembles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "application" / "run_batch.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["A", "B"], default=None)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-case", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, str(RUNNER),
               "--cases", str(ROOT / "experiments" / "application" / "cases.json"),
               "--seeds", args.seeds,
               "--workers", str(args.workers),
               "--threads-per-case", str(args.threads_per_case)]
    if args.tier:
        command.extend(["--tier", args.tier])
    else:
        command.extend(["--tier", "A"])
    if args.force:
        command.append("--force")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
