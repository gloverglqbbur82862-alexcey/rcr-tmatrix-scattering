"""Run the lightweight reproducibility pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ensembles", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if not args.skip_validation:
        code = subprocess.call([sys.executable, str(ROOT / "scripts" / "reproduce_validation.py")], cwd=ROOT)
        if code:
            return code
    if not args.skip_ensembles:
        code = subprocess.call([sys.executable, str(ROOT / "scripts" / "reproduce_ensembles.py")], cwd=ROOT)
        if code:
            return code
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "reproduce_figures.py")], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
