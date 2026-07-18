"""Remote-side entrypoint for RCR Workbench solver jobs.

This script runs inside the project checkout on a remote workstation. It reads a
serialized CaseConfig, executes the regular workbench runner with a local solver
realization, and writes a serialized SolverResult for the local UI to load.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.rcr_workbench.models import CaseConfig, SolverResult
from apps.rcr_workbench.runner import run_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RCR Workbench case on a remote workstation.")
    parser.add_argument("--case-json", required=True, help="Path to serialized CaseConfig JSON.")
    parser.add_argument("--output-json", required=True, help="Path where serialized SolverResult JSON is written.")
    parser.add_argument("--local-realization", required=True, help="Local solver realization to execute on the remote host.")
    parser.add_argument("--public-realization", required=True, help="Public solver realization reported back to the UI.")
    parser.add_argument("--host-label", default="remote_workstation", help="Public host label to store in metadata.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_path = Path(args.case_json)
    output_path = Path(args.output_json)

    case = CaseConfig.model_validate_json(case_path.read_text(encoding="utf-8"))
    case.solver.realization = args.local_realization
    case.solver.demo_mode = args.local_realization == "demo"

    t0 = time.time()
    result = run_case(case)

    metadata = dict(result.metadata)
    metadata.update(
        {
            "solver_realization": args.public_realization,
            "remote_local_realization": args.local_realization,
            "remote_execution": True,
            "remote_host_label": args.host_label,
            "remote_platform": platform.platform(),
            "remote_python": sys.version.split()[0],
            "remote_wall_s": time.time() - t0,
            "remote_pid": os.getpid(),
        }
    )

    result = SolverResult(
        qext=result.qext,
        qsca=result.qsca,
        qabs=result.qabs,
        ssa=result.ssa,
        g=result.g,
        angles_deg=result.angles_deg,
        phase_matrix=result.phase_matrix,
        timing=result.timing,
        metadata=metadata,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote remote solver result to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
