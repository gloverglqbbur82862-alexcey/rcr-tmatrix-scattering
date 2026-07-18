#!/usr/bin/env python3
"""Batch orchestrator for multiple scattering cases with controlled thread budgets."""

import os
import sys
import json
import time
import argparse
import subprocess
import concurrent.futures
from pathlib import Path


def run_single_case(case_id, cases_path, out_path, env, dry_run=False):
    """Run a single run_case.py command as a subprocess."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_case.py"),
        "--cases", str(cases_path),
        "--case-id", case_id,
        "--out", str(out_path)
    ]
    
    if dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        print(f"[DRY-RUN] Environment: OMP_NUM_THREADS={env.get('OMP_NUM_THREADS')}, NUMBA_NUM_THREADS={env.get('NUMBA_NUM_THREADS')}")
        return case_id, "skipped_dry_run", 0.0, "", ""

    t0 = time.time()
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        t_elapsed = time.time() - t0
        return case_id, "passed", t_elapsed, res.stdout, res.stderr
    except subprocess.CalledProcessError as exc:
        t_elapsed = time.time() - t0
        print(f"Error executing case {case_id}: Code {exc.returncode}", file=sys.stderr)
        print(f"Stdout:\n{exc.stdout}", file=sys.stderr)
        print(f"Stderr:\n{exc.stderr}", file=sys.stderr)
        return case_id, "failed", t_elapsed, exc.stdout, exc.stderr


def main():
    parser = argparse.ArgumentParser(description="Batch orchestrator for scattering sweeps")
    parser.add_argument("--cases", default="experiments/jqsrt_application/cases.json", help="Path to cases.json")
    parser.add_argument("--tier", choices=["A", "B", "C"], help="Run all cases in a specific tier")
    parser.add_argument("--case-id", help="Run a single specific case ID")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent worker processes")
    parser.add_argument("--threads-per-case", type=int, default=4, help="OMP_NUM_THREADS per case worker")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if result JSON exists")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")
    args = parser.parse_args()

    # Load cases
    cases_path = Path(args.cases).resolve()
    if not cases_path.exists():
        print(f"Error: {cases_path} not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(cases_path, "r") as f:
        cases = json.load(f)

    # Filter cases based on tier or case_id
    selected_cases = []
    if args.case_id:
        selected_cases = [c for c in cases if c["case_id"] == args.case_id]
        if not selected_cases:
            print(f"Error: Case ID '{args.case_id}' not found.", file=sys.stderr)
            sys.exit(1)
    elif args.tier:
        selected_cases = [c for c in cases if c.get("tier") == args.tier]
        if not selected_cases:
            print(f"Error: No cases found for tier '{args.tier}'.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: You must specify --tier (A, B) or --case-id.", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine execution task list (checking for existing results)
    run_list = []
    skipped_list = []
    for case in selected_cases:
        case_id = case["case_id"]
        out_path = results_dir / f"{case_id}.json"
        if out_path.exists() and not args.force:
            skipped_list.append(case_id)
        else:
            run_list.append((case_id, out_path))

    print(f"Batch orchestrator planned to run: {len(run_list)} cases (Skipped existing: {len(skipped_list)})")
    for skip_id in skipped_list:
        print(f"  [SKIPPED] {skip_id} (result exists)")

    # Prepare environments for workers
    worker_env = os.environ.copy()
    worker_env["OMP_NUM_THREADS"] = str(args.threads_per_case)
    worker_env["NUMBA_NUM_THREADS"] = "1"
    worker_env["MKL_NUM_THREADS"] = str(args.threads_per_case)
    worker_env["OPENBLAS_NUM_THREADS"] = str(args.threads_per_case)
    worker_env["VECLIB_MAXIMUM_THREADS"] = str(args.threads_per_case)
    worker_env["NUMEXPR_NUM_THREADS"] = str(args.threads_per_case)

    # Load existing manifest if it exists
    manifest_path = results_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            pass

    if "cases" not in manifest:
        manifest["cases"] = {}

    manifest["run_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["workers"] = args.workers
    manifest["threads_per_case"] = args.threads_per_case

    # Set up ThreadPoolExecutor for subprocess concurrency
    t_start = time.time()
    if run_list:
        print(f"Starting execution with {args.workers} workers and {args.threads_per_case} threads per case...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_single_case, 
                    case_id, 
                    cases_path, 
                    out_path, 
                    worker_env, 
                    dry_run=args.dry_run
                ): case_id for case_id, out_path in run_list
            }
            
            for future in concurrent.futures.as_completed(futures):
                case_id = futures[future]
                try:
                    cid, status, t_elapsed, stdout, stderr = future.result()
                    manifest["cases"][case_id] = {
                        "status": status,
                        "time_total_s": t_elapsed,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    if status == "passed":
                        print(f"  [PASSED] {case_id} in {t_elapsed:.2f}s")
                    elif status == "skipped_dry_run":
                        pass
                    else:
                        print(f"  [FAILED] {case_id} in {t_elapsed:.2f}s")
                except Exception as exc:
                    print(f"  [ERROR] {case_id} raised exception: {exc}", file=sys.stderr)
                    manifest["cases"][case_id] = {
                        "status": "exception",
                        "error": str(exc),
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }

    t_total = time.time() - t_start
    manifest["total_wall_clock_time_s"] = t_total
    print(f"Batch execution completed in {t_total:.2f}s.")

    # Mark skipped cases in manifest
    for skip_id in skipped_list:
        if skip_id not in manifest["cases"]:
            manifest["cases"][skip_id] = {
                "status": "passed_cached",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

    if not args.dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Run manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
