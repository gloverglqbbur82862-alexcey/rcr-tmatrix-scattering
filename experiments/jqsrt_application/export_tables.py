#!/usr/bin/env python3
"""Export simulation results to CSV tables and generate benchmark environment description."""

import os
import sys
import glob
import json
import csv
import multiprocessing
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export simulation results to CSV tables")
    parser.add_argument("--results", default="experiments/jqsrt_application/results", help="Directory containing result JSON files")
    parser.add_argument("--out", default="writing/jqsrt_application", help="Output directory for tables and md")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_pattern = os.path.join(results_dir, "*.json")
    result_files = glob.glob(json_pattern)
    result_files = [f for f in result_files if "run_manifest" not in os.path.basename(f)]

    if not result_files:
        print("Error: No result JSON files found to export.", file=sys.stderr)
        sys.exit(1)

    print(f"Exporting tables from {len(result_files)} result files...")

    # Load results
    results_list = []
    for filepath in result_files:
        try:
            with open(filepath, "r") as f:
                results_list.append(json.load(f))
        except Exception as exc:
            print(f"Error loading {filepath}: {exc}", file=sys.stderr)

    # Sort results for clean table output
    results_list.sort(key=lambda x: (x["input_parameters"]["tier"], x["input_parameters"]["family"], x["input_parameters"]["n_monomers"], x["input_parameters"]["wavelength_nm"]))

    # 1. Export table_global_observables.csv
    global_csv_path = out_dir / "table_global_observables.csv"
    with open(global_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case ID", "Tier", "Family", "N", "Radius (nm)", "Wavelength (nm)", 
            "Qext", "Qsca", "Qabs", "SSA", "g", "Wall Time (s)"
        ])
        for r in results_list:
            ip = r["input_parameters"]
            obs = r["observables"]
            tm = r["timing_metadata"]
            writer.writerow([
                r["case_id"], ip["tier"], ip["family"], ip["n_monomers"], ip["radius_nm"], ip["wavelength_nm"],
                f"{obs['Qext']:.6f}", f"{obs['Qsca']:.6f}", f"{obs['Qabs']:.6f}", f"{obs['SSA']:.6f}", f"{obs['g']:.6f}",
                f"{tm['time_total_s']:.2f}"
            ])
    print(f"Saved global observables table to {global_csv_path}")

    # 2. Export table_mueller_selected_angles.csv
    # Selected angles: 0, 30, 60, 90, 120, 150, 180 degrees
    selected_angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    mueller_csv_path = out_dir / "table_mueller_selected_angles.csv"
    
    with open(mueller_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case ID", "Family", "N", "Wavelength (nm)", "Angle (deg)",
            "F11", "F12_over_F11", "F22_over_F11", "F33_over_F11", "F34_over_F11", "F44_over_F11"
        ])
        
        for r in results_list:
            ip = r["input_parameters"]
            theta_deg = r["theta_deg"]
            mueller = r["mueller"]
            
            # Map selected angles to their indices in the results (ref theta_deg)
            for target_angle in selected_angles:
                # Find closest index
                idx = min(range(len(theta_deg)), key=lambda i: abs(theta_deg[i] - target_angle))
                # Confirm we are within small tolerance
                if abs(theta_deg[idx] - target_angle) > 0.1:
                    continue
                    
                writer.writerow([
                    r["case_id"], ip["family"], ip["n_monomers"], ip["wavelength_nm"], f"{theta_deg[idx]:.1f}",
                    f"{mueller['F11'][idx]:.6e}", f"{mueller['F12_over_F11'][idx]:.6f}", f"{mueller['F22_over_F11'][idx]:.6f}",
                    f"{mueller['F33_over_F11'][idx]:.6f}", f"{mueller['F34_over_F11'][idx]:.6f}", f"{mueller['F44_over_F11'][idx]:.6f}"
                ])
    print(f"Saved Mueller selected angles table to {mueller_csv_path}")

    # 3. Export methods_benchmark_environment.md
    methods_path = out_dir / "methods_benchmark_environment.md"
    
    # Extract metadata from one of the results
    env = results_list[0]["environment_metadata"]
    
    # The manifest is overwritten by each batch invocation, so it records the
    # latest segment rather than the full Tier A/B execution plan.
    manifest_path = results_dir / "run_manifest.json"
    manifest_note = "Unavailable"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
                manifest_note = (
                    f"Last batch invocation used workers={manifest.get('workers', 'Unknown')}, "
                    f"threads_per_case={manifest.get('threads_per_case', 'Unknown')}, "
                    f"wall_clock={manifest.get('total_wall_clock_time_s', 0.0):.2f} s."
                )
        except Exception:
            pass

    content = f"""# Benchmark and Simulation Environment Specifications

The simulation runs for the JQSRT / Optics Express application package were executed on the reference workstation high-core benchmark processor.

## Hardware Platform Specifications

- **Logical Processors**: {multiprocessing.cpu_count()} cores
- **OS Platform**: {env.get('os', 'Unknown')}
- **Hostname**: {env.get('hostname', 'Unknown')}

## Software & Library Environment

- **Python Version**: {env.get('python_version', 'Unknown')}
- **NumPy Version**: {env.get('numpy_version', 'Unknown')}
- **SciPy Version**: {env.get('scipy_version', 'Unknown')}
- **Numba Version**: {env.get('numba_version', 'Unknown')}

## Batch Runtime Configurations

- **Tier A Case Parallel Workers**: `4`
- **Tier A Subprocess Threads (OMP_NUM_THREADS)**: `8`
- **Tier B N <= 32 Case Parallel Workers**: `1`
- **Tier B N <= 32 Subprocess Threads (OMP_NUM_THREADS)**: `8`
- **Tier B N = 64 Case Parallel Workers**: `1`
- **Tier B N = 64 Subprocess Threads (OMP_NUM_THREADS)**: `24`
- **Subprocess JIT Threads (NUMBA_NUM_THREADS)**: `1` (Locked to single-thread context inside Numba to prevent concurrent write loops)
- **Manifest Note**: {manifest_note}

## Execution Commands

### Tier A (Wavelength Sweep):
```bash
./.venv/bin/python experiments/jqsrt_application/run_batch.py \\
  --cases experiments/jqsrt_application/cases.json \\
  --tier A \\
  --workers 4 \\
  --threads-per-case 8
```

### Tier B (Size Scaling Sweep):
```bash
./.venv/bin/python experiments/jqsrt_application/run_batch.py \\
  --cases experiments/jqsrt_application/cases.json \\
  --case-id bcca_n8_w550_seed42 --workers 1 --threads-per-case 8
./.venv/bin/python experiments/jqsrt_application/run_batch.py \\
  --cases experiments/jqsrt_application/cases.json \\
  --case-id bpca_n8_w550_seed42 --workers 1 --threads-per-case 8
./.venv/bin/python experiments/jqsrt_application/run_batch.py \\
  --cases experiments/jqsrt_application/cases.json \\
  --case-id bcca_n32_w550_seed42 --workers 1 --threads-per-case 8
./.venv/bin/python experiments/jqsrt_application/run_batch.py \\
  --cases experiments/jqsrt_application/cases.json \\
  --case-id bpca_n32_w550_seed42 --workers 1 --threads-per-case 8
./.venv/bin/python experiments/jqsrt_application/run_batch.py \\
  --cases experiments/jqsrt_application/cases.json \\
  --case-id bcca_n64_w550_seed42 --workers 1 --threads-per-case 24
./.venv/bin/python experiments/jqsrt_application/run_batch.py \\
  --cases experiments/jqsrt_application/cases.json \\
  --case-id bpca_n64_w550_seed42 --workers 1 --threads-per-case 24
```
"""
    with open(methods_path, "w") as f:
        f.write(content)
    print(f"Saved benchmark environment report to {methods_path}")


if __name__ == "__main__":
    main()
