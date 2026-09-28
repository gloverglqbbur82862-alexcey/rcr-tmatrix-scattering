#!/usr/bin/env python3
"""Export ensemble CSV tables and generate benchmark environment description."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np

from ensemble_summary import group_results, load_results, parse_seed_spec, scalar_observable_summary


def format_value(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def filter_results_by_seed_spec(results_list, seed_spec: str):
    seed_filter = set(parse_seed_spec(seed_spec))
    return [r for r in results_list if r["input_parameters"]["seed"] in seed_filter]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export simulation results to CSV tables")
    default_results = Path(__file__).parent / "results" / "seed0-9"
    parser.add_argument("--results", default=str(default_results), help="Directory containing result JSON files")
    parser.add_argument("--out", default="data/results", help="Output directory for tables and md")
    parser.add_argument("--seeds", default="0-9", help="Seed set to include in the exported ensemble tables")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_list = filter_results_by_seed_spec(load_results(results_dir), args.seeds)
    if not results_list:
        print("Error: No result JSON files found to export.", file=sys.stderr)
        sys.exit(1)

    results_list.sort(key=lambda x: (x["input_parameters"]["tier"], x["input_parameters"]["family"], x["input_parameters"]["n_monomers"], x["input_parameters"]["wavelength_nm"], x["input_parameters"]["seed"]))
    grouped = group_results(results_list)

    detailed_csv_path = out_dir / "table_global_observables_by_seed.csv"
    with detailed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case ID",
            "Family",
            "N",
            "Radius (nm)",
            "Wavelength (nm)",
            "Seed",
            "Qext",
            "Qsca",
            "Qabs",
            "SSA",
            "g",
            "Wall Time (s)",
        ])
        for r in results_list:
            ip = r["input_parameters"]
            obs = r["observables"]
            tm = r["timing_metadata"]
            writer.writerow([
                r["case_id"],
                ip["family"],
                ip["n_monomers"],
                ip["radius_nm"],
                ip["wavelength_nm"],
                ip["seed"],
                format_value(obs["Qext"]),
                format_value(obs["Qsca"]),
                format_value(obs["Qabs"]),
                format_value(obs["SSA"]),
                format_value(obs["g"]),
                format_value(tm["time_total_s"], 3),
            ])
    print(f"Saved detailed per-seed table to {detailed_csv_path}")

    ensemble_csv_path = out_dir / "table_global_observables_ensemble.csv"
    with ensemble_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Family",
            "N",
            "Radius (nm)",
            "Wavelength (nm)",
            "Seed Count",
            "Qext Mean",
            "Qext Std",
            "Qsca Mean",
            "Qsca Std",
            "Qabs Mean",
            "Qabs Std",
            "SSA Mean",
            "SSA Std",
            "g Mean",
            "g Std",
            "Wall Time Median (s)",
        ])
        for key in sorted(grouped.keys(), key=lambda item: (item[0], item[1], item[3], item[2], item[7])):
            group = grouped[key]
            ip = group[0]["input_parameters"]
            summary = scalar_observable_summary(group)
            writer.writerow([
                ip["family"],
                ip["n_monomers"],
                ip["radius_nm"],
                ip["wavelength_nm"],
                int(summary["seed_count"]),
                format_value(summary["Qext_mean"]),
                format_value(summary["Qext_std"]),
                format_value(summary["Qsca_mean"]),
                format_value(summary["Qsca_std"]),
                format_value(summary["Qabs_mean"]),
                format_value(summary["Qabs_std"]),
                format_value(summary["SSA_mean"]),
                format_value(summary["SSA_std"]),
                format_value(summary["g_mean"]),
                format_value(summary["g_std"]),
                format_value(summary["time_total_s_median"], 3),
            ])
    print(f"Saved ensemble observables table to {ensemble_csv_path}")

    mueller_csv_path = out_dir / "table_mueller_selected_angles_by_seed.csv"
    selected_angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    with mueller_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case ID",
            "Family",
            "N",
            "Wavelength (nm)",
            "Seed",
            "Angle (deg)",
            "F11",
            "F12_over_F11",
            "F22_over_F11",
            "F33_over_F11",
            "F34_over_F11",
            "F44_over_F11",
        ])
        for r in results_list:
            ip = r["input_parameters"]
            theta_deg = r["theta_deg"]
            mueller = r["mueller"]
            for target_angle in selected_angles:
                idx = min(range(len(theta_deg)), key=lambda i: abs(theta_deg[i] - target_angle))
                if abs(theta_deg[idx] - target_angle) > 0.1:
                    continue
                writer.writerow([
                    r["case_id"],
                    ip["family"],
                    ip["n_monomers"],
                    ip["wavelength_nm"],
                    ip["seed"],
                    f"{theta_deg[idx]:.1f}",
                    f"{mueller['F11'][idx]:.6e}",
                    f"{mueller['F12_over_F11'][idx]:.6f}",
                    f"{mueller['F22_over_F11'][idx]:.6f}",
                    f"{mueller['F33_over_F11'][idx]:.6f}",
                    f"{mueller['F34_over_F11'][idx]:.6f}",
                    f"{mueller['F44_over_F11'][idx]:.6f}",
                ])
    print(f"Saved per-seed Mueller table to {mueller_csv_path}")

    methods_path = out_dir / "methods_benchmark_environment.md"
    env = results_list[0]["environment_metadata"]
    manifest_path = results_dir / "run_manifest.json"
    manifest_note = "Unavailable"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
                manifest_note = (
                    f"Last batch invocation used workers={manifest.get('workers', 'Unknown')}, "
                    f"threads_per_case={manifest.get('threads_per_case', 'Unknown')}, "
                    f"wall_clock={manifest.get('total_wall_clock_time_s', 0.0):.2f} s."
                )
        except Exception:
            pass

    content = f"""# Benchmark and Simulation Environment Specifications

The simulation runs for the JQSRT / Optics Express application package were executed on the DR-X high-core benchmark processor.

## Hardware Platform Specifications

- **Logical Processors**: {multiprocessing.cpu_count()} cores
- **OS Platform**: {env.get('os', 'Unknown')}
- **Hostname**: {env.get('hostname', 'Unknown')}

## Software & Library Environment

- **Python Version**: {env.get('python_version', 'Unknown')}
- **NumPy Version**: {env.get('numpy_version', 'Unknown')}
- **SciPy Version**: {env.get('scipy_version', 'Unknown')}
- **Numba Version**: {env.get('numba_version', 'Unknown')}

## Ensemble Runtime Configurations

- **Seed Ensemble**: `0-9`
- **Formal Repeats per Case**: `10`
- **Manifest Note**: {manifest_note}

## Output Files

- `table_global_observables_by_seed.csv`
- `table_global_observables_ensemble.csv`
- `table_mueller_selected_angles_by_seed.csv`

"""
    with methods_path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved benchmark environment report to {methods_path}")


if __name__ == "__main__":
    main()
