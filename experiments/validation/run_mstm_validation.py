#!/usr/bin/env python3
"""Freeze MSTM v4.0 validation inputs, outputs, and parsed parity data.

This script generates BCCA soot-aggregate MSTM input files for the canonical
validation cases recorded in state.yaml. With --run, it runs the MSTM binary and
stores raw output files under experiments/mstm_validation/outputs/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from geometry.geometry import build_bcca


R_MONOMER_NM = 25.0
WAVELENGTH_NM = 550.0
K_MEDIUM = 2.0 * np.pi / WAVELENGTH_NM
R_DIMLESS = R_MONOMER_NM * K_MEDIUM
M_COMPLEX = 1.95 + 0.79j
SEED = 42
N_VALUES = [4, 8, 16, 32, 64, 128]

STATE_PARITY = {
    4: {"mstm": 0.63154, "rcr": 0.6315419685, "diff_pct": +0.0003},
    8: {"mstm": 0.86602, "rcr": 0.8660198540, "diff_pct": -0.00002},
    16: {"mstm": 1.16050, "rcr": 1.1605109947, "diff_pct": +0.0009},
    32: {"mstm": 1.48500, "rcr": 1.4849924375, "diff_pct": -0.0005},
    64: {"mstm": 1.91560, "rcr": 1.9155928712, "diff_pct": -0.0004},
    128: {"mstm": 2.47330, "rcr": 2.4732545401, "diff_pct": -0.0018},
}


def write_mstm_input(inp_path: Path, out_rel: str, positions_dimless: np.ndarray) -> None:
    lines = [
        "output_file",
        out_rel,
        "number_spheres",
        str(len(positions_dimless)),
        "sphere_data",
    ]
    for pos in positions_dimless:
        lines.append(
            f"{pos[0]:.10f},{pos[1]:.10f},{pos[2]:.10f},"
            f"{R_DIMLESS:.10f},({M_COMPLEX.real:.8f},{M_COMPLEX.imag:.8f})"
        )
    lines.extend(
        [
            "end_of_sphere_data",
            "length_scale_factor",
            "1.d0",
            "incident_beta_deg",
            "0.d0",
            "incident_alpha_deg",
            "0.0d0",
            "random_orientation",
            "t",
            "calculate_scattering_matrix",
            "t",
            "calculate_near_field",
            "f",
            "solution_epsilon",
            "1.d-12",
            "translation_epsilon",
            "1.d-12",
            "end_of_options",
        ]
    )
    inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_mstm_output(out_path: Path) -> dict[str, float | int | None]:
    qext = None
    qsca = None
    t_matrix_order = None
    if not out_path.exists():
        return {"qext": None, "qsca": None, "t_matrix_order": None}

    lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        ll = line.lower().strip()
        if "estimated t matrix order" in ll and idx + 1 < len(lines):
            try:
                t_matrix_order = int(lines[idx + 1].strip().split()[0])
            except Exception:
                pass
        if "t matrix order" in ll and "estimated" not in ll and idx + 1 < len(lines):
            try:
                t_matrix_order = int(lines[idx + 1].strip().split()[0])
            except Exception:
                pass
        if "total extinction" in ll and "efficiencies" in ll and idx + 1 < len(lines):
            parts = lines[idx + 1].strip().split()
            try:
                qext = float(parts[0])
            except Exception:
                pass
            if len(parts) >= 3:
                try:
                    qsca = float(parts[2])
                except Exception:
                    pass
        if "t matrix ext, sca" in ll:
            parts = line.replace(",", " ").split()
            nums = []
            for part in parts:
                try:
                    nums.append(float(part))
                except Exception:
                    pass
            if len(nums) >= 2:
                qext = nums[-2]
                qsca = nums[-1]
    return {"qext": qext, "qsca": qsca, "t_matrix_order": t_matrix_order}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mstm-bin", default="/Users/kaer/Downloads/MSTM-main/code/mstm")
    parser.add_argument("--run", action="store_true", help="Run MSTM after generating inputs")
    parser.add_argument("--n-values", default=",".join(str(n) for n in N_VALUES))
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    input_dir = base / "inputs"
    output_dir = base / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    results: dict[str, dict] = {}

    for n in n_values:
        positions_nm, _ = build_bcca(n_monomers=n, radius=R_MONOMER_NM, overlap=0.0, seed=SEED)
        positions_dimless = positions_nm * K_MEDIUM
        inp_path = input_dir / f"mstm_bcca_n{n}_w550_seed42.inp"
        out_rel = f"outputs/mstm_bcca_n{n}_w550_seed42.out"
        out_path = base / out_rel
        write_mstm_input(inp_path, out_rel, positions_dimless)

        row = {
            "n": n,
            "input": str(inp_path.relative_to(ROOT)),
            "output": str(out_path.relative_to(ROOT)),
            "state_yaml": STATE_PARITY.get(n),
            "mstm_runtime_s": None,
            "parsed": parse_mstm_output(out_path),
        }

        if args.run:
            if not os.path.exists(args.mstm_bin):
                raise FileNotFoundError(f"MSTM binary not found: {args.mstm_bin}")
            t0 = time.perf_counter()
            proc = subprocess.run(
                [args.mstm_bin, str(inp_path.relative_to(base))],
                cwd=str(base),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,
            )
            row["mstm_runtime_s"] = time.perf_counter() - t0
            row["returncode"] = proc.returncode
            row["stdout_tail"] = proc.stdout[-2000:]
            row["stderr_tail"] = proc.stderr[-2000:]
            if proc.returncode != 0:
                raise RuntimeError(
                    f"MSTM failed for N={n} with rc={proc.returncode}\n"
                    f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
                )
            row["parsed"] = parse_mstm_output(out_path)

        parsed_qext = row["parsed"].get("qext")
        state = STATE_PARITY.get(n)
        if parsed_qext is not None and state is not None:
            row["parsed_vs_state_diff_pct"] = 100.0 * (parsed_qext - state["mstm"]) / state["mstm"]
        results[str(n)] = row
        print(json.dumps(row, indent=2))

    manifest = {
        "purpose": "Frozen MSTM v4.0 validation artifacts for JCP manuscript evidence pack",
        "mstm_binary": args.mstm_bin,
        "run_mstm": args.run,
        "parameters": {
            "aggregate": "BCCA",
            "seed": SEED,
            "monomer_radius_nm": R_MONOMER_NM,
            "wavelength_nm": WAVELENGTH_NM,
            "k_medium": K_MEDIUM,
            "radius_dimless": R_DIMLESS,
            "refractive_index": {"real": M_COMPLEX.real, "imag": M_COMPLEX.imag},
            "solution_epsilon": "1.d-12",
            "translation_epsilon": "1.d-12",
            "random_orientation": True,
            "calculate_scattering_matrix": True,
        },
        "results": results,
    }
    (base / "mstm_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
