#!/usr/bin/env python3
"""Matched local benchmark for MSTM and the RCR solver.

This benchmark is intentionally separate from the validation artifacts.  It
uses the same BCCA geometry and optical parameters for each solver, records the
timing policy explicitly, and keeps MSTM as a reference implementation rather
than assuming speed superiority.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from geometry.geometry import build_bcca
from solver.bicgstab_solver import MultisphereRCRSystem


R_MONOMER_NM = 25.0
WAVELENGTH_NM = 550.0
K_MEDIUM = 2.0 * np.pi / WAVELENGTH_NM
R_DIMLESS = R_MONOMER_NM * K_MEDIUM
M_COMPLEX = 1.95 + 0.79j
SEED = 42
MONOMER_ORDER = 3


def shell_first_line(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    text = (proc.stdout or proc.stderr).strip().splitlines()
    return text[0] if text else f"returncode={proc.returncode}"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


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


def parse_mstm_output(out_path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "qext": None,
        "qabs": None,
        "qsca": None,
        "estimated_t_matrix_order": None,
        "t_matrix_order": None,
    }
    if not out_path.exists():
        return parsed
    lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        ll = line.lower().strip()
        if "estimated t matrix order" in ll and idx + 1 < len(lines):
            try:
                parsed["estimated_t_matrix_order"] = int(lines[idx + 1].strip().split()[0])
            except Exception:
                pass
        elif ll == "t matrix order:" and idx + 1 < len(lines):
            try:
                parsed["t_matrix_order"] = int(lines[idx + 1].strip().split()[0])
            except Exception:
                pass
        elif "total extinction" in ll and "efficiencies" in ll and idx + 1 < len(lines):
            parts = lines[idx + 1].strip().split()
            if len(parts) >= 3:
                try:
                    parsed["qext"] = float(parts[0])
                    parsed["qabs"] = float(parts[1])
                    parsed["qsca"] = float(parts[2])
                except ValueError:
                    pass
        if "t matrix ext, sca" in ll:
            values = []
            for token in line.replace(",", " ").split():
                try:
                    values.append(float(token))
                except ValueError:
                    pass
            if len(values) >= 2:
                parsed["qext"] = values[-2]
                parsed["qsca"] = values[-1]
    return parsed


def run_mstm_trial(
    *,
    base_dir: Path,
    solver_label: str,
    command_prefix: list[str],
    binary: Path,
    n: int,
    rep: int,
    positions_dimless: np.ndarray,
    timeout_s: int,
) -> dict[str, Any]:
    input_dir = base_dir / "inputs"
    output_dir = base_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{solver_label}_bcca_n{n}_w550_seed42_rep{rep}"
    inp_path = input_dir / f"{stem}.inp"
    out_rel = f"outputs/{stem}.out"
    out_path = base_dir / out_rel
    write_mstm_input(inp_path, out_rel, positions_dimless)

    cmd = [*command_prefix, str(binary), str(inp_path.relative_to(base_dir))]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    parsed = parse_mstm_output(out_path)
    return {
        "solver": solver_label,
        "n": n,
        "rep": rep,
        "command": cmd,
        "input": str(inp_path.relative_to(ROOT)),
        "output": str(out_path.relative_to(ROOT)),
        "wall_time_s": elapsed,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "parsed": parsed,
        "status": "success" if proc.returncode == 0 and parsed["qext"] is not None else "failed",
    }


def rcr_subprocess_code() -> str:
    return r"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import spherical_jn, spherical_yn

ROOT = os.environ["RCR_PROJECT_ROOT"]
sys.path.insert(0, ROOT)

from solver.bicgstab_solver import MultisphereRCRSystem


def mie_coefficients(m, x, p):
    def psi(nn, z):
        return z * spherical_jn(nn, z)

    def dpsi(nn, z):
        return spherical_jn(nn, z) + z * spherical_jn(nn, z, derivative=True)

    def xi(nn, z):
        return z * (spherical_jn(nn, z) + 1j * spherical_yn(nn, z))

    def dxi(nn, z):
        h = spherical_jn(nn, z) + 1j * spherical_yn(nn, z)
        dh = spherical_jn(nn, z, derivative=True) + 1j * spherical_yn(nn, z, derivative=True)
        return h + z * dh

    a_vals = []
    b_vals = []
    for n in range(1, p + 1):
        psi_n_x = psi(n, x)
        dpsi_n_x = dpsi(n, x)
        psi_n_mx = psi(n, m * x)
        dpsi_n_mx = dpsi(n, m * x)
        xi_n_x = xi(n, x)
        dxi_n_x = dxi(n, x)
        a_n = (m * psi_n_mx * dpsi_n_x - psi_n_x * dpsi_n_mx) / (
            m * psi_n_mx * dxi_n_x - xi_n_x * dpsi_n_mx
        )
        b_n = (psi_n_mx * dpsi_n_x - m * psi_n_x * dpsi_n_mx) / (
            psi_n_mx * dxi_n_x - m * xi_n_x * dpsi_n_mx
        )
        a_vals.append(complex(a_n))
        b_vals.append(complex(b_n))
    return np.array(a_vals), np.array(b_vals)


def mie_array(m, x, p):
    a, b = mie_coefficients(m, x, p)
    an = np.zeros((2, 2, p), dtype=np.complex128)
    diag = -0.5 * (a + b)
    off = -0.5 * (a - b)
    an[0, 0, :] = diag
    an[1, 1, :] = diag
    an[0, 1, :] = off
    an[1, 0, :] = off
    return an


positions = np.load(os.environ["RCR_POSITIONS_NPY"])
backend = os.environ["RCR_BACKEND"]
p = int(os.environ["RCR_MONOMER_ORDER"])
aggregate_order = int(os.environ["RCR_AGGREGATE_ORDER"])
r_dimless = float(os.environ["RCR_RADIUS_DIMLESS"])
m_complex = complex(float(os.environ["RCR_M_REAL"]), float(os.environ["RCR_M_IMAG"]))
tol = float(os.environ["RCR_TOL"])

an_single = mie_array(m_complex, r_dimless, p)
an_list = [an_single] * len(positions)
nodr_list = [p] * len(positions)

# Warm JIT and library initialization outside the timed window.
dummy_pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
dummy = MultisphereRCRSystem(
    dummy_pos,
    [p, p],
    [an_single, an_single],
    k_medium=1.0,
    backend=backend,
)
dummy_b = np.zeros(2 * p * (p + 2) * 2, dtype=np.complex128)
dummy_b[0] = 1.0
try:
    _ = dummy.solve(dummy_b, tol=1e-8, max_iter=2)
except Exception:
    pass
dummy_T = np.zeros((2 * p * (p + 2), 2 * p * (p + 2)), dtype=np.complex128)
try:
    _ = dummy.calculate_orientation_averages(dummy_T, [r_dimless, r_dimless])
except Exception:
    pass

t0 = time.perf_counter()
system = MultisphereRCRSystem(
    positions,
    nodr_list,
    an_list,
    k_medium=1.0,
    backend=backend,
)
T_agg = system.calculate_aggregate_tmatrix(aggregate_order=aggregate_order, tol=tol)
averages = system.calculate_orientation_averages(T_agg, [r_dimless] * len(positions))
elapsed = time.perf_counter() - t0

payload = {
    "wall_time_s": elapsed,
    "qext": float(averages["Qext"]),
    "qsca": float(averages["Qsca"]),
    "qabs": float(averages["Qabs"]),
    "g": float(averages["g"]),
    "profile_data": getattr(system, "profile_data", {}),
}
print("RCR_RESULT_JSON:" + json.dumps(payload, sort_keys=True))
"""


def run_rcr_trial(
    *,
    base_dir: Path,
    backend: str,
    threads: int,
    n: int,
    rep: int,
    positions_dimless: np.ndarray,
    aggregate_order: int,
    timeout_s: int,
) -> dict[str, Any]:
    input_dir = base_dir / "rcr_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    positions_path = input_dir / f"rcr_bcca_n{n}_seed42_rep{rep}.npy"
    np.save(positions_path, positions_dimless)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(rcr_subprocess_code())
        script_path = Path(handle.name)

    env = os.environ.copy()
    env.update(
        {
            "RCR_PROJECT_ROOT": str(ROOT),
            "RCR_POSITIONS_NPY": str(positions_path),
            "RCR_BACKEND": backend,
            "RCR_MONOMER_ORDER": str(MONOMER_ORDER),
            "RCR_AGGREGATE_ORDER": str(aggregate_order),
            "RCR_RADIUS_DIMLESS": str(R_DIMLESS),
            "RCR_M_REAL": str(M_COMPLEX.real),
            "RCR_M_IMAG": str(M_COMPLEX.imag),
            "RCR_TOL": "1e-12",
            "NUMBA_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        }
    )

    try:
        t0_process = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        process_wall = time.perf_counter() - t0_process
    finally:
        script_path.unlink(missing_ok=True)

    parsed = None
    for line in proc.stdout.splitlines():
        if line.startswith("RCR_RESULT_JSON:"):
            parsed = json.loads(line.split(":", 1)[1])
    return {
        "solver": f"rcr_{backend}_t{threads}",
        "n": n,
        "rep": rep,
        "backend": backend,
        "threads": threads,
        "aggregate_order": aggregate_order,
        "process_wall_time_s": process_wall,
        "timed_window_s": None if parsed is None else parsed["wall_time_s"],
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "parsed": parsed,
        "status": "success" if proc.returncode == 0 and parsed is not None else "failed",
    }


def summarize_trials(trials: list[dict[str, Any]], time_key: str) -> dict[str, Any]:
    good = [t for t in trials if t.get("status") == "success" and t.get(time_key) is not None]
    times = [float(t[time_key]) for t in good]
    qexts = []
    for trial in good:
        parsed = trial.get("parsed")
        if isinstance(parsed, dict):
            if "qext" in parsed and parsed["qext"] is not None:
                qexts.append(float(parsed["qext"]))
            elif "Qext" in parsed and parsed["Qext"] is not None:
                qexts.append(float(parsed["Qext"]))
    return {
        "count": len(good),
        "median_time_s": statistics.median(times) if times else None,
        "min_time_s": min(times) if times else None,
        "mean_time_s": statistics.mean(times) if times else None,
        "stdev_time_s": statistics.stdev(times) if len(times) > 1 else None,
        "median_qext": statistics.median(qexts) if qexts else None,
        "min_qext": min(qexts) if qexts else None,
        "max_qext": max(qexts) if qexts else None,
        "failures": len(trials) - len(good),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Matched MSTM/RCR Timing Benchmark",
        "",
        "This artifact records a same-geometry local timing comparison. It is not a",
        "claim that the present RCR implementation is generally faster than MSTM.",
        "",
        "## Timing Policy",
        "",
    ]
    for item in payload["timing_policy"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Environment", ""])
    env = payload["environment"]
    for key in [
        "timestamp_utc",
        "platform",
        "python",
        "git_commit",
        "gfortran",
        "mpifort",
        "mpirun",
    ]:
        lines.append(f"- `{key}`: {env.get(key)}")
    lines.append(
        "- `mstm_mpi_build`: run `experiments/mstm_matched_benchmark/"
        "build_mstm_parallel.sh` before reproducing MPI timings."
    )
    lines.extend(["", "## Summary", ""])
    lines.append(
        "| N | Solver | Repeats | Median time (s) | Min time (s) | Median Qext | Time / MSTM serial |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for n_str in sorted(payload["summary"], key=lambda x: int(x)):
        rows = payload["summary"][n_str]
        serial_time = rows.get("mstm_serial", {}).get("median_time_s")
        for solver_name, stats in rows.items():
            med = stats.get("median_time_s")
            rel = None
            if serial_time and med:
                rel = med / serial_time
            lines.append(
                "| {n} | `{solver}` | {count} | {med} | {min_t} | {qext} | {rel} |".format(
                    n=n_str,
                    solver=solver_name,
                    count=stats.get("count", 0),
                    med="N/A" if med is None else f"{med:.6g}",
                    min_t="N/A" if stats.get("min_time_s") is None else f"{stats['min_time_s']:.6g}",
                    qext="N/A" if stats.get("median_qext") is None else f"{stats['median_qext']:.10g}",
                    rel="N/A" if rel is None else f"{rel:.3f}x",
                )
            )
    lines.extend(
        [
            "",
            "## Manuscript Use",
            "",
            "Allowed: cite this as a local matched timing artifact with the timing",
            "policy above, and use it to explain that MSTM has serial and MPI",
            "realizations while the current RCR advantage is structural and",
            "accelerator-oriented.",
            "",
            "Not allowed: claim broad speed superiority over MSTM from this artifact",
            "alone, or mix these local CPU/MPI timings with the separate remote GPU",
            "benchmark as if they were collected on the same hardware.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="4,8,16")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mstm-bin", help="Path to a locally built serial MSTM executable.")
    parser.add_argument(
        "--mstm-mpi-bin",
        default=str(ROOT / "experiments/mstm_matched_benchmark/build_mstm_parallel/mstm_parallel"),
    )
    parser.add_argument("--mpi-procs", type=int, default=4)
    parser.add_argument("--skip-mstm-serial", action="store_true")
    parser.add_argument("--skip-mstm-mpi", action="store_true")
    parser.add_argument("--rcr-backends", default="direct_cpu_parallel")
    parser.add_argument("--rcr-threads", default="8")
    parser.add_argument(
        "--aggregate-order-policy",
        choices=["mstm_estimated", "rcr_auto"],
        default="mstm_estimated",
        help="Use the MSTM estimated random-orientation T-matrix order for RCR, or the RCR auto rule.",
    )
    parser.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args()

    if not args.skip_mstm_serial and not args.mstm_bin:
        parser.error("--mstm-bin is required unless --skip-mstm-serial is specified.")

    base_dir = Path(__file__).resolve().parent
    n_values = [int(item) for item in args.n_values.split(",") if item.strip()]
    rcr_backends = [item.strip() for item in args.rcr_backends.split(",") if item.strip()]
    rcr_threads = [int(item) for item in args.rcr_threads.split(",") if item.strip()]

    all_trials: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    aggregate_orders_by_n: dict[str, dict[str, Any]] = {}

    for n in n_values:
        positions_nm, _ = build_bcca(n_monomers=n, radius=R_MONOMER_NM, overlap=0.0, seed=SEED)
        positions_dimless = positions_nm * K_MEDIUM
        aggregate_order_auto = MultisphereRCRSystem.auto_aggregate_order(
            positions_dimless,
            R_DIMLESS,
            k_medium=1.0,
        )
        aggregate_order = aggregate_order_auto
        n_summary: dict[str, Any] = {}
        mstm_estimated_orders: list[int] = []

        if not args.skip_mstm_serial:
            trials = []
            for rep in range(1, args.repeats + 1):
                trial = run_mstm_trial(
                    base_dir=base_dir,
                    solver_label="mstm_serial",
                    command_prefix=[],
                    binary=Path(args.mstm_bin),
                    n=n,
                    rep=rep,
                    positions_dimless=positions_dimless,
                    timeout_s=args.timeout_s,
                )
                trials.append(trial)
                all_trials.append(trial)
                parsed_order = trial.get("parsed", {}).get("estimated_t_matrix_order")
                if isinstance(parsed_order, int):
                    mstm_estimated_orders.append(parsed_order)
                print(
                    f"N={n} mstm_serial rep={rep}: "
                    f"{trial['wall_time_s']:.3f}s status={trial['status']}",
                    flush=True,
                )
            n_summary["mstm_serial"] = summarize_trials(trials, "wall_time_s")

        mpi_bin = Path(args.mstm_mpi_bin)
        if not args.skip_mstm_mpi and mpi_bin.exists() and shutil.which("mpirun"):
            trials = []
            for rep in range(1, args.repeats + 1):
                trial = run_mstm_trial(
                    base_dir=base_dir,
                    solver_label=f"mstm_mpi_np{args.mpi_procs}",
                    command_prefix=["mpirun", "-np", str(args.mpi_procs)],
                    binary=mpi_bin,
                    n=n,
                    rep=rep,
                    positions_dimless=positions_dimless,
                    timeout_s=args.timeout_s,
                )
                trials.append(trial)
                all_trials.append(trial)
                parsed_order = trial.get("parsed", {}).get("estimated_t_matrix_order")
                if isinstance(parsed_order, int):
                    mstm_estimated_orders.append(parsed_order)
                print(
                    f"N={n} mstm_mpi_np{args.mpi_procs} rep={rep}: "
                    f"{trial['wall_time_s']:.3f}s status={trial['status']}",
                    flush=True,
                )
            n_summary[f"mstm_mpi_np{args.mpi_procs}"] = summarize_trials(trials, "wall_time_s")

        if args.aggregate_order_policy == "mstm_estimated" and mstm_estimated_orders:
            aggregate_order = max(mstm_estimated_orders)
        aggregate_orders_by_n[str(n)] = {
            "rcr_auto": aggregate_order_auto,
            "mstm_estimated_values": sorted(set(mstm_estimated_orders)),
            "used_for_rcr": aggregate_order,
            "policy": args.aggregate_order_policy,
        }

        for backend in rcr_backends:
            for threads in rcr_threads:
                trials = []
                for rep in range(1, args.repeats + 1):
                    trial = run_rcr_trial(
                        base_dir=base_dir,
                        backend=backend,
                        threads=threads,
                        n=n,
                        rep=rep,
                        positions_dimless=positions_dimless,
                        aggregate_order=aggregate_order,
                        timeout_s=args.timeout_s,
                    )
                    trials.append(trial)
                    all_trials.append(trial)
                    timed = trial.get("timed_window_s")
                    print(
                        f"N={n} rcr_{backend}_t{threads} rep={rep}: "
                        f"{'N/A' if timed is None else f'{timed:.3f}s'} "
                        f"status={trial['status']}",
                        flush=True,
                    )
                n_summary[f"rcr_{backend}_t{threads}"] = summarize_trials(
                    trials,
                    "timed_window_s",
                )

        summary[str(n)] = n_summary

    payload = {
        "purpose": "same-geometry local matched timing comparison for MSTM and RCR",
        "parameters": {
            "aggregate": "BCCA",
            "seed": SEED,
            "monomer_radius_nm": R_MONOMER_NM,
            "wavelength_nm": WAVELENGTH_NM,
            "radius_dimless": R_DIMLESS,
            "refractive_index": {"real": M_COMPLEX.real, "imag": M_COMPLEX.imag},
            "monomer_order": MONOMER_ORDER,
            "solution_epsilon": "1.d-12",
            "translation_epsilon": "1.d-12",
            "random_orientation": True,
            "calculate_scattering_matrix": True,
            "aggregate_order_policy": args.aggregate_order_policy,
        },
        "timing_policy": [
            "Input-deck generation and aggregate-coordinate generation are excluded from timed values.",
            "MSTM wall time includes executable startup, input parsing, solve, and output-file writing.",
            "RCR timed_window_s is measured inside a Python subprocess after a small JIT/library warm-up; it includes solver construction, aggregate T-matrix construction, and orientation averaging.",
            "RCR process_wall_time_s is retained in raw trials but is not used for the summary table.",
            "GPU RCR backends are only included when the local Python environment provides CuPy/CUDA.",
            "Timings are local-machine measurements and must not be mixed with the separate remote GPU benchmark as a same-hardware comparison.",
        ],
        "environment": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "git_commit": git_hash(),
            "gfortran": shell_first_line(["gfortran", "--version"]),
            "mpifort": shell_first_line(["mpifort", "--version"]),
            "mpirun": shell_first_line(["mpirun", "--version"]),
            "mstm_serial_binary": str(Path(args.mstm_bin)),
            "mstm_mpi_binary": str(Path(args.mstm_mpi_bin)),
        },
        "summary": summary,
        "aggregate_orders_by_n": aggregate_orders_by_n,
        "trials": all_trials,
    }

    json_path = base_dir / "matched_benchmark_results.json"
    md_path = base_dir / "matched_benchmark_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
