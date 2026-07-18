#!/usr/bin/env python3
"""Same-workstation GPU-RCR versus MSTM benchmark.

This runner is intended for a reference workstation where the RCR GPU backend is
available from native Windows Python and MSTM v4.0 is available inside WSL2.
It records the comparison explicitly as a same-physical-machine benchmark, not
as a same-runtime benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
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


def shell_first_line(cmd: list[str], timeout_s: int = 20) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    text = (proc.stdout or proc.stderr).strip().splitlines()
    return text[0] if text else f"returncode={proc.returncode}"


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    rest = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def write_mstm_input(inp_path: Path, out_rel: str, positions_dimless: np.ndarray) -> None:
    lines = [
        "output_file",
        out_rel.replace("\\", "/"),
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


def read_positions_from_mstm_input(inp_path: Path) -> np.ndarray:
    """Read dimensionless sphere centers from a frozen MSTM input deck."""
    lines = inp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    positions: list[list[float]] = []
    in_sphere_data = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower == "sphere_data":
            in_sphere_data = True
            continue
        if lower == "end_of_sphere_data":
            break
        if not in_sphere_data:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not positions:
        raise ValueError(f"No sphere_data positions found in {inp_path}")
    return np.asarray(positions, dtype=np.float64)


def run_mstm_wsl_trial(
    *,
    base_dir: Path,
    wsl_distro: str,
    mstm_wsl_bin: str,
    solver_label: str,
    mpi_procs: int | None,
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

    base_wsl = wsl_path(base_dir)
    input_arg = shlex.quote(inp_path.relative_to(base_dir).as_posix())
    if mpi_procs is None:
        run_text = f"{shlex.quote(mstm_wsl_bin)} {input_arg}"
    else:
        run_text = (
            "OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 "
            f"mpirun --allow-run-as-root --oversubscribe -np {int(mpi_procs)} "
            f"{shlex.quote(mstm_wsl_bin)} {input_arg}"
        )
    cmd_text = f"cd {shlex.quote(base_wsl)} && {run_text}"
    cmd = ["wsl.exe", "-d", wsl_distro, "--", "bash", "-lc", cmd_text]
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
        "mpi_procs": mpi_procs,
        "command": cmd,
        "input": str(inp_path.relative_to(ROOT)),
        "output": str(out_path.relative_to(ROOT)),
        "wall_time_s": elapsed,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1600:],
        "stderr_tail": proc.stderr[-1600:],
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

# Warm JIT and GPU library initialization outside the timed compute window.
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
    finite_observables = False
    if parsed is not None:
        try:
            finite_observables = all(
                np.isfinite(float(parsed[key])) for key in ("qext", "qsca", "qabs")
            )
        except Exception:
            finite_observables = False
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
        "stdout_tail": proc.stdout[-1600:],
        "stderr_tail": proc.stderr[-1600:],
        "parsed": parsed,
        "status": (
            "success"
            if proc.returncode == 0 and parsed is not None and finite_observables
            else "failed"
        ),
    }


def summarize_trials(trials: list[dict[str, Any]], time_key: str) -> dict[str, Any]:
    good = [t for t in trials if t.get("status") == "success" and t.get(time_key) is not None]
    times = [float(t[time_key]) for t in good]
    qexts = []
    for trial in good:
        parsed = trial.get("parsed")
        if isinstance(parsed, dict):
            if "qext" in parsed and parsed["qext"] is not None:
                qext = float(parsed["qext"])
                if np.isfinite(qext):
                    qexts.append(qext)
            elif "Qext" in parsed and parsed["Qext"] is not None:
                qext = float(parsed["Qext"])
                if np.isfinite(qext):
                    qexts.append(qext)
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


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def parse_aggregate_order_overrides(text: str) -> dict[int, int]:
    overrides: dict[int, int] = {}
    if not text.strip():
        return overrides
    for item in text.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid aggregate-order override {item!r}; expected N:order")
        n_text, order_text = item.split(":", 1)
        overrides[int(n_text.strip())] = int(order_text.strip())
    return overrides


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Same-Workstation GPU-RCR versus MSTM Benchmark",
        "",
        "This artifact records a same-physical-workstation comparison. MSTM is run",
        "inside WSL2 as CPU serial and, when available, MPI-parallel executables.",
        "RCR GPU backends are run from native Windows Python/CuPy on the same reference workstation",
        "workstation. It is therefore a same-machine benchmark, not a same-runtime",
        "benchmark.",
        "",
        "## Timing Policy",
        "",
    ]
    for item in payload["timing_policy"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Environment", ""])
    for key, value in payload["environment"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Geometry Sources", ""])
    for n_str in sorted(payload.get("geometry_sources_by_n", {}), key=lambda x: int(x)):
        lines.append(f"- `N={n_str}`: `{payload['geometry_sources_by_n'][n_str]}`")
        metadata = payload.get("geometry_metadata_by_n", {}).get(n_str)
        if metadata:
            description = metadata.get("description")
            source_type = metadata.get("source_type")
            if source_type or description:
                lines.append(
                    "  - provenance: "
                    + "; ".join(str(item) for item in (source_type, description) if item)
                )

    lines.extend(["", "## Summary", ""])
    lines.append(
        "| N | Solver | Repeats | Median compute time (s) | Median process/wall time (s) | Median Qext | Speedup vs fastest MSTM |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for n_str in sorted(payload["summary"], key=lambda x: int(x)):
        rows = payload["summary"][n_str]
        mstm_times = [
            stats.get("median_time_s")
            for solver_name, stats in rows.items()
            if solver_name.startswith("mstm_")
            and not solver_name.endswith("_process")
            and stats.get("median_time_s") is not None
        ]
        mstm_time = min(mstm_times) if mstm_times else None
        for solver_name, stats in rows.items():
            if solver_name.endswith("_process"):
                continue
            med = stats.get("median_time_s")
            process_med = None
            if solver_name.startswith("rcr_"):
                process_med = rows.get(f"{solver_name}_process", {}).get("median_time_s")
            else:
                process_med = med
            speedup = None
            if mstm_time and med:
                speedup = mstm_time / med
            lines.append(
                "| {n} | `{solver}` | {count} | {med} | {process} | {qext} | {speedup} |".format(
                    n=n_str,
                    solver=solver_name,
                    count=stats.get("count", 0),
                    med=fmt(med),
                    process=fmt(process_med),
                    qext=fmt(stats.get("median_qext"), 10),
                    speedup="N/A" if speedup is None else f"{speedup:.2f}x",
                )
            )

    lines.extend(
        [
            "",
            "## Manuscript Use",
            "",
            "Allowed: use successful rows as a same-workstation GPU-RCR versus MSTM",
            "CPU/MPI context, with the WSL2/native-Windows distinction stated explicitly.",
            "Use the RCR process/wall time column when making conservative statements;",
            "use the RCR compute-window column only when also stating that JIT/library",
            "warm-up is excluded.",
            "",
            "Not allowed: claim general large-N superiority beyond the tested geometry,",
            "truncation, repeat count, and MPI rank count.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default="64,128,256")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    parser.add_argument("--mstm-wsl-bin", default="/root/mstm_build/mstm")
    parser.add_argument("--mstm-mpi-wsl-bin", default="/root/mstm_build_parallel/mstm_parallel")
    parser.add_argument("--mpi-procs", default="4,8")
    parser.add_argument(
        "--geometry-input-dir",
        default=str(ROOT / "experiments/mstm_validation/inputs"),
        help="Directory of frozen MSTM input decks used as geometry source when available.",
    )
    parser.add_argument("--skip-mstm", action="store_true")
    parser.add_argument("--skip-mstm-serial", action="store_true")
    parser.add_argument("--skip-mstm-mpi", action="store_true")
    parser.add_argument("--rcr-backends", default="direct_gpu,direct_gpu_dense")
    parser.add_argument("--rcr-threads", type=int, default=64)
    parser.add_argument(
        "--aggregate-order-policy",
        choices=["mstm_estimated", "rcr_auto"],
        default="mstm_estimated",
    )
    parser.add_argument(
        "--aggregate-order-overrides",
        default="",
        help="Optional comma-separated fixed aggregate orders, e.g. 256:21,512:26.",
    )
    parser.add_argument("--timeout-s", type=int, default=14400)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    n_values = [int(item) for item in args.n_values.split(",") if item.strip()]
    rcr_backends = [item.strip() for item in args.rcr_backends.split(",") if item.strip()]
    mpi_procs_values = [int(item) for item in args.mpi_procs.split(",") if item.strip()]
    aggregate_order_overrides = parse_aggregate_order_overrides(args.aggregate_order_overrides)

    base_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_mstm:
        subprocess.run(
            ["wsl.exe", "-d", args.wsl_distro, "--", "bash", "-lc", "true"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    all_trials: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    aggregate_orders_by_n: dict[str, dict[str, Any]] = {}
    geometry_sources_by_n: dict[str, str] = {}
    geometry_metadata_by_n: dict[str, Any] = {}
    geometry_input_dir = Path(args.geometry_input_dir)

    for n in n_values:
        frozen_input = geometry_input_dir / f"mstm_bcca_n{n}_w550_seed42.inp"
        if frozen_input.exists():
            positions_dimless = read_positions_from_mstm_input(frozen_input)
            geometry_sources_by_n[str(n)] = str(frozen_input)
            provenance_path = frozen_input.with_suffix(".provenance.json")
            if provenance_path.exists():
                try:
                    geometry_metadata_by_n[str(n)] = json.loads(
                        provenance_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as exc:
                    geometry_metadata_by_n[str(n)] = {
                        "source_type": "provenance_parse_failed",
                        "description": str(exc),
                    }
        else:
            positions_nm, _ = build_bcca(n_monomers=n, radius=R_MONOMER_NM, overlap=0.0, seed=SEED)
            positions_dimless = positions_nm * K_MEDIUM
            geometry_sources_by_n[str(n)] = "generated_by_geometry.build_bcca"
        aggregate_order_auto = MultisphereRCRSystem.auto_aggregate_order(
            positions_dimless,
            R_DIMLESS,
            k_medium=1.0,
        )
        aggregate_order = aggregate_order_auto
        n_summary: dict[str, Any] = {}
        mstm_estimated_orders: list[int] = []

        if not args.skip_mstm and not args.skip_mstm_serial:
            trials = []
            for rep in range(1, args.repeats + 1):
                trial = run_mstm_wsl_trial(
                    base_dir=base_dir,
                    wsl_distro=args.wsl_distro,
                    mstm_wsl_bin=args.mstm_wsl_bin,
                    solver_label="mstm_wsl_serial",
                    mpi_procs=None,
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
                    f"N={n} mstm_wsl_serial rep={rep}: "
                    f"{trial['wall_time_s']:.3f}s status={trial['status']}",
                    flush=True,
                )
            n_summary["mstm_wsl_serial"] = summarize_trials(trials, "wall_time_s")

        if not args.skip_mstm and not args.skip_mstm_mpi:
            for mpi_procs in mpi_procs_values:
                solver_label = f"mstm_wsl_mpi_np{mpi_procs}"
                trials = []
                for rep in range(1, args.repeats + 1):
                    trial = run_mstm_wsl_trial(
                        base_dir=base_dir,
                        wsl_distro=args.wsl_distro,
                        mstm_wsl_bin=args.mstm_mpi_wsl_bin,
                        solver_label=solver_label,
                        mpi_procs=mpi_procs,
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
                        f"N={n} {solver_label} rep={rep}: "
                        f"{trial['wall_time_s']:.3f}s status={trial['status']}",
                        flush=True,
                    )
                n_summary[solver_label] = summarize_trials(trials, "wall_time_s")

        if n in aggregate_order_overrides:
            aggregate_order = aggregate_order_overrides[n]
        elif args.aggregate_order_policy == "mstm_estimated" and mstm_estimated_orders:
            aggregate_order = max(mstm_estimated_orders)
        aggregate_orders_by_n[str(n)] = {
            "rcr_auto": aggregate_order_auto,
            "mstm_estimated_values": sorted(set(mstm_estimated_orders)),
            "used_for_rcr": aggregate_order,
            "policy": (
                "explicit_override"
                if n in aggregate_order_overrides
                else args.aggregate_order_policy
            ),
        }

        for backend in rcr_backends:
            trials = []
            for rep in range(1, args.repeats + 1):
                trial = run_rcr_trial(
                    base_dir=base_dir,
                    backend=backend,
                    threads=args.rcr_threads,
                    n=n,
                    rep=rep,
                    positions_dimless=positions_dimless,
                    aggregate_order=aggregate_order,
                    timeout_s=args.timeout_s,
                )
                trials.append(trial)
                all_trials.append(trial)
                timed = trial.get("timed_window_s")
                process_wall = trial.get("process_wall_time_s")
                print(
                    f"N={n} rcr_{backend}_t{args.rcr_threads} rep={rep}: "
                    f"compute={'N/A' if timed is None else f'{timed:.3f}s'} "
                    f"process={process_wall:.3f}s status={trial['status']}",
                    flush=True,
                )
            solver_key = f"rcr_{backend}_t{args.rcr_threads}"
            n_summary[solver_key] = summarize_trials(trials, "timed_window_s")
            n_summary[f"{solver_key}_process"] = summarize_trials(trials, "process_wall_time_s")

        summary[str(n)] = n_summary

    payload = {
        "purpose": "same-physical-workstation GPU RCR versus MSTM CPU/MPI benchmark",
        "parameters": {
            "aggregate": "frozen sphere-aggregate geometries; see geometry_sources_by_n and geometry_metadata_by_n",
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
            "aggregate_order_overrides": aggregate_order_overrides,
        },
        "timing_policy": [
            "Aggregate-coordinate generation and input-deck generation are excluded from timed values.",
            "MSTM wall time is measured around the WSL2 command invocation after a one-command WSL warm-up; it includes executable startup, input parsing, solve, and output-file writing.",
            "RCR process/wall time includes native Python subprocess startup plus JIT/GPU warm-up and the physical computation.",
            "RCR compute-window time excludes the small dummy JIT/GPU warm-up and includes solver construction, aggregate T-matrix construction, and orientation averaging.",
            "For each N, all solvers use the same frozen sphere-aggregate geometry, monomer radius, wavelength, refractive index, random-orientation output, and tolerance policy recorded here.",
            "MSTM MPI timings use the OpenMPI rank counts recorded in the environment metadata.",
        ],
        "environment": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "git_commit": git_hash(),
            "wsl_distro": args.wsl_distro,
            "mstm_wsl_binary": args.mstm_wsl_bin,
            "mstm_mpi_wsl_binary": args.mstm_mpi_wsl_bin,
            "requested_mpi_procs": mpi_procs_values,
            "wsl_gfortran": shell_first_line(
                ["wsl.exe", "-d", args.wsl_distro, "--", "bash", "-lc", "gfortran --version"]
            ),
            "wsl_mpifort": shell_first_line(
                ["wsl.exe", "-d", args.wsl_distro, "--", "bash", "-lc", "command -v mpifort || echo not-found"]
            ),
            "wsl_mpirun": shell_first_line(
                ["wsl.exe", "-d", args.wsl_distro, "--", "bash", "-lc", "command -v mpirun || echo not-found"]
            ),
            "nvidia_smi": shell_first_line(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
        "summary": summary,
        "aggregate_orders_by_n": aggregate_orders_by_n,
        "geometry_sources_by_n": geometry_sources_by_n,
        "geometry_metadata_by_n": geometry_metadata_by_n,
        "trials": all_trials,
    }

    json_path = base_dir / "gpu_mstm_matched_results.json"
    md_path = base_dir / "gpu_mstm_matched_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
