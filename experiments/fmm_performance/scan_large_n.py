#!/usr/bin/env python3
"""Large-N memory-time scalability sweep script.

Runs Direct JIT, FMM Speed (tol=1e-6), and FMM Reference (tol=1e-12)
solvers at N = [128, 256, 512, 1024] and p = 3, using separate subprocesses
to prevent memory pollution and measure accurate process-level peak RSS.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import resource
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solver.bicgstab_solver import MultisphereRCRSystem
from fmm.fmm_solver import FMMMultisphereSystem
from experiments.fmm_performance.benchmark import make_mie_coefficients, build_random_cluster


def run_single_config(N: int, p: int, solver: str):
    """Executes a single solver configuration and prints the result as JSON."""
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm

    # Build geometry
    positions = build_random_cluster(N, radius=r_monomer_nm, seed=42)
    positions_dimless = positions * k_medium
    nodr_list = [p] * N
    an_list = [make_mie_coefficients(p)] * N

    # Setup random input vector
    nblk_half = p * (p + 2)
    N_total = 2 * nblk_half * N
    rng = np.random.default_rng(42)
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)

    # 1. Initialize
    t0 = time.perf_counter()
    if solver == "direct":
        sys_obj = MultisphereRCRSystem(positions_dimless, nodr_list, an_list, k_medium=1.0)
    elif solver == "fmm_speed":
        sys_obj = FMMMultisphereSystem(
            positions_dimless, nodr_list, an_list, k_medium=1.0,
            p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
            fmm_mode="speed", fmm_tol=1e-6, theta=0.55,
            use_precomputed_passes=True, d_cache_budget_mb=500.0,
            verbose=False
        )
    elif solver == "fmm_ref":
        sys_obj = FMMMultisphereSystem(
            positions_dimless, nodr_list, an_list, k_medium=1.0,
            p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
            fmm_mode="reference", fmm_tol=1e-12, theta=0.55,
            use_precomputed_passes=True, d_cache_budget_mb=500.0,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")
    init_time = time.perf_counter() - t0

    # 2. Warmup (includes compilation)
    t0 = time.perf_counter()
    sys_obj.apply_system_operator(X)
    warmup_time = time.perf_counter() - t0

    # 3. Iterations
    iter_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sys_obj.apply_system_operator(X)
        iter_times.append(time.perf_counter() - t0)
    avg_matvec = float(np.mean(iter_times))

    # Measure memory
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_rss_mb = max_rss / (1024.0 * 1024.0)
    else:
        peak_rss_mb = max_rss / 1024.0

    res = {
        "init_time": init_time,
        "warmup_time": warmup_time,
        "matvec_time": avg_matvec,
        "peak_rss_mb": peak_rss_mb
    }
    print(json.dumps(res))


def run_orchestrator(p: int):
    """Orchestrates the subprocess runs and prints a summary table."""
    N_list = [128, 256, 512, 1024]
    solvers = ["direct", "fmm_speed", "fmm_ref"]
    results = {s: {} for s in solvers}

    print("==================================================================")
    # Print header info
    import platform
    print(f"Starting FMM vs Direct Large-N Scalability Sweep (p={p})")
    print(f"OS: {sys.platform} | Python: {platform.python_version()}")
    print("==================================================================")

    for N in N_list:
        for solver in solvers:
            print(f"Running N={N}, solver={solver} in subprocess...", flush=True)
            cmd = [sys.executable, __file__, "--sub", "--N", str(N), "--p", str(p), "--solver", solver]
            
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                res = json.loads(proc.stdout.strip())
                results[solver][N] = res
                print(f"  Success: init={res['init_time']:.3f}s, warmup={res['warmup_time']:.3f}s, matvec={res['matvec_time']:.3f}s, peak_rss={res['peak_rss_mb']:.1f} MB")
            except Exception as e:
                print(f"  Failed solver={solver}, N={N}: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"  Stderr: {e.stderr}")
                results[solver][N] = {
                    "init_time": float("nan"),
                    "warmup_time": float("nan"),
                    "matvec_time": float("nan"),
                    "peak_rss_mb": float("nan"),
                    "error": str(e)
                }

    # Generate Markdown table
    markdown_lines = []
    markdown_lines.append("# FMM vs Direct JIT Performance Sweep")
    markdown_lines.append(f"Truncation order: $p = {p}$")
    markdown_lines.append("")
    markdown_lines.append("| N | Solver | Init Time (s) | Warmup Time (s) | Avg Matvec (s) | Speedup vs Direct | Peak RSS (MB) |")
    markdown_lines.append("|---|--------|---------------|-----------------|----------------|-------------------|---------------|")

    for N in N_list:
        d_res = results["direct"].get(N, {})
        fs_res = results["fmm_speed"].get(N, {})
        fr_res = results["fmm_ref"].get(N, {})
        
        d_matvec = d_res.get("matvec_time", float("nan"))
        fs_matvec = fs_res.get("matvec_time", float("nan"))
        fr_matvec = fr_res.get("matvec_time", float("nan"))

        def fmt_val(v, fmt=".3f"):
            return f"{v:{fmt}}" if not np.isnan(v) else "Error/Skipped"

        def fmt_mem(v):
            return f"{v:.1f}" if not np.isnan(v) else "Error/Skipped"

        # Direct
        markdown_lines.append(
            f"| {N} | Direct JIT | {fmt_val(d_res.get('init_time', float('nan')))} | {fmt_val(d_res.get('warmup_time', float('nan')))} | {fmt_val(d_matvec)} | - | {fmt_mem(d_res.get('peak_rss_mb', float('nan')))} |"
        )
        
        # FMM Speed
        if not np.isnan(d_matvec) and not np.isnan(fs_matvec) and fs_matvec > 0:
            speedup_s = f"{d_matvec / fs_matvec:.2f}x"
        else:
            speedup_s = "N/A"
        markdown_lines.append(
            f"| {N} | FMM Speed (1e-6) | {fmt_val(fs_res.get('init_time', float('nan')))} | {fmt_val(fs_res.get('warmup_time', float('nan')))} | {fmt_val(fs_matvec)} | {speedup_s} | {fmt_mem(fs_res.get('peak_rss_mb', float('nan')))} |"
        )

        # FMM Reference
        if not np.isnan(d_matvec) and not np.isnan(fr_matvec) and fr_matvec > 0:
            speedup_r = f"{d_matvec / fr_matvec:.2f}x"
        else:
            speedup_r = "N/A"
        markdown_lines.append(
            f"| {N} | FMM Reference (1e-12) | {fmt_val(fr_res.get('init_time', float('nan')))} | {fmt_val(fr_res.get('warmup_time', float('nan')))} | {fmt_val(fr_matvec)} | {speedup_r} | {fmt_mem(fr_res.get('peak_rss_mb', float('nan')))} |"
        )

    summary_md = "\n".join(markdown_lines)
    print("\n" + summary_md + "\n")

    # Save to file
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_large_n_results.md")
    with open(out_path, "w") as f:
        f.write(summary_md)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", action="store_true", help="Run in subprocess mode for a single configuration")
    parser.add_argument("--N", type=int, default=128, help="Number of monomers")
    parser.add_argument("--p", type=int, default=3, help="Truncation order")
    parser.add_argument("--solver", type=str, choices=["direct", "fmm_speed", "fmm_ref"], default="fmm_speed", help="Solver type")
    args = parser.parse_args()

    if args.sub:
        run_single_config(args.N, args.p, args.solver)
    else:
        run_orchestrator(args.p)
