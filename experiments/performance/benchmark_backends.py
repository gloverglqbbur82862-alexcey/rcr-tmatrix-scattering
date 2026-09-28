#!/usr/bin/env python3
"""Unified Backend Benchmark Harness.

Sweeps N, p, and backends to measure initialization time, warmup time, average matvec time, peak memory usage, and relative error.
"""

import os
import sys
import time
import json
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solver.bicgstab_solver import MultisphereRCRSystem
from fmm.fmm_solver import FMMMultisphereSystem
from experiments.fmm_performance.benchmark import make_mie_coefficients, build_random_cluster

def get_peak_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        try:
            import resource
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == 'darwin':
                return maxrss / (1024.0 * 1024.0)
            else:
                return maxrss / 1024.0
        except Exception:
            return 0.0

def is_cuda_available():
    try:
        import cupy as cp
        return cp.is_available()
    except Exception:
        try:
            from numba import cuda
            return cuda.is_available()
        except Exception:
            return False

def run_bench():
    print("==================================================")
    print("Starting Unified Backend Benchmark Harness")
    print("==================================================")

    # Sweep settings
    N_list = [128, 256, 512, 1024]
    p_list = [3, 5]
    backends = [
        "direct_cpu_serial",
        "direct_cpu_parallel",
        "fmm_cpu_serial",
        "direct_gpu",
        "direct_gpu_dense",
        "fmm_gpu",
        "auto"
    ]
    runs = 5
    
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm
    cuda_avail = is_cuda_available()
    
    print(f"CUDA Available: {cuda_avail}")
    print(f"OS: {sys.platform}, Python: {sys.version.split()[0]}")
    print("Starting warmups...")
    
    # Warmup
    pos_warm = build_random_cluster(8, radius=r_monomer_nm, seed=42)
    pos_warm_dimless = pos_warm * k_medium
    nodr_warm = [3] * 8
    an_warm = [make_mie_coefficients(3)] * 8
    
    # Warm up direct serial
    sys_warm_ds = FMMMultisphereSystem(
        pos_warm_dimless, nodr_warm, an_warm, k_medium=1.0, 
        backend="direct_cpu_serial"
    )
    X_warm = np.random.default_rng(42).standard_normal(sys_warm_ds.N_total) + 1j * np.random.default_rng(42).standard_normal(sys_warm_ds.N_total)
    sys_warm_ds.apply_system_operator(X_warm)
    
    # Warm up direct parallel
    sys_warm_dp = FMMMultisphereSystem(
        pos_warm_dimless, nodr_warm, an_warm, k_medium=1.0, 
        backend="direct_cpu_parallel"
    )
    sys_warm_dp.apply_system_operator(X_warm)
    
    # Warm up FMM serial
    sys_warm_fs = FMMMultisphereSystem(
        pos_warm_dimless, nodr_warm, an_warm, k_medium=1.0, 
        backend="fmm_cpu_serial", fmm_mode="speed", fmm_tol=1e-6
    )
    sys_warm_fs.apply_system_operator(X_warm)
    print("Warmups complete.\n")

    results = {}

    for N in N_list:
        results[str(N)] = {}
        for p in p_list:
            print(f"--- Sweeping N={N}, p={p} ---")
            results[str(N)][str(p)] = {}
            
            # Setup cluster
            positions = build_random_cluster(N, radius=r_monomer_nm, seed=42)
            positions_dimless = positions * k_medium
            nodr_list = [p] * N
            an_list = [make_mie_coefficients(p)] * N
            
            nblk_half = p * (p + 2)
            N_total = 2 * nblk_half * N
            rng = np.random.default_rng(42)
            X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
            
            # Run direct_cpu_serial as baseline for rel_err & speedup
            baseline_time = None
            AX_baseline = None
            
            # Skip direct solver for large N if memory overflow is possible
            # We will run up to N=1024 for p=3, but restrict for p=5
            if N == 1024 and p > 3:
                print("  Skipping p > 3 for N=1024 direct solver to avoid excessive time/OOM.")
            
            for b in backends:
                print(f"  Backend: {b} ...")
                
                # Check GPU eligibility
                if b in ("direct_gpu", "fmm_gpu", "direct_gpu_dense") and not cuda_avail:
                    print("    -> Skipped (CUDA Unavailable)")
                    results[str(N)][str(p)][b] = {
                        "status": "Skipped (CUDA Unavailable)",
                        "init_time": None,
                        "warmup_time": None,
                        "matvec_time": None,
                        "peak_memory_mb": None,
                        "rel_error": None,
                        "speedup": None
                    }
                    continue
                
                # Skip if we restrict large direct solver configurations
                if b in ("direct_cpu_serial", "direct_cpu_parallel") and N == 1024 and p > 3:
                    print(f"    -> Skipped (Direct large size restriction)")
                    results[str(N)][str(p)][b] = {
                        "status": "Skipped (Direct large size restriction)",
                        "init_time": None,
                        "warmup_time": None,
                        "matvec_time": None,
                        "peak_memory_mb": None,
                        "rel_error": None,
                        "speedup": None
                    }
                    continue
                
                # Try running the backend
                try:
                    # Time initialization
                    t0 = time.perf_counter()
                    sys_obj = FMMMultisphereSystem(
                        positions_dimless, nodr_list, an_list, k_medium=1.0,
                        backend=b, fmm_mode="speed", fmm_tol=1e-6
                    )
                    init_time = time.perf_counter() - t0
                    
                    # Warmup
                    t0 = time.perf_counter()
                    AX = sys_obj.apply_system_operator(X)
                    warmup_time = time.perf_counter() - t0
                    
                    # Benchmark runs
                    mv_times = []
                    for _ in range(runs):
                        t0 = time.perf_counter()
                        sys_obj.apply_system_operator(X)
                        mv_times.append(time.perf_counter() - t0)
                    avg_mv = float(np.mean(mv_times))
                    
                    # Memory usage
                    peak_mem = get_peak_memory_mb()
                    
                    # Compare correctness
                    if b == "direct_cpu_serial":
                        baseline_time = avg_mv
                        AX_baseline = AX.copy()
                        rel_err = 0.0
                    else:
                        if AX_baseline is not None:
                            diff = np.linalg.norm(AX - AX_baseline)
                            rel_err = float(diff / np.linalg.norm(AX_baseline))
                        else:
                            rel_err = None
                    
                    # Calculate speedup
                    speedup = float(baseline_time / avg_mv) if (baseline_time is not None and avg_mv > 0) else None
                    
                    err_str = f"{rel_err:.2e}" if rel_err is not None else "N/A"
                    spd_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
                    print(f"    Success: init={init_time:.4f}s, warmup={warmup_time:.4f}s, matvec={avg_mv:.4f}s, mem={peak_mem:.1f}MB, error={err_str}, speedup={spd_str}")
                    
                    results[str(N)][str(p)][b] = {
                        "status": "Success",
                        "init_time": init_time,
                        "warmup_time": warmup_time,
                        "matvec_time": avg_mv,
                        "peak_memory_mb": peak_mem,
                        "rel_error": rel_err,
                        "speedup": speedup
                    }
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"    -> Failed: {str(e)}")
                    results[str(N)][str(p)][b] = {
                        "status": f"Failed: {str(e)}",
                        "init_time": None,
                        "warmup_time": None,
                        "matvec_time": None,
                        "peak_memory_mb": None,
                        "rel_error": None,
                        "speedup": None
                    }
                    
    # Write JSON results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "backend_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nJSON results saved to {json_path}")
    
    # Generate Markdown summary
    md_lines = []
    md_lines.append("# Unified Backend Benchmark Report")
    md_lines.append("")
    md_lines.append(f"- **CUDA Available**: {cuda_avail}")
    md_lines.append("")
    md_lines.append("| N | p | Backend | Status | Init Time (s) | Warmup (s) | Avg Matvec (s) | Memory (MB) | Rel Error | Speedup |")
    md_lines.append("|---|---|---------|--------|---------------|------------|----------------|-------------|-----------|---------|")
    
    for N in N_list:
        for p in p_list:
            for b in backends:
                res = results[str(N)][str(p)].get(b, {})
                status = res.get("status", "N/A")
                init_t = f"{res.get('init_time'):.4f}" if res.get('init_time') is not None else "N/A"
                warm_t = f"{res.get('warmup_time'):.4f}" if res.get('warmup_time') is not None else "N/A"
                mv_t = f"{res.get('matvec_time'):.4f}" if res.get('matvec_time') is not None else "N/A"
                mem = f"{res.get('peak_memory_mb'):.1f}" if res.get('peak_memory_mb') is not None else "N/A"
                err = f"{res.get('rel_error'):.2e}" if res.get('rel_error') is not None else "N/A"
                spd = f"{res.get('speedup'):.2f}x" if res.get('speedup') is not None else "N/A"
                
                md_lines.append(f"| {N} | {p} | `{b}` | {status} | {init_t} | {warm_t} | {mv_t} | {mem} | {err} | {spd} |")
                
    md_content = "\n".join(md_lines)
    md_path = os.path.join(out_dir, "backend_benchmark.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown table saved to {md_path}")

if __name__ == "__main__":
    run_bench()
