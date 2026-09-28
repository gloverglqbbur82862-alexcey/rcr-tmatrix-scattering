#!/usr/bin/env python3
"""Benchmark Full Physical Solve: CPU vs. GPU.

Measures the total wall time for system creation, aggregate T-matrix computation (solving all columns),
and orientation averaging for N = [16, 32, 64, 128, 256].
"""

import os
import sys
import time
import json
import subprocess
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geometry.geometry import build_bcca

def run_subprocess_solve(N: int, backend: str, thread_count: int) -> dict:
    """Execute the full solve in a separate Python process to isolate JIT compiler and threads."""
    import tempfile
    
    # Generate BCCA soot cluster coordinates deterministically
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm
    r_dimless = r_monomer_nm * k_medium
    m_complex = 1.95 + 0.79j
    p = 3
    
    # Try multiple seeds if the BCCA build fails (e.g. on different platforms/numpy versions)
    positions = None
    for s in range(42, 142):
        try:
            positions, _ = build_bcca(n_monomers=N, radius=r_monomer_nm, overlap=0.0, seed=s)
            break
        except RuntimeError:
            continue
    if positions is None:
        raise RuntimeError(f"Could not build BCCA cluster for N={N} with seeds in range 42 to 142")
    positions_dimless = positions * k_medium
    
    # Use standard Wiscombe-like formula for aggregate order
    # (matches mstm estimation logic)
    cm = np.mean(positions_dimless, axis=0)
    r_circ = float(np.max(np.linalg.norm(positions_dimless - cm, axis=1))) + r_dimless
    agg_order = int(np.ceil(r_circ + 4.05 * r_circ**(1/3) + 2))
    agg_order = max(agg_order, 3)
    
    # Save arrays to temp files
    fd_pos, path_pos = tempfile.mkstemp(suffix=".npy")
    np.save(path_pos, positions_dimless)
    os.close(fd_pos)
    
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\', '/')
    path_pos_esc = path_pos.replace('\\', '/')
    
    py_code = f"""
import sys
sys.path.insert(0, '{proj_root}')
import numpy as np
import time
from solver.bicgstab_solver import MultisphereRCRSystem

pos = np.load("{path_pos_esc}")
nsph = len(pos)
nodr_list = [{p}] * nsph

# Compute monomer Mie coefficients
from scipy.special import spherical_jn, spherical_yn
def mie_coefficients(m, x, p):
    def psi(nn, z): return z * spherical_jn(nn, z)
    def dpsi(nn, z): return spherical_jn(nn, z) + z * spherical_jn(nn, z, derivative=True)
    def xi(nn, z): return z * (spherical_jn(nn, z) + 1j * spherical_yn(nn, z))
    def dxi(nn, z):
        h = spherical_jn(nn, z) + 1j * spherical_yn(nn, z)
        dh = spherical_jn(nn, z, derivative=True) + 1j * spherical_yn(nn, z, derivative=True)
        return h + z * dh
    an_list = []
    bn_list = []
    for n in range(1, p + 1):
        psi_n_x = psi(n, x)
        dpsi_n_x = dpsi(n, x)
        psi_n_mx = psi(n, m * x)
        dpsi_n_mx = dpsi(n, m * x)
        xi_n_x = xi(n, x)
        dxi_n_x = dxi(n, x)
        a_n = (m * psi_n_mx * dpsi_n_x - psi_n_x * dpsi_n_mx) / (m * psi_n_mx * dxi_n_x - xi_n_x * dpsi_n_mx)
        b_n = (psi_n_mx * dpsi_n_x - m * psi_n_x * dpsi_n_mx) / (psi_n_mx * dxi_n_x - m * xi_n_x * dpsi_n_mx)
        an_list.append(complex(a_n))
        bn_list.append(complex(b_n))
    
    # shape (2, 2, p)
    an_arr = np.zeros((2, 2, p), dtype=np.complex128)
    an_np = np.array(an_list)
    bn_np = np.array(bn_list)
    diag = -0.5 * (an_np + bn_np)
    off = -0.5 * (an_np - bn_np)
    an_arr[0, 0, :] = diag
    an_arr[1, 1, :] = diag
    an_arr[0, 1, :] = off
    an_arr[1, 0, :] = off
    return an_arr

an_single = mie_coefficients({m_complex}, {r_dimless}, {p})
an_list = [an_single] * nsph

# Warm up JIT compiler on a small dummy case
dummy_pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
dummy_system = MultisphereRCRSystem(dummy_pos, [{p}, {p}], [an_single, an_single], k_medium=1.0, backend="{backend}")
dummy_b = np.zeros(2 * {p} * ({p} + 2) * 2, dtype=np.complex128)
dummy_b[0] = 1.0
_ = dummy_system.solve(dummy_b, tol=1e-8, max_iter=2)

# Warm up JIT compiler for orientation averaging as well to separate JIT compile time from physical runtime
dummy_T = np.zeros((2 * {p} * ({p} + 2), 2 * {p} * ({p} + 2)), dtype=np.complex128)
_ = dummy_system.calculate_orientation_averages(dummy_T, [{r_dimless}, {r_dimless}])


# Start measuring full solve
t0 = time.perf_counter()
system = MultisphereRCRSystem(pos, nodr_list, an_list, k_medium=1.0, backend="{backend}")
T_agg = system.calculate_aggregate_tmatrix(aggregate_order={agg_order}, tol=1e-8)
radii_dimless = [{r_dimless}] * nsph
averages = system.calculate_orientation_averages(T_agg, radii_dimless)
t_elap = time.perf_counter() - t0

p_data = getattr(system, 'profile_data', {{}})
out_parts = [
    f"time={{t_elap:.6f}}",
    f"qext={{averages['Qext']:.8f}}",
    f"assembly_time={{p_data.get('assembly_time', 0.0):.6f}}",
    f"h2d_time={{p_data.get('h2d_time', 0.0):.6f}}",
    f"rhs_time={{p_data.get('rhs_time', 0.0):.6f}}",
    f"solve_time={{p_data.get('solve_time', 0.0):.6f}}",
    f"merge_time={{p_data.get('merge_time', 0.0):.6f}}",
    f"precompute_merge_time={{p_data.get('precompute_merge_time', 0.0):.6f}}",
    f"merge_gemm_time={{p_data.get('merge_gemm_time', 0.0):.6f}}",
    f"orientation_time={{p_data.get('orientation_time', 0.0):.6f}}",
    f"krylov_iters={{p_data.get('krylov_iters', 0):d}}",
    f"vram_mb={{p_data.get('vram_mb', 0.0):.2f}}",
    f"used_mixed_precision={{int(p_data.get('used_mixed_precision', False))}}",
    f"used_fallback={{int(p_data.get('used_fallback', False))}}",
    f"final_residual={{p_data.get('final_residual', 0.0):.6e}}",
    f"refinement_iters={{p_data.get('refinement_iters', 0)}}",
    f"sp_krylov_iters={{p_data.get('sp_krylov_iters', 0)}}"
]
print("OUTPUT:" + ",".join(out_parts))
"""
    fd_py, path_py = tempfile.mkstemp(suffix=".py")
    with open(path_py, "w") as f:
        f.write(py_code)
    os.close(fd_py)
    
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = str(thread_count)
    env["OMP_NUM_THREADS"] = str(thread_count)
    env["MKL_NUM_THREADS"] = str(thread_count)
    
    try:
        proc = subprocess.run(
            [sys.executable, path_py],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Cleanup
        for p_file in [path_pos, path_py]:
            if os.path.exists(p_file):
                os.remove(p_file)
                
        if proc.returncode != 0:
            raise RuntimeError(f"RCR Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            
        p_info = {}
        for line in proc.stdout.splitlines():
            if line.startswith("OUTPUT:"):
                parts = line[7:].split(",")
                for p_part in parts:
                    k, v = p_part.split("=")
                    if k == "krylov_iters":
                        p_info[k] = int(v)
                    elif k == "qext":
                        p_info[k] = float(v)
                    else:
                        p_info[k] = float(v)
        return p_info
        
    except Exception as e:
        for p_file in [path_pos, path_py]:
            if os.path.exists(p_file):
                os.remove(p_file)
        raise e

def get_git_hash():
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hash_txt = os.path.join(proj_dir, "git_hash.txt")
    if os.path.exists(hash_txt):
        try:
            with open(hash_txt, "r") as f:
                h = f.read().strip()
                if h:
                    return h
        except Exception:
            pass
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=proj_dir).decode("utf-8").strip()
    except Exception:
        return "Unknown"

def get_hardware_info():
    cpu_info = "Unknown"
    gpu_info = "Unknown"
    
    # Try getting CPU model
    try:
        import platform
        sys_name = platform.system()
        if sys_name == "Windows":
            try:
                # Use powershell to query processor name
                cmd = 'powershell -Command "(Get-CimInstance Win32_Processor).Name"'
                res = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
                lines = [line.strip() for line in res.split('\n') if line.strip()]
                if lines:
                    cpu_info = lines[0] # Take first processor name
            except Exception:
                # Fallback to wmic
                try:
                    res = subprocess.check_output("wmic cpu get name", shell=True).decode("utf-8").strip()
                    lines = [line.strip() for line in res.split('\n') if line.strip()]
                    if len(lines) > 1:
                        cpu_info = lines[1]
                except Exception:
                    pass
        elif sys_name == "Darwin":
            cpu_info = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode("utf-8").strip()
        else:
            cpu_info = subprocess.check_output("grep -m 1 'model name' /proc/cpuinfo", shell=True).decode("utf-8").split(':')[-1].strip()
    except Exception:
        pass
        
    # Try getting GPU model via CuPy runtime or nvidia-smi
    try:
        import cupy as cp
        if cp.is_available():
            dev_id = cp.cuda.Device().id
            props = cp.cuda.runtime.getDeviceProperties(dev_id)
            name_val = props['name']
            if isinstance(name_val, bytes):
                gpu_info = name_val.decode("utf-8").strip()
            else:
                gpu_info = str(name_val).strip()
    except Exception:
        pass

    if gpu_info == "Unknown":
        try:
            res = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode("utf-8").strip()
            lines = [line.strip() for line in res.split('\n') if line.strip()]
            if lines:
                gpu_info = lines[0]
        except Exception:
            pass
        
    return cpu_info, gpu_info

def run_bench():
    N_list = [16, 32, 64, 128, 256]
    backends = ["direct_cpu_serial", "direct_cpu_parallel", "direct_gpu", "direct_gpu_dense"]
    thread_count_parallel = 64
    
    results = {}
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_solve_benchmark.json")
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                results = json.load(f)
            print("Loaded existing benchmark results for reuse.")
        except Exception:
            results = {}
            
    # Check if CUDA is available on this system
    cuda_avail = False
    try:
        import cupy as cp
        cuda_avail = cp.is_available()
    except Exception:
        pass
        
    cpu_model, gpu_model = get_hardware_info()
    git_hash = get_git_hash()
        
    print("==================================================")
    print("Starting Full Solve CPU vs GPU Benchmark Sweep")
    print(f"CUDA Available: {cuda_avail} | Parallel Threads: {thread_count_parallel}")
    print(f"CPU: {cpu_model} | GPU: {gpu_model}")
    print(f"Git Commit: {git_hash}")
    print("==================================================")
    
    for N in N_list:
        if str(N) not in results:
            results[str(N)] = {}
        print(f"\n--- N = {N} ---")
        
        for b in backends:
            # Reuses successful results from a previous run to save time
            # Only reuse if it contains detailed profiling data (has rhs_time) to ensure paper-grade report
            if b in results[str(N)] and results[str(N)][b].get("status") == "Success" and "rhs_time" in results[str(N)][b]:
                print(f"  Backend: {b:<20} ... Reusing previous success. Time: {results[str(N)][b]['time']:.3f} s")
                continue
            
            # Check GPU eligibility
            if b in ("direct_gpu", "direct_gpu_dense") and not cuda_avail:
                print(f"  Backend: {b:<20} -> Skipped (CUDA Unavailable)")
                results[str(N)][b] = {"time": None, "qext": None, "status": "Skipped (CUDA Unavailable)"}
                continue
                
            # Skip large CPU serial sizes to avoid long waits
            if b == "direct_cpu_serial" and N > 64:
                print(f"  Backend: {b:<20} -> Skipped (Time limit)")
                results[str(N)][b] = {"time": None, "qext": None, "status": "Skipped (Time limit)"}
                continue
                
            # Skip large CPU parallel sizes to avoid long waits
            if b == "direct_cpu_parallel" and N > 256:
                print(f"  Backend: {b:<20} -> Skipped (Time limit)")
                results[str(N)][b] = {"time": None, "qext": None, "status": "Skipped (Time limit)"}
                continue
                
            threads = thread_count_parallel if b in ("direct_cpu_parallel", "direct_gpu", "direct_gpu_dense") else 1
            print(f"  Backend: {b:<20} (T={threads}) ... ", end="", flush=True)
            
            try:
                p_info = run_subprocess_solve(N, b, threads)
                t_val = p_info.get("time", 0.0)
                q_val = p_info.get("qext", 0.0)
                print(f"Success. Time: {t_val:.3f} s | Qext: {q_val:.8f}")
                results[str(N)][b] = {
                    "status": "Success",
                    "time": t_val,
                    "qext": q_val,
                    "assembly_time": p_info.get("assembly_time", 0.0),
                    "h2d_time": p_info.get("h2d_time", 0.0),
                    "rhs_time": p_info.get("rhs_time", 0.0),
                    "solve_time": p_info.get("solve_time", 0.0),
                    "merge_time": p_info.get("merge_time", 0.0),
                    "precompute_merge_time": p_info.get("precompute_merge_time", 0.0),
                    "merge_gemm_time": p_info.get("merge_gemm_time", 0.0),
                    "orientation_time": p_info.get("orientation_time", 0.0),
                    "krylov_iters": p_info.get("krylov_iters", 0),
                    "vram_mb": p_info.get("vram_mb", 0.0),
                    "used_mixed_precision": int(p_info.get("used_mixed_precision", 0)),
                    "used_fallback": int(p_info.get("used_fallback", 0)),
                    "final_residual": p_info.get("final_residual", 0.0),
                    "refinement_iters": int(p_info.get("refinement_iters", 0)),
                    "sp_krylov_iters": int(p_info.get("sp_krylov_iters", 0))
                }
            except Exception as e:
                print(f"FAILED: {e}")
                results[str(N)][b] = {"time": None, "qext": None, "status": f"Failed: {e}"}

    # Generate Markdown report
    markdown_lines = []
    markdown_lines.append("# Full Solve CPU vs GPU Benchmark Report")
    markdown_lines.append(f"- **Metadata & Provenance**:")
    markdown_lines.append(f"  - **Git Commit**: `{git_hash}`")
    markdown_lines.append(f"  - **CPU Model**: {cpu_model}")
    markdown_lines.append(f"  - **GPU Model**: {gpu_model}")
    markdown_lines.append(f"  - **CPU Parallel Threads**: {thread_count_parallel}")
    markdown_lines.append("")
    markdown_lines.append("## Summary Table")
    markdown_lines.append("")
    markdown_lines.append("| N | Backend | Status | Solve Time (s) | Speedup vs CPU Serial | Speedup vs CPU Parallel | Qext |")
    markdown_lines.append("|---|---------|--------|----------------|------------------------|--------------------------|------|")
    
    for N in N_list:
        cpu_s_time = results[str(N)].get("direct_cpu_serial", {}).get("time")
        cpu_p_time = results[str(N)].get("direct_cpu_parallel", {}).get("time")
        
        for b in backends:
            res_b = results[str(N)][b]
            status = res_b["status"]
            t_val = res_b.get("time")
            q_val = res_b.get("qext")
            
            if t_val is not None:
                t_str = f"{t_val:.3f}"
                q_str = f"{q_val:.8f}"
                
                # Speedup vs CPU Serial
                if cpu_s_time is not None and cpu_s_time > 0:
                    su_s = f"{cpu_s_time / t_val:.2f}x"
                else:
                    su_s = "N/A"
                    
                # Speedup vs CPU Parallel
                if cpu_p_time is not None and cpu_p_time > 0:
                    su_p = f"{cpu_p_time / t_val:.2f}x"
                else:
                    su_p = "N/A"
            else:
                t_str = "N/A"
                q_str = "N/A"
                su_s = "N/A"
                su_p = "N/A"
                
            markdown_lines.append(
                f"| {N} | `{b}` | {status} | {t_str} | {su_s} | {su_p} | {q_str} |"
            )
            
    markdown_lines.append("")
    markdown_lines.append("## Detailed Phase Breakdown")
    markdown_lines.append("")
    markdown_lines.append("| N | Backend | Total Time (s) | Assembly (s) | H2D (s) | RHS Build (s) | Solve/Krylov (s) | Merge Total (s) | Merge Precompute (s) | Merge GEMM (ms) | Orientation (s) | Iters (SP/DP/Ref) | Final Res | Fallback? | VRAM (MB) |")
    markdown_lines.append("|---|---------|----------------|--------------|---------|---------------|------------------|-----------------|----------------------|----------------|-----------------|-------------------|-----------|-----------|-----------|")
    
    for N in N_list:
        for b in backends:
            res_b = results[str(N)][b]
            if res_b["status"] != "Success":
                continue
                
            total_time = res_b.get("time", 0.0)
            asm_time = res_b.get("assembly_time", 0.0)
            h2d_time = res_b.get("h2d_time", 0.0)
            rhs_time = res_b.get("rhs_time", 0.0)
            solve_time = res_b.get("solve_time", 0.0)
            merge_time = res_b.get("merge_time", 0.0)
            pre_merge = res_b.get("precompute_merge_time", 0.0)
            gemm_merge = res_b.get("merge_gemm_time", 0.0)
            ori_time = res_b.get("orientation_time", 0.0)
            vram = res_b.get("vram_mb", 0.0)
            
            # Format Iterations
            sp_iters = int(res_b.get("sp_krylov_iters", 0))
            dp_iters = int(res_b.get("krylov_iters", 0))
            ref_iters = int(res_b.get("refinement_iters", 0))
            iter_str = f"{sp_iters} / {dp_iters} / {ref_iters}"
            
            # Format Residual & Fallback
            if res_b.get("used_mixed_precision", 0) or "final_residual" in res_b:
                final_res_val = res_b.get("final_residual", 0.0)
                res_str = f"{final_res_val:.2e}"
                fallback_str = "Yes" if res_b.get("used_fallback", 0) else "No"
            else:
                res_str = "N/A"
                fallback_str = "N/A"
            
            markdown_lines.append(
                f"| {N} | `{b}` | {total_time:.3f} | {asm_time:.4f} | {h2d_time:.4f} | {rhs_time:.4f} | {solve_time:.3f} | {merge_time:.4f} | {pre_merge:.4f} | {gemm_merge * 1000:.3f} | {ori_time:.4f} | {iter_str} | {res_str} | {fallback_str} | {vram:.1f} |"
            )
            
    summary_md = "\n".join(markdown_lines)
    
    # Save results
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
        
    out_md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_solve_gpu_vs_cpu.md")
    with open(out_md, "w") as f:
        f.write(summary_md)
        
    print("\n" + summary_md + "\n")
    print(f"Results saved to:\n  - JSON: {out_json}\n  - Markdown: {out_md}")

if __name__ == "__main__":
    run_bench()
