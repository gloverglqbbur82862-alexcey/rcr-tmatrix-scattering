#!/usr/bin/env python3
"""Run complete physical solve comparisons: Native Fortran MSTM vs. Our JIT RCR Engine."""

import os
import sys
import time
import json
import subprocess
import argparse
import numpy as np

# Add parent path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.geometry import build_bcca
from solver.bicgstab_solver import MultisphereRCRSystem, apply_tmatrix_model1
from tests.test_rcr_translation import vector_model2_to_model1

def mie_coefficients_manual(m: complex, x: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import spherical_jn, spherical_yn
    def psi(nn: int, z: complex) -> complex:
        return z * spherical_jn(nn, z)
    def dpsi(nn: int, z: complex) -> complex:
        return spherical_jn(nn, z) + z * spherical_jn(nn, z, derivative=True)
    def xi(nn: int, z: complex) -> complex:
        return z * (spherical_jn(nn, z) + 1j * spherical_yn(nn, z))
    def dxi(nn: int, z: complex) -> complex:
        h = spherical_jn(nn, z) + 1j * spherical_yn(nn, z)
        dh = (spherical_jn(nn, z, derivative=True) + 1j * spherical_yn(nn, z, derivative=True))
        return h + z * dh
        
    an_list = []
    bn_list = []
    for n in range(1, n_max + 1):
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
    return np.array(an_list), np.array(bn_list)

def reproduce_incident_coefficients_simple(p: int) -> np.ndarray:
    """Simplified model-2 plane-wave excitation coefficients (polarization 1)."""
    pmnp0 = np.zeros((2 * p * (p + 2), 2), dtype=np.complex128)
    pmnp0[0, 0] = 1.0  # dipole-like excitation
    return pmnp0

def write_mstm_input(inp_path: str, out_path: str, positions: np.ndarray, r_dimless: float, m_complex: complex):
    n = len(positions)
    out_filename = os.path.basename(out_path)
    lines = [
        "output_file",
        out_filename,
        "number_spheres",
        str(n),
        "sphere_data"
    ]
    for i in range(n):
        pos = positions[i]
        lines.append(f"{pos[0]:.8f},{pos[1]:.8f},{pos[2]:.8f},{r_dimless:.8f},({m_complex.real:.6f},{m_complex.imag:.6f})")
    lines.append("end_of_sphere_data")
    lines.append("length_scale_factor")
    lines.append("1.d0")
    lines.append("incident_beta_deg")
    lines.append("0.d0")
    lines.append("incident_alpha_deg")
    lines.append("0.0d0")
    lines.append("calculate_scattering_matrix")
    lines.append("t")
    lines.append("calculate_near_field")
    lines.append("f")
    lines.append("random_orientation")
    lines.append("t")
    lines.append("solution_epsilon")
    lines.append("1.d-12")
    lines.append("translation_epsilon")
    lines.append("1.d-12")
    lines.append("end_of_options")
    
    with open(inp_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def run_mstm_process(mstm_bin: str, inp_path: str) -> float:
    t0 = time.perf_counter()
    inp_filename = os.path.basename(inp_path)
    proc = subprocess.run(
        [mstm_bin, inp_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(inp_path)
    )
    t_elap = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"MSTM execution failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return t_elap

def parse_mstm_observables(out_path: str) -> tuple[float, int]:
    """Parse MSTM output to extract extinction efficiency Qext and t_matrix_order."""
    qext = 0.0
    t_matrix_order = 11
    if not os.path.exists(out_path):
        return qext, t_matrix_order
    with open(out_path) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if "estimated t matrix order:" in line.lower():
                try:
                    t_matrix_order = int(lines[idx + 1].strip())
                except:
                    pass
            elif "total extinction" in line.lower() and "efficiencies" in line.lower():
                # Values are on the next line
                next_line = lines[idx + 1].strip()
                parts = next_line.split()
                try:
                    qext = float(parts[0])
                except:
                    pass
            elif "t matrix ext, sca:" in line.lower():
                parts = line.strip().split()
                try:
                    qext = float(parts[-2])
                except:
                    pass
            elif "total ext" in line.lower() and "efficiency" in line.lower():
                parts = line.strip().split()
                try:
                    qext = float(parts[-3])
                except:
                    pass
    return qext, t_matrix_order

def run_comparison(n_values: list[int], thread_counts: list[int], mstm_bin: str, use_numa: bool):
    os.makedirs("experiments", exist_ok=True)
    results = {}
    
    # Check if MSTM binary exists
    if not os.path.exists(mstm_bin):
        print(f"ERROR: Native MSTM binary not found at {mstm_bin}")
        print("Please check your build path or compile MSTM first.")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("🚀 Running Native Fortran MSTM vs. Our RCR JIT Solver")
    print(f"   MSTM Binary: {mstm_bin}")
    print(f"   Particle Scales (N): {n_values}")
    print(f"   Optimal Thread Counts: {thread_counts}")
    print("=" * 80 + "\n")
    
    # Common physical parameters (BCCA Soot cluster)
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm
    r_dimless = r_monomer_nm * k_medium
    m_complex = 1.95 + 0.79j
    p = 3
    
    for n in n_values:
        results[n] = {}
        print(f"\n--- Benchmark N = {n} ---")
        
        # 1. Geometry Setup
        positions, _ = build_bcca(n_monomers=n, radius=r_monomer_nm, overlap=0.0, seed=42)
        positions_dimless = positions * k_medium
        
        # 2. Run Native Compiled Fortran MSTM (Serial)
        inp_path = f"experiments/mstm_temp_n{n}.inp"
        out_path = f"experiments/mstm_temp_n{n}.out"
        write_mstm_input(inp_path, out_path, positions_dimless, r_dimless, m_complex)
        
        print(f"  Running MSTM native Fortran...", end="", flush=True)
        t_mstm = run_mstm_process(mstm_bin, inp_path)
        qext_mstm, agg_order = parse_mstm_observables(out_path)
        print(f" Completed. Time: {t_mstm:.3f} s | Qext: {qext_mstm:.8f} | AggOrder: {agg_order}")
        
        results[n]["mstm"] = {
            "time": t_mstm,
            "qext": qext_mstm
        }
        
        # 3. Setup JIT Solver
        nodr_list = [p] * n
        an_list = [_mie_array(m_complex, r_dimless, p)] * n
        pmnp0_base = reproduce_incident_coefficients_simple(p)
        
        b_m1_blocks = []
        for i in range(n):
            phase = np.exp(1j * positions_dimless[i][2])
            p_m1 = vector_model2_to_model1(pmnp0_base[:, 0], p) * phase
            b_m1_blocks.append(apply_tmatrix_model1(p_m1, an_list[0], p))
        b_m1 = np.concatenate(b_m1_blocks)
        
        # 4. Run our JIT Solver for T = 1 (Single-Core)
        if n < 128:
            print("  Running JIT Solver (Single-Core T=1)... ", end="", flush=True)
            t_rcr_s, qext_rcr_s, res_s, iters_s = run_rcr_solve(positions_dimless, nodr_list, an_list, b_m1, r_dimless, agg_order, thread_count=1, use_numa=False, use_fmm=False)
            print(f"Completed. Time: {t_rcr_s:.3f} s | Qext: {qext_rcr_s:.8f} | Iters: {iters_s} | Res: {res_s:.2e}")
            
            results[n]["rcr_single"] = {
                "time": t_rcr_s,
                "qext": qext_rcr_s
            }
        else:
            t_rcr_s = 0.0
            results[n]["rcr_single"] = {"time": 0.0, "qext": 0.0}
        
        # 5. Run our JIT Solver for Multi-Core
        for t in thread_counts:
            # Skip if thread count is 1
            if t == 1:
                continue
            print(f"  Running JIT Solver (Multi-Core T={t:2d})... ", end="", flush=True)
            t_rcr_p, qext_rcr_p, res_p, iters_p = run_rcr_solve(positions_dimless, nodr_list, an_list, b_m1, r_dimless, agg_order, thread_count=t, use_numa=use_numa, use_fmm=False)
            print(f"Completed. Time: {t_rcr_p:.3f} s | Qext: {qext_rcr_p:.8f} | Iters: {iters_p} | Res: {res_p:.2e}")
            
            results[n][f"rcr_multi_t{t}"] = {
                "time": t_rcr_p,
                "qext": qext_rcr_p
            }

        # 6. Run our FMM Cascade Solver
        if n < 128:
            print("  Running FMM Cascade Solver (Serial)... ", end="", flush=True)
            t_fmm, qext_fmm, res_fmm, iters_fmm = run_rcr_solve(positions_dimless, nodr_list, an_list, b_m1, r_dimless, agg_order, thread_count=1, use_numa=False, use_fmm=True)
            print(f"Completed. Time: {t_fmm:.3f} s | Qext: {qext_fmm:.8f} | Iters: {iters_fmm} | Res: {res_fmm:.2e}")
            
            results[n]["fmm_serial"] = {
                "time": t_fmm,
                "qext": qext_fmm
            }
        else:
            t_fmm = 0.0
            results[n]["fmm_serial"] = {"time": 0.0, "qext": 0.0}
            
    # Save comparison data
    out_json = "experiments/full_solve_comparison.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 90)
    print("📊 FULL SOLVE COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'N':<5} | {'MSTM Fortran (s)':<18} | {'RCR Single (s)':<15} | {'RCR Multi (s)':<15} | {'FMM Cascade (s)':<15} | {'Speedup vs MSTM':<18}")
    print("-" * 90)
    for n in n_values:
        t_m = results[n]["mstm"]["time"]
        t_s = results[n]["rcr_single"]["time"]
        t_fmm = results[n].get("fmm_serial", {}).get("time", 0.0)
        
        # Find best parallel time
        t_p_best = float("inf")
        best_t_val = None
        for k, v in results[n].items():
            if k.startswith("rcr_multi_t"):
                if v["time"] < t_p_best:
                    t_p_best = v["time"]
                    best_t_val = int(k.replace("rcr_multi_t", ""))
                    
        if best_t_val is None:
            t_p_str = "N/A"
            best_t_str = "N/A"
            speedup = 0.0
        else:
            t_p_str = f"{t_p_best:.3f}"
            best_t_str = f"{best_t_val:2d}"
            
            # The speedup could be from FMM if FMM is faster!
            best_time = min(t_p_best, t_fmm) if t_fmm > 0 else t_p_best
            speedup = t_m / best_time if best_time > 0 else 0.0
            
        print(f"{n:<5d} | {t_m:<18.3f} | {t_s:<15.3f} | {t_p_str:<15s} | {t_fmm:<15.3f} | {speedup:<16.2f}x")
    print("=" * 90 + "\n")

def _mie_array(m: complex, r: float, p: int) -> np.ndarray:
    a_n, b_n = mie_coefficients_manual(m, r, p)
    an = np.zeros((2, 2, p), dtype=np.complex128)
    diag = -0.5 * (a_n + b_n)
    off = -0.5 * (a_n - b_n)
    an[0, 0, :] = diag
    an[1, 1, :] = diag
    an[0, 1, :] = off
    an[1, 0, :] = off
    return an

def run_rcr_solve(positions, nodr_list, an_list, b_vec, r_dimless, agg_order: int, thread_count: int, use_numa: bool, use_fmm: bool = False) -> tuple[float, float, float, int]:
    """Execute complete physical solver inside a subprocess to isolate threads."""
    import tempfile
    
    # We will use temporary files to pass arrays and receive timing
    fd_pos, path_pos = tempfile.mkstemp(suffix=".npy")
    fd_an, path_an = tempfile.mkstemp(suffix=".npy")
    fd_b, path_b = tempfile.mkstemp(suffix=".npy")
    
    np.save(path_pos, positions)
    np.save(path_an, an_list[0])
    np.save(path_b, b_vec)
    
    os.close(fd_pos)
    os.close(fd_an)
    os.close(fd_b)
    
    # Python code to run in subprocess
    p_val = nodr_list[0]
    
    if use_fmm:
        import_stmt = "from fmm.fmm_solver import FMMMultisphereSystem as SolverSys"
        # Set p_fmm to physical order + 4 for exact precision
        sys_init = f"system = SolverSys(pos, nodr_list, an_list, k_medium=1.0, p_fmm={p_val+4}, use_fmm=True)"
        warmup_sys = f"dummy_system = SolverSys(dummy_pos, [{p_val}, {p_val}], [an_single, an_single], k_medium=1.0, p_fmm={p_val+4}, use_fmm=True)"
    else:
        import_stmt = "from solver.bicgstab_solver import MultisphereRCRSystem as SolverSys"
        sys_init = "system = SolverSys(pos, nodr_list, an_list, k_medium=1.0)"
        warmup_sys = f"dummy_system = SolverSys(dummy_pos, [{p_val}, {p_val}], [an_single, an_single], k_medium=1.0)"

    py_code = f"""
import sys
sys.path.insert(0, '/Users/kaer/doc/fast_mtp_rcr_scattering')
import numpy as np
import time
{import_stmt}

pos = np.load("{path_pos}")
an_single = np.load("{path_an}")
b = np.load("{path_b}")

nsph = len(pos)
nodr_list = [{p_val}] * nsph
an_list = [an_single] * nsph

# Warm up JIT compiling first (run a single dummy iteration)
dummy_pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
{warmup_sys}
dummy_b = np.zeros(2 * {p_val} * ({p_val} + 2) * 2, dtype=np.complex128)
dummy_b[0] = 1.0
_ = dummy_system.solve(dummy_b, tol=1e-12, max_iter=2)

# Start measuring full solve
t0 = time.perf_counter()
{sys_init}
x_sol, res = system.solve(b, tol=1e-12, max_iter=2000)

# Calculate orientation averages
radii_dimless = [{r_dimless}] * nsph
T_agg = system.calculate_aggregate_tmatrix(aggregate_order={agg_order}, tol=1e-12)
averages = system.calculate_orientation_averages(T_agg, radii_dimless)
t_elap = time.perf_counter() - t0

print(f"OUTPUT:time={{t_elap:.6f}},qext={{averages['Qext']:.8f}},res={{res:.4e}},iters=-1")
"""
    
    fd_py, path_py = tempfile.mkstemp(suffix=".py")
    with open(path_py, "w") as f:
        f.write(py_code)
    os.close(fd_py)
    
    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = str(thread_count)
    env["OMP_NUM_THREADS"] = str(thread_count)
    env["MKL_NUM_THREADS"] = str(thread_count)
    
    cmd = [sys.executable, path_py]
    if use_numa:
        cmd = ["numactl", "--interleave=all"] + cmd
        
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None,
            env=env
        )
        
        # Cleanup
        for p in [path_pos, path_an, path_b, path_py]:
            if os.path.exists(p):
                os.remove(p)
                
        if proc.returncode != 0:
            raise RuntimeError(f"RCR Solver subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            
        t_val = 0.0
        q_val = 0.0
        res_val = 0.0
        iters_val = 0
        for line in proc.stdout.splitlines():
            if line.startswith("OUTPUT:"):
                parts = line[7:].split(",")
                for p in parts:
                    k, v = p.split("=")
                    if k == "time":
                        t_val = float(v)
                    elif k == "qext":
                        q_val = float(v)
                    elif k == "res":
                        res_val = float(v)
                    elif k == "iters":
                        iters_val = int(v)
                        
        return t_val, q_val, res_val, iters_val
        
    except Exception as e:
        for p in [path_pos, path_an, path_b, path_py]:
            if os.path.exists(p):
                os.remove(p)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, default="16,32,64,128", help="Particle count list")
    parser.add_argument("-t", type=str, default="1,16,24,48,64,128", help="Thread list")
    parser.add_argument("--bin", type=str, default="/root/mstm_build/mstm", help="MSTM Fortran binary")
    parser.add_argument("--numa", action="store_true", help="Wrap RCR with numactl --interleave=all")
    args = parser.parse_args()
    
    n_vals = [int(x) for x in args.n.split(",")]
    t_vals = [int(x) for x in args.t.split(",")]
    
    run_comparison(n_vals, t_vals, args.bin, args.numa)
