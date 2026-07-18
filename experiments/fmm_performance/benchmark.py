#!/usr/bin/env python3
"""Benchmark script for FMM vs Direct JIT Matvec performance.
Measures initialization time, warmup time, and average matvec execution time.
Includes Speed and Reference mode configurations.
"""

import os
import sys
import time
import json
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geometry.geometry import build_bcca
from solver.bicgstab_solver import MultisphereRCRSystem, apply_tmatrix_model1
from fmm.fmm_solver import FMMMultisphereSystem

def make_mie_coefficients(p: int) -> np.ndarray:
    """Build a diagonal Mie coefficient array for testing."""
    an = np.zeros((2, 2, p), dtype=np.complex128)
    for n in range(p):
        an[0, 0, n] = -0.5 + 0.1j * n  # TM
        an[1, 1, n] = -0.3 + 0.05j * n  # TE
    return an

def build_random_cluster(n_monomers: int, radius: float, seed: int = 42) -> np.ndarray:
    """Generate a random non-overlapping cluster of equal-sized spheres in a box."""
    rng = np.random.default_rng(seed)
    positions = []
    # Determine box size based on density
    box_size = radius * (n_monomers ** (1.0 / 3.0)) * 4.0
    attempts = 0
    while len(positions) < n_monomers and attempts < 100000:
        pos = rng.uniform(-box_size, box_size, size=3)
        if len(positions) == 0:
            positions.append(pos)
        else:
            dists = np.linalg.norm(np.array(positions) - pos, axis=1)
            if np.all(dists >= 2.0 * radius):
                positions.append(pos)
        attempts += 1
    if len(positions) < n_monomers:
        # Fallback to sparse grid to guarantee success
        positions = []
        grid_size = int(np.ceil(n_monomers ** (1.0 / 3.0)))
        spacing = 2.1 * radius
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(positions) < n_monomers:
                        positions.append(np.array([i, j, k]) * spacing)
    return np.array(positions)

def run_benchmark():
    print("==================================================")
    print("Starting FMM vs Direct Matvec Performance Benchmark")
    print("==================================================")

    # Physical parameters
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm

    # Benchmark configurations
    N_list = [128, 256, 512, 1024]
    p_list = [3, 5, 7]

    # Warm up JIT compiler first
    print("Warming up JIT compiler...")
    pos_warm = build_random_cluster(8, radius=r_monomer_nm, seed=42)
    pos_warm_dimless = pos_warm * k_medium
    nodr_warm = [3] * 8
    an_warm = [make_mie_coefficients(3)] * 8
    
    direct_warm = MultisphereRCRSystem(pos_warm_dimless, nodr_warm, an_warm, k_medium=1.0)
    fmm_warm = FMMMultisphereSystem(
        pos_warm_dimless, nodr_warm, an_warm, k_medium=1.0, 
        p_fmm=3, max_per_leaf=4, max_level=8, use_fmm=True,
        fmm_mode="speed", fmm_tol=1e-6, theta=0.55
    )
    X_warm = np.random.default_rng(42).standard_normal(direct_warm.N_total) + 1j * np.random.default_rng(42).standard_normal(direct_warm.N_total)
    
    # Run once to compile
    direct_warm.apply_system_operator(X_warm)
    fmm_warm.apply_system_operator(X_warm)
    print("JIT Warmup complete.\n")

    results = {}

    for N in N_list:
        results[str(N)] = {}
        for p in p_list:
            print(f"--- Running N={N}, p={p} ---")
            
            # Setup geometry: try BCCA first, fall back to random cluster if it fails
            is_bcca = True
            try:
                positions, _ = build_bcca(n_monomers=N, radius=r_monomer_nm, overlap=0.0, seed=42)
                print(f"  Using BCCA soot geometry.")
            except Exception:
                positions = build_random_cluster(N, radius=r_monomer_nm, seed=42)
                is_bcca = False
                print(f"  Using random non-overlapping geometry (BCCA fallback).")

            positions_dimless = positions * k_medium
            nodr_list = [p] * N
            an_list = [make_mie_coefficients(p)] * N

            # Setup input vector
            nblk_half = p * (p + 2)
            N_total = 2 * nblk_half * N
            rng = np.random.default_rng(42)
            X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)

            direct_results = {
                "init_time": None,
                "warmup_time": None,
                "matvec_time": None
            }

            # Skip direct solver for N=1024 to prevent memory overflow (d_blocks matrix can be >5GB)
            if N == 1024:
                print("  [Direct] Skipping N=1024 to prevent memory overflow.")
            else:
                # 1. Bench Direct JIT
                print("  Initializing direct solver...")
                t0 = time.perf_counter()
                direct_sys = MultisphereRCRSystem(positions_dimless, nodr_list, an_list, k_medium=1.0)
                direct_results["init_time"] = time.perf_counter() - t0

                print("  Running direct matvec (warmup)...")
                t0 = time.perf_counter()
                AX_direct = direct_sys.apply_system_operator(X)
                direct_results["warmup_time"] = time.perf_counter() - t0

                print("  Running direct matvec iterations...")
                direct_times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    direct_sys.apply_system_operator(X)
                    direct_times.append(time.perf_counter() - t0)
                direct_results["matvec_time"] = float(np.mean(direct_times))
                print(f"    Direct: init={direct_results['init_time']:.4f}s, warmup={direct_results['warmup_time']:.4f}s, avg_matvec={direct_results['matvec_time']:.4f}s")

            # 2. Bench FMM Reference (Correctness target, fmm_tol=1e-12)
            fmm_ref_results = {
                "init_time": None,
                "warmup_time": None,
                "matvec_time": None
            }
            rel_err_ref = None

            # Skip FMM Reference for N >= 512 because pure Python loops under reference mode are extremely slow
            if N >= 512:
                print(f"  [FMM Reference] Skipping N={N}, p={p} to avoid excessive runtimes.")
            else:
                print("  Initializing FMM reference solver...")
                t0 = time.perf_counter()
                fmm_ref = FMMMultisphereSystem(
                    positions_dimless, nodr_list, an_list, k_medium=1.0,
                    p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
                    fmm_mode="reference", fmm_tol=1e-12, theta=0.55
                )
                fmm_ref_results["init_time"] = time.perf_counter() - t0

                print("  Running FMM reference matvec (warmup)...")
                t0 = time.perf_counter()
                AX_fmm_ref = fmm_ref.apply_system_operator(X)
                fmm_ref_results["warmup_time"] = time.perf_counter() - t0

                print("  Running FMM reference matvec iterations...")
                ref_times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    fmm_ref.apply_system_operator(X)
                    ref_times.append(time.perf_counter() - t0)
                fmm_ref_results["matvec_time"] = float(np.mean(ref_times))
                print(f"    FMM Ref: init={fmm_ref_results['init_time']:.4f}s, warmup={fmm_ref_results['warmup_time']:.4f}s, avg_matvec={fmm_ref_results['matvec_time']:.4f}s")

                if N != 1024:
                    diff = np.linalg.norm(AX_fmm_ref - AX_direct)
                    rel_err_ref = float(diff / np.linalg.norm(AX_direct))
                    print(f"    Ref L2 error vs Direct: {rel_err_ref:.4e}")

            # 3. Bench FMM Speed (Performance target, fmm_tol=1e-6)
            fmm_speed_results = {
                "init_time": None,
                "warmup_time": None,
                "matvec_time": None
            }
            rel_err_speed = None

            print("  Initializing FMM speed solver...")
            t0 = time.perf_counter()
            fmm_speed = FMMMultisphereSystem(
                positions_dimless, nodr_list, an_list, k_medium=1.0,
                p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
                fmm_mode="speed", fmm_tol=1e-6, theta=0.55
            )
            fmm_speed_results["init_time"] = time.perf_counter() - t0

            # Log expansion order sizes for verification
            p_levels = fmm_speed.fmm_op.p_levels
            print(f"    FMM Speed cell orders per level (P_l): {p_levels}")

            print("  Running FMM speed matvec (warmup)...")
            t0 = time.perf_counter()
            AX_fmm_speed = fmm_speed.apply_system_operator(X)
            fmm_speed_results["warmup_time"] = time.perf_counter() - t0

            print("  Running FMM speed matvec iterations...")
            speed_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                fmm_speed.apply_system_operator(X)
                speed_times.append(time.perf_counter() - t0)
            fmm_speed_results["matvec_time"] = float(np.mean(speed_times))
            print(f"    FMM Speed: init={fmm_speed_results['init_time']:.4f}s, warmup={fmm_speed_results['warmup_time']:.4f}s, avg_matvec={fmm_speed_results['matvec_time']:.4f}s")

            if N != 1024:
                diff = np.linalg.norm(AX_fmm_speed - AX_direct)
                rel_err_speed = float(diff / np.linalg.norm(AX_direct))
                print(f"    Speed L2 error vs Direct: {rel_err_speed:.4e}")

            results[str(N)][str(p)] = {
                "is_bcca": is_bcca,
                "direct": direct_results,
                "fmm_reference": fmm_ref_results,
                "fmm_speed": fmm_speed_results,
                "rel_err_reference": rel_err_ref,
                "rel_err_speed": rel_err_speed,
                "p_levels": p_levels
            }

    # Save to JSON
    os.makedirs(os.path.join("experiments", "fmm_performance"), exist_ok=True)
    out_path = os.path.join("experiments", "fmm_performance", "fmm_baseline_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to {out_path}\n")

if __name__ == "__main__":
    run_benchmark()
