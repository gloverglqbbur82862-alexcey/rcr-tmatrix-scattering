#!/usr/bin/env python3
"""Benchmark solver scaling: Direct pairwise JIT vs FMM JIT."""

import time
import numpy as np
import scipy.special as sp
from geometry.geometry import build_bcca
from solver.bicgstab_solver import MultisphereRCRSystem, apply_tmatrix_model1
from fmm.fmm_solver import FMMMultisphereSystem

def mie_coefficients_manual(m: complex, x: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    an_list = []
    bn_list = []
    
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

def benchmark_n(n: int, p: int = 7):
    print(f"\n=================== Benchmarking N = {n} ===================")
    
    # 1. Geometry Setup
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm
    
    positions, _ = build_bcca(n_monomers=n, radius=r_monomer_nm, overlap=0.0, seed=42)
    positions_dimless = positions * k_medium
    r_dimless = r_monomer_nm * k_medium
    
    nodr_list = [p] * n
    m_complex = 1.95 + 0.79j
    a_n, b_n = mie_coefficients_manual(m_complex, r_dimless, p)
    
    an = np.zeros((2, 2, p), dtype=np.complex128)
    an[0, 0, :] = a_n
    an[1, 1, :] = b_n
    an_list = [an] * n
    
    # Precompute dummy plane-wave RHS (polarization 1)
    pmnp0_base = np.zeros((2 * p * (p + 2), 2), dtype=np.complex128)
    pmnp0_base[0, 0] = 1.0  # simple dipole excitation
    
    b_m1_blocks = []
    for i in range(n):
        phase = np.exp(1j * positions_dimless[i][2])
        p_m1 = pmnp0_base[:, 0] * phase
        b_m1_blocks.append(apply_tmatrix_model1(p_m1, an, p))
    b_m1 = np.concatenate(b_m1_blocks)
    
    # 2. Bench Direct pairwise JIT
    print("  Initializing direct RCR system...")
    t0 = time.perf_counter()
    direct_system = MultisphereRCRSystem(positions_dimless, nodr_list, an_list, k_medium=1.0)
    init_direct_time = time.perf_counter() - t0
    
    print("  Solving via direct RCR system (Matrix-Free BiCGSTAB)...")
    t0 = time.perf_counter()
    x_direct, res_direct = direct_system.solve(b_m1, tol=1e-12, max_iter=200)
    solve_direct_time = time.perf_counter() - t0
    
    print(f"    Direct RCR solved. residual: {res_direct:.2e}, solve time: {solve_direct_time:.4f}s")
    
    # 3. Bench FMM JIT
    print("  Initializing FMM RCR system...")
    t0 = time.perf_counter()
    fmm_system = FMMMultisphereSystem(
        positions_dimless, nodr_list, an_list, k_medium=1.0, p_fmm=p, max_per_leaf=4, max_level=8
    )
    init_fmm_time = time.perf_counter() - t0
    
    print("  Solving via FMM RCR system (Matrix-Free BiCGSTAB)...")
    t0 = time.perf_counter()
    x_fmm, res_fmm = fmm_system.solve(b_m1, tol=1e-12, max_iter=200)
    solve_fmm_time = time.perf_counter() - t0
    
    print(f"    FMM RCR solved. residual: {res_fmm:.2e}, solve time: {solve_fmm_time:.4f}s")
    
    # Verify correctness between them
    diff = np.linalg.norm(x_direct - x_fmm)
    rel_err = diff / np.linalg.norm(x_direct)
    print(f"    Agreement between FMM and Direct: {rel_err:.2e}")
    
    return solve_direct_time, solve_fmm_time

if __name__ == "__main__":
    # Warm up JIT compiling first
    print("Warming up JIT compiler...")
    benchmark_n(8, p=3)
    
    # Large scale tests
    n_values = [16, 32, 64]
    results = {}
    for n in n_values:
        td, tf = benchmark_n(n, p=7)
        results[n] = (td, tf)
        
    print("\n\n=================== Scaling Benchmark Summary ===================")
    print(f"{'N':<6} | {'Direct RCR Solve (s)':<22} | {'FMM RCR Solve (s)':<20} | {'FMM Speedup':<12}")
    print("-" * 70)
    for n in n_values:
        td, tf = results[n]
        speedup = td / tf if tf > 0 else 0.0
        print(f"{n:<6} | {td:<22.4f} | {tf:<20.4f} | {speedup:<12.1f}x")
