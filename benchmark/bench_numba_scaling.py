#!/usr/bin/env python3
"""Benchmark runner for a single N and thread configuration.

Designed to be called as a subprocess to accurately control Numba thread counts.
"""

import sys
import time
import argparse
import numpy as np

# Add parent path to import modules
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.geometry import build_bcca
from solver.bicgstab_solver import MultisphereRCRSystem, apply_tmatrix_model1

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

def run_bench(n: int, p: int = 7, num_mvm: int = 10):
    # Setup Geometry
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
    
    # Initialize Multisphere RCR System
    t0 = time.perf_counter()
    system = MultisphereRCRSystem(positions_dimless, nodr_list, an_list, k_medium=1.0)
    init_time = time.perf_counter() - t0
    
    # Preallocate a random scattered vector X
    N_total = system.N_total
    rng = np.random.default_rng(42)
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
    
    # 1. Warm up JIT Compiler (separates compile time from core execution)
    t0 = time.perf_counter()
    _ = system.apply_system_operator(X)
    warmup_time = time.perf_counter() - t0
    
    # 2. Run MVM iterations
    mvm_times = []
    for _ in range(num_mvm):
        t0 = time.perf_counter()
        _ = system.apply_system_operator(X)
        mvm_times.append(time.perf_counter() - t0)
        
    avg_mvm = np.mean(mvm_times)
    std_mvm = np.std(mvm_times)
    
    print(f"RESULT:init_time={init_time:.6f},warmup_time={warmup_time:.6f},avg_mvm={avg_mvm:.6f},std_mvm={std_mvm:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=16, help="Number of particles")
    parser.add_argument("-p", type=int, default=7, help="Truncation order")
    parser.add_argument("--mvm", type=int, default=10, help="Number of Matvecs to run")
    args = parser.parse_args()
    
    run_bench(args.n, args.p, args.mvm)
