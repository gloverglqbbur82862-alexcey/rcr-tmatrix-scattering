#!/usr/bin/env python3
"""Tests for Analytical Orientation Averaging: verifying regular translation invariance and Mie limits."""

import sys
import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from solver.bicgstab_solver import MultisphereRCRSystem


def mie_coefficients_manual(m_complex: complex, x: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    m = m_complex
    an_list = []
    bn_list = []
    
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


def test_mie_limit_and_translational_invariance():
    """Verify that the analytical orientation averaging matches the exact Mie series,
    and is perfectly invariant under arbitrary spatial translations (testing RCR regular translations).
    """
    # ── 1. Setup Physical Sphere parameters ──────────────────────────────────
    radius_x = 0.22847947
    m_complex = 1.95 + 0.79j
    p_sphere = 3
    p_agg = 12
    
    a, b = mie_coefficients_manual(m_complex, radius_x, p_sphere)
    
    # Standard Mie efficiencies
    qext_mie = 0.0
    qsca_mie = 0.0
    for n in range(1, p_sphere + 1):
        qext_mie += (2 * n + 1) * (a[n-1] + b[n-1]).real
        qsca_mie += (2 * n + 1) * (abs(a[n-1])**2 + abs(b[n-1])**2)
    qext_mie *= 2.0 / (radius_x**2)
    qsca_mie *= 2.0 / (radius_x**2)
    
    # Format Mie coefficients for RCR (T-matrix elements are -a and -b)
    an_rcr = np.zeros((2, 2, p_sphere), dtype=np.complex128)
    an_rcr[0, 0, :] = -a
    an_rcr[1, 1, :] = -b
    
    # ── 2. Run for Sphere at Origin ──────────────────────────────────────────
    rcr_system_origin = MultisphereRCRSystem(
        positions=np.array([[0.0, 0.0, 0.0]]),
        nodr_list=[p_sphere],
        an_list=[an_rcr],
        k_medium=1.0
    )
    
    T_agg_origin = rcr_system_origin.calculate_aggregate_tmatrix(p_agg, tol=1e-14)
    q_origin = rcr_system_origin.calculate_orientation_averages(T_agg_origin, [radius_x])
    
    print("\n  Sphere at Origin:")
    print(f"    Mie Qext: {qext_mie:.10f}, Qsca: {qsca_mie:.10f}")
    print(f"    RCR Qext: {q_origin['Qext']:.10f}, Qsca: {q_origin['Qsca']:.10f}")
    
    assert abs(q_origin["Qext"] - qext_mie) < 1e-13
    assert abs(q_origin["Qsca"] - qsca_mie) < 1e-13
    
    # ── 3. Run for Sphere Displaced from Origin ──────────────────────────────
    # We place the sphere at an arbitrary off-origin position
    displacements = [
        np.array([1.2, -0.8, 0.5]),
        np.array([-2.5, 0.0, 1.8]),
    ]
    
    for idx, pos in enumerate(displacements):
        rcr_system_displaced = MultisphereRCRSystem(
            positions=np.array([pos]),
            nodr_list=[p_sphere],
            an_list=[an_rcr],
            k_medium=1.0
        )
        
        T_agg_displaced = rcr_system_displaced.calculate_aggregate_tmatrix(p_agg, tol=1e-14)
        q_displaced = rcr_system_displaced.calculate_orientation_averages(T_agg_displaced, [radius_x])
        
        print(f"  Displaced Sphere {idx+1} (pos={pos}):")
        print(f"    RCR Qext: {q_displaced['Qext']:.10f}, Qsca: {q_displaced['Qsca']:.10f}")
        
        # Test translational invariance
        assert abs(q_displaced["Qext"] - qext_mie) < 1e-12, f"Qext translational invariance failed for pos={pos}"
        assert abs(q_displaced["Qsca"] - qsca_mie) < 1e-12, f"Qsca translational invariance failed for pos={pos}"
        
    print("  Analytical Orientation Averaging successfully validated to 10^-12 precision under spatial translation!")


def test_bcca16_analytical_orientation_average():
    """Verify analytical orientation-averaged Qext, Qsca, g and F_ij ratios for BCCA N=16 cluster against MSTM."""
    from geometry.geometry import build_bcca
    
    r_nm = 25.0
    wl_nm = 550.0
    k = 2.0 * np.pi / wl_nm
    x_param = r_nm * k
    m_ri = 1.95 + 0.79j
    
    # 1. Setup BCCA N=16 cluster
    N = 16
    pos_raw = np.array([
        [-9.265991765213371, 53.070374956460924, 5.125464427486769],
        [-53.434101037671596, 69.8770907204946, 21.45679715091078],
        [36.394715815822536, 63.13878096258197, -12.58687544321387],
        [35.16652516036126, 20.10713177010107, 12.844772468179741],
        [42.21211005960575, -32.98939592469294, -14.462803979365475],
        [-0.8556655651591143, -12.092012479292364, -0.024449922520688006],
        [-29.102553823226543, -104.70790196915938, -1.1035037044779017],
        [-21.11503884451892, -56.40406803649388, -11.249400996999356],
        [-165.7386848635556, 9.980256665184356, 17.06099089250648],
        [-136.56631618469157, 48.77980173662161, 29.04298604832013],
        [-109.04831815470698, 17.83977709631006, 1.0166652109583936],
        [-96.86138243679056, -30.368869066395177, -4.218316447698413],
        [-122.61587673058068, -64.6547984835438, -32.98387510524762],
        [-72.78065229473009, -61.29734718780867, -35.25938081281046],
        [-29.73209731818929, -57.98766050549909, -60.47580182169449],
        [16.509392771201988, -69.49547062887552, -45.33331522017974]
    ])
    pos_d = pos_raw * k
    p = 3
    
    # Mie coefficients
    a, b = mie_coefficients_manual(m_ri, x_param, p)
    an_rcr = np.zeros((2, 2, p), dtype=np.complex128)
    diag = -0.5 * (a + b)
    off = -0.5 * (a - b)
    an_rcr[0, 0, :] = diag
    an_rcr[1, 1, :] = diag
    an_rcr[0, 1, :] = off
    an_rcr[1, 0, :] = off
    
    sys16 = MultisphereRCRSystem(pos_d, [p]*N, [an_rcr]*N, k_medium=1.0)
    agg_order = 6
    T_agg = sys16.calculate_aggregate_tmatrix(aggregate_order=agg_order, tol=1e-12)
    
    # 2. Compute averages
    avgs = sys16.calculate_orientation_averages(T_agg, [x_param]*N)
    
    # Check match with MSTM values (relative tolerance 1e-4)
    assert abs(avgs["Qext"] - 1.160511) / 1.160511 < 1e-4
    assert abs(avgs["Qsca"] - 0.188928) / 0.188928 < 1e-4
    assert abs(avgs["g"] - 0.26857102) < 1e-4
    
    # 3. Compute phase matrix at theta = 90 deg and 54 deg (matching MSTM values)
    theta_mstm = np.array([90.0, 54.0])
    sm_coef = sys16.calculate_orientation_averaged_phase_matrix_coefficients(T_agg)
    pm = sys16.calculate_phase_matrix_from_coefficients(sm_coef, theta_mstm, normalize=True)
    
    # Check 90 deg values
    F11_90 = pm[0, 0, 0]
    F12_F11_90 = pm[0, 0, 1] / pm[0, 0, 0]
    F22_F11_90 = pm[0, 1, 1] / pm[0, 0, 0]
    
    # MSTM reference values at 90 deg:
    # F11 = 0.170980, F12/F11 = -0.970820, F22/F11 = 0.973900
    assert abs(F11_90 - 0.170980) < 1e-4
    assert abs(F12_F11_90 - (-0.970820)) < 1e-4
    assert abs(F22_F11_90 - 0.973900) < 1e-4

    # Check 54 deg values
    F11_54 = pm[1, 0, 0]
    F12_F11_54 = pm[1, 0, 1] / pm[1, 0, 0]
    F22_F11_54 = pm[1, 1, 1] / pm[1, 0, 0]

    # MSTM reference values at 54 deg:
    # F11 = 0.347790, F12/F11 = -0.465180, F22/F11 = 0.982000
    assert abs(F11_54 - 0.347790) < 1e-4
    assert abs(F12_F11_54 - (-0.465180)) < 1e-4
    assert abs(F22_F11_54 - 0.982000) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
