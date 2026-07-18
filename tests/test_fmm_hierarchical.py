#!/usr/bin/env python3
"""Precision verification for VSWF FMM operators and hierarchical translations."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmm.fmm_operator import _rcr_translate, _ProviderCache


def test_vswf_addition_theorem_factorization():
    """Verify that direct singular translation from sphere j to sphere i
    is mathematically identical to the factored pipeline:
      direct = L2P (regular) * M2L (singular) * P2M (regular)
    This validates P2M, M2L, and L2P translation logic and regular coaxial translation.
    """
    cache = _ProviderCache()
    p_sphere = 3
    P_fmm = 12  # Higher translation order for FMM cell expansions to minimize truncation error
    
    # 1. Setup coordinates
    # Source sphere j near source cell center S
    r_j = np.array([0.1, -0.2, 0.15])
    r_S = np.array([0.0, 0.0, 0.0])
    
    # Target sphere i near target cell center T
    r_T = np.array([6.0, -2.0, 4.0])
    r_i = np.array([6.15, -1.8, 3.85])
    
    # Outgoing coefficients for sphere j (random complex values)
    rng = np.random.default_rng(12345)
    coeff_size_sphere = 2 * p_sphere * (p_sphere + 2)
    x_j = rng.standard_normal(coeff_size_sphere) + 1j * rng.standard_normal(coeff_size_sphere)
    
    # Pad x_j to P_fmm order for cell translations
    nblk_half_sphere = p_sphere * (p_sphere + 2)
    nblk_half_fmm = P_fmm * (P_fmm + 2)
    coeff_size_fmm = 2 * nblk_half_fmm
    
    x_j_padded = np.zeros(coeff_size_fmm, dtype=np.complex128)
    x_j_padded[:nblk_half_sphere] = x_j[:nblk_half_sphere]
    x_j_padded[nblk_half_fmm:nblk_half_fmm + nblk_half_sphere] = x_j[nblk_half_sphere:]
    
    # 2. Compute Direct Translation from j to i (singular translation, regular=False)
    tvec_direct = r_i - r_j
    y_direct_fmm = _rcr_translate(tvec_direct, x_j_padded, P_fmm, cache, regular=False)
    
    # Slice back to sphere order for comparison
    y_direct = np.zeros(coeff_size_sphere, dtype=np.complex128)
    y_direct[:nblk_half_sphere] = y_direct_fmm[:nblk_half_sphere]
    y_direct[nblk_half_sphere:] = y_direct_fmm[nblk_half_fmm:nblk_half_fmm + nblk_half_sphere]
    
    # 3. Compute Factored Pipeline
    # Step A: P2M (translate outgoing waves from j to S, regular=True)
    tvec_p2m = r_S - r_j
    x_S = _rcr_translate(tvec_p2m, x_j_padded, P_fmm, cache, regular=True)
    
    # Step B: M2L (translate outgoing from S to incoming regular at T, regular=False)
    tvec_m2l = r_T - r_S
    x_T = _rcr_translate(tvec_m2l, x_S, P_fmm, cache, regular=False)
    
    # Step C: L2P (translate regular from T to i, regular=True)
    tvec_l2p = r_i - r_T
    y_factored_fmm = _rcr_translate(tvec_l2p, x_T, P_fmm, cache, regular=True)
    
    # Slice back to sphere order for comparison
    y_factored = np.zeros(coeff_size_sphere, dtype=np.complex128)
    y_factored[:nblk_half_sphere] = y_factored_fmm[:nblk_half_sphere]
    y_factored[nblk_half_sphere:] = y_factored_fmm[nblk_half_fmm:nblk_half_fmm + nblk_half_sphere]
    
    # 4. Assert accuracy
    l2_error = np.linalg.norm(y_factored - y_direct) / np.linalg.norm(y_direct)
    print(f"\nFactored FMM Addition Theorem L2 Error: {l2_error:.4e}")
    
    assert l2_error < 1e-10, f"VSWF addition theorem factorization failed, L2 error={l2_error:.4e}"

from fmm.fmm_solver import FMMMultisphereSystem
from solver.bicgstab_solver import MultisphereRCRSystem

def _make_mie_coefficients(nodr: int) -> np.ndarray:
    """Build a simple diagonal Mie coefficient array for testing."""
    an = np.zeros((2, 2, nodr), dtype=np.complex128)
    for n in range(nodr):
        an[0, 0, n] = -0.5 + 0.1j * n  # TM diagonal
        an[1, 1, n] = -0.3 + 0.05j * n  # TE diagonal
    return an

def test_fmm_native_calculate_aggregate_tmatrix():
    """Verify FMM-native Multi-RHS aggregate T-matrix solver against direct solver.
    Also tests thread-safety and parallel column batching.
    """
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
    ])
    nodr_list = [3, 3, 3]
    an_list = [_make_mie_coefficients(3)] * 3
    radii_dimless = [1.0, 1.0, 1.0]
    agg_order = 4
    tol = 1e-12

    # 1. Direct solve
    direct = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    T_agg_direct = direct.calculate_aggregate_tmatrix(agg_order, tol=tol)
    avg_direct = direct.calculate_orientation_averages(T_agg_direct, radii_dimless)

    # 2. FMM solve (Serial)
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    fmm_serial = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=1, max_level=4, use_fmm=True,
        use_fmm_tmatrix=True
    )
    T_agg_fmm_serial = fmm_serial.calculate_aggregate_tmatrix(agg_order, tol=tol)
    avg_fmm_serial = fmm_serial.calculate_orientation_averages(T_agg_fmm_serial, radii_dimless)

    # 3. FMM solve (Parallel ThreadPoolExecutor)
    os.environ["OMP_NUM_THREADS"] = "4"
    fmm_parallel = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=1, max_level=4, use_fmm=True,
        use_fmm_tmatrix=True
    )
    T_agg_fmm_parallel = fmm_parallel.calculate_aggregate_tmatrix(agg_order, tol=tol)
    avg_fmm_parallel = fmm_parallel.calculate_orientation_averages(T_agg_fmm_parallel, radii_dimless)

    # Clean up environment variable
    if "OMP_NUM_THREADS" in os.environ:
        del os.environ["OMP_NUM_THREADS"]

    # 4. Compare Serial FMM vs Direct
    rel_err_serial = np.linalg.norm(T_agg_fmm_serial - T_agg_direct) / np.linalg.norm(T_agg_direct)
    print(f"\nSerial FMM vs Direct T-matrix L2 error: {rel_err_serial:.4e}")
    assert rel_err_serial < 1e-11, f"Serial FMM T-matrix mismatch, rel_err={rel_err_serial:.4e}"

    # 5. Compare Parallel FMM vs Direct
    rel_err_parallel = np.linalg.norm(T_agg_fmm_parallel - T_agg_direct) / np.linalg.norm(T_agg_direct)
    print(f"Parallel FMM vs Direct T-matrix L2 error: {rel_err_parallel:.4e}")
    assert rel_err_parallel < 1e-11, f"Parallel FMM T-matrix mismatch, rel_err={rel_err_parallel:.4e}"

    # 6. Compare averages
    for key in ["Qext", "Qsca", "Qabs", "g"]:
        diff_serial = abs(avg_fmm_serial[key] - avg_direct[key])
        diff_parallel = abs(avg_fmm_parallel[key] - avg_direct[key])
        print(f"Orientation average {key} - Direct: {avg_direct[key]:.6f}, FMM serial diff: {diff_serial:.4e}, FMM parallel diff: {diff_parallel:.4e}")
        assert diff_serial < 1e-11, f"Serial FMM average {key} mismatch, diff={diff_serial:.4e}"
        assert diff_parallel < 1e-11, f"Parallel FMM average {key} mismatch, diff={diff_parallel:.4e}"


def test_fmm_tmatrix_flag_ignored_when_fmm_disabled():
    """FMM T-matrix opt-in should not crash when the system itself uses direct mode."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ])
    nodr_list = [2, 2]
    an_list = [_make_mie_coefficients(2)] * 2

    direct = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    fallback = FMMMultisphereSystem(
        positions,
        nodr_list,
        an_list,
        k_medium,
        use_fmm=False,
        use_fmm_tmatrix=True,
    )

    T_direct = direct.calculate_aggregate_tmatrix(aggregate_order=2, tol=1e-12)
    T_fallback = fallback.calculate_aggregate_tmatrix(aggregate_order=2, tol=1e-12)

    rel_err = np.linalg.norm(T_fallback - T_direct) / np.linalg.norm(T_direct)
    assert rel_err < 1e-12


def test_coaxial_decay_zeroing_regression():
    """Verify that Clebsch-Gordan coefficients computed with early break and
    upward recurrence match a full downward-recurrence run (without early break)
    for all significant elements, and that any omitted/zeroed elements are negligible.
    """
    from rcr_engine.coaxial_translation import _vcfunc, _precompute_fnr, _precompute_bcof
    
    # Precompute tables for high-order case
    p_max = 50
    fnr = _precompute_fnr(p_max)
    bcof = _precompute_bcof(p_max, fnr)
    
    # We choose a high-order case that triggers the early break (downward to upward transition)
    def _vcfunc_no_break(m, n, k, l):
        wmax = n + l
        wmin = max(abs(m + k), abs(n - l))
        if wmin > wmax:
            return {}
        vcn = {}
        vcn[wmax] = bcof[n - m, l - k] * bcof[n + m, l + k] / bcof[n + n, l + l]
        if wmin == wmax:
            return {wmax: vcn[wmax]}
        
        denom_w1 = fnr[n + l + m + k] * fnr[n + l - m - k]
        if abs(denom_w1) < 1e-300 or fnr[l] < 1e-300 or fnr[n] < 1e-300:
            vcn[wmax - 1] = 0.0
        else:
            vcn[wmax - 1] = (vcn[wmax] * float(l * m - k * n)
                             * fnr[2 * (l + n) - 1]
                             / fnr[l] / fnr[n]
                             / denom_w1)
        if wmin == wmax - 1:
            return {w: vcn[w] for w in range(wmin, wmax + 1) if vcn[w] != 0.0}
            
        # Run downward recurrence all the way to wmin without breaking
        for w in range(wmax, wmin + 1, -1):
            i_wmk = w - (m + k)
            i_wpk = w + (m + k)
            i_nlw = n - l + w
            i_lnw = l - n + w
            i_nlw1 = n + l - w + 1
            i_nlw2 = n + l + w + 1
            if (i_wmk < 0 or i_wpk < 0 or i_nlw < 0 or i_lnw < 0 or
                    i_nlw1 < 0 or i_nlw2 < 0):
                vcn[w - 2] = 0.0
                continue
            denom_t1 = (fnr[i_wpk] * fnr[i_wmk] * fnr[i_nlw] * fnr[i_lnw]
                        * fnr[i_nlw1] * fnr[i_nlw2])
            if abs(denom_t1) < 1e-300:
                vcn[w - 2] = 0.0
                continue
            t1 = 2.0 * w * fnr[2 * w + 1] * fnr[2 * w - 1] / denom_t1
            t2 = 0.0 if w <= 1 else (float((m - k) * w * (w - 1) - (m + k) * n * (n + 1) + (m + k) * l * (l + 1)) / float(2 * w * (w - 1)))
            
            i_wmk1 = w - (m + k) - 1
            i_wpk1 = w + (m + k) - 1
            i_lnw1 = l - n + w - 1
            i_nlw1b = n - l + w - 1
            i_nlw2b = n + l - w + 2
            i_nlwb = n + l + w
            denom_t3 = float(2 * (w - 1)) * fnr[2 * w - 3] * fnr[2 * w - 1]
            if abs(denom_t3) < 1e-300:
                vcn[w - 2] = 0.0
                continue
            t3 = (fnr[i_wmk1] * fnr[i_wpk1] * fnr[i_lnw1] * fnr[i_nlw1b]
                  * fnr[i_nlw2b] * fnr[i_nlwb] / denom_t3)
            vcn[w - 2] = (t2 * vcn[w - 1] - vcn[w] / t1) / t3
            
        return {w: vcn[w] for w in range(wmin, wmax + 1) if vcn[w] != 0.0}

    # High order parameters
    m, n, k, l = 5, 30, 5, 30
    res_default = _vcfunc(m, n, k, l, bcof, fnr)
    res_no_break = _vcfunc_no_break(m, n, k, l)
    
    # Assert that all keys match or are negligible
    wmin = max(abs(m+k), abs(n-l))
    wmax = n + l
    for w in range(wmin, wmax + 1):
        val_def = res_default.get(w, 0.0)
        val_nb = res_no_break.get(w, 0.0)
        if abs(val_nb) > 1e-12:
            assert np.allclose(val_def, val_nb, rtol=1e-10, atol=1e-12), f"Mismatch at w={w}: def={val_def}, no_break={val_nb}"
        else:
            assert abs(val_def) < 1e-12, f"Omitted/zeroed element is not negligible at w={w}: def={val_def}, no_break={val_nb}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
