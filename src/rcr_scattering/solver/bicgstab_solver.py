#!/usr/bin/env python3
"""Matrix-Free Multi-Sphere Krylov Solver using RCR Translation Engine with Allocation-Free JIT-Parallelism."""

import time
import math
import os
import numpy as np
from numba import njit, prange
import numba
import concurrent.futures
from rcr_engine.basis_conventions import packed_mode_index
from rcr_engine.wigner_rotation import WignerRotationProvider, sh_index, FAC_TABLE, _rotate_to_z_jit, _rotate_from_z_jit, _rotate_to_z_cached_jit, _rotate_from_z_cached_jit, _build_cached_d_jit
from rcr_engine.coaxial_translation import CoaxialTranslationProvider, _mstm_normalizedlegendre_zero_jit, _apply_coaxial_translation_loop


def apply_tmatrix_rcr_basis(x: np.ndarray, an: np.ndarray, nodr: int) -> np.ndarray:
    """Apply a per-sphere 2x2 T-matrix in the transformed RCR basis.
    
    Args:
        x: Complex coefficient vector of shape (2 * nblk_half,) or (2 * nblk_half, cols).
           The first block stores Psi^(1), the second block stores Psi^(2).
        an: Complex per-order T-matrix array of shape (2, 2, nodr).
        nodr: Truncation order.
        
    Returns:
        Complex vector of same shape as x in the same transformed two-block basis.
    """
    nblk_half = nodr * (nodr + 2)
    y = np.zeros_like(x)
    
    is_2d = (x.ndim == 2)
    
    for n in range(1, nodr + 1):
        m_vals = np.arange(-n, n + 1)
        idxs = n * (n + 1) + m_vals - 1
        idxs_p2 = idxs + nblk_half
        
        if is_2d:
            cx_p1 = x[idxs, :]
            cx_p2 = x[idxs_p2, :]
            
            # Apply 2x2 matrix
            y[idxs, :] = an[0, 0, n-1] * cx_p1 + an[0, 1, n-1] * cx_p2
            y[idxs_p2, :] = an[1, 0, n-1] * cx_p1 + an[1, 1, n-1] * cx_p2
        else:
            cx_p1 = x[idxs]
            cx_p2 = x[idxs_p2]
            
            # Apply 2x2 matrix
            y[idxs] = an[0, 0, n-1] * cx_p1 + an[0, 1, n-1] * cx_p2
            y[idxs_p2] = an[1, 0, n-1] * cx_p1 + an[1, 1, n-1] * cx_p2
            
    return y


def apply_tmatrix_model1(x: np.ndarray, an: np.ndarray, nodr: int) -> np.ndarray:
    """Legacy compatibility wrapper for the transformed RCR two-block basis.

    Historical name retained for external callers. The implementation is the
    transformed Psi^(1)/Psi^(2) basis used throughout the project, not a direct
    TE/TM model-1 basis.
    """
    return apply_tmatrix_rcr_basis(x, an, nodr)


def bicgstab(A_op, b: np.ndarray, x0: np.ndarray = None, tol: float = 1e-12, max_iter: int = 2000) -> tuple[np.ndarray, float, int]:
    """Solve A x = b using BiCGSTAB algorithm."""
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()
        
    r = b - A_op(x)
    r_tilde = r.copy()
    
    rho = 1.0
    alpha = 1.0
    omega = 1.0
    
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-300:
        return np.zeros_like(b), 0.0, 0
        
    for i in range(max_iter):
        rho_new = np.dot(r_tilde.conj(), r)
        if abs(rho_new) < 1e-300:
            return x, np.linalg.norm(r) / norm_b, i+1
            
        if i == 0:
            p = r.copy()
        else:
            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
            
        rho = rho_new
        
        v = A_op(p)
        denom = np.dot(r_tilde.conj(), v)
        if abs(denom) < 1e-300:
            return x, np.linalg.norm(r) / norm_b, i+1
        alpha = rho / denom
        
        s = r - alpha * v
        if np.linalg.norm(s) / norm_b < tol:
            x = x + alpha * p
            return x, np.linalg.norm(s) / norm_b, i+1
            
        t = A_op(s)
        denom_t = np.dot(t.conj(), t)
        if abs(denom_t) < 1e-300:
            omega = 0.0
        else:
            omega = np.dot(t.conj(), s) / denom_t
            
        x = x + alpha * p + omega * s
        r = s - omega * t
        
        norm_r = np.linalg.norm(r)
        if norm_r / norm_b < tol:
            return x, norm_r / norm_b, i+1
            
    return x, np.linalg.norm(r) / norm_b, max_iter


@njit(parallel=True, fastmath=True, nogil=True)
def _apply_system_operator_parallel_jit(
    n: int,
    p: int,
    nblk_half: int,
    X: np.ndarray,             # shape (N, 2*nblk_half)
    pair_r: np.ndarray,        # shape (N, N)
    pair_phi: np.ndarray,      # shape (N, N)
    pair_d_blocks: np.ndarray, # shape (N, N, d_cache_size)
    an_array: np.ndarray,      # shape (N, 2, 2, p)
    fac_table: np.ndarray,
    fnr_table: np.ndarray,
    tran_coef_arr: np.ndarray, # shape (nblk_half+1, nblk_half+1, 2*p+1)
    dc: np.ndarray,            # precomputed, shape (2*p + 1)
) -> np.ndarray:
    """Core multi-sphere translation system operator compiled and parallelized via Numba prange.

    Buffers hn_i and incoming_i are allocated locally per prange iteration, which is
    thread-safe: Numba guarantees each parallel lane has its own stack/heap slice.
    This eliminates the race condition that occurred when pre-allocated per-sphere
    buffers were shared across concurrent outer Python threads in calculate_aggregate_tmatrix.
    """
    AX = X.copy()
    wmax_global = 2 * p

    # Parallel target sphere loop — each prange lane is isolated
    for i in prange(n):
        # Local allocation: thread-safe, Numba allocates these on the lane's private heap
        incoming_i = np.zeros(2 * nblk_half, dtype=np.complex128)
        hn_i = np.empty(wmax_global + 1, dtype=np.complex128)

        for j in range(n):
            if i == j:
                continue

            r   = pair_r[i, j]
            phi = pair_phi[i, j]

            # Apply RCR: forward rotation -> coaxial translation -> inverse rotation
            x_rot = _rotate_to_z_cached_jit(p, nblk_half, phi, X[j], pair_d_blocks[i, j])

            # Evaluate Hankel recurrence inline
            exp_phase = np.exp(1j * r)
            hn_i[0] = -1j * exp_phase / r
            hn_i[1] = (-1.0/r - 1j/(r**2)) * exp_phase
            for w in range(2, wmax_global + 1):
                hn_i[w] = ((2 * w - 1) / r) * hn_i[w - 1] - hn_i[w - 2]

            # Apply coaxial translation loop using precomputed dc
            y_rot = _apply_coaxial_translation_loop(p, nblk_half, x_rot, hn_i, dc, tran_coef_arr)

            # Inverse rotation
            incoming_i += _rotate_from_z_cached_jit(p, nblk_half, phi, y_rot, pair_d_blocks[i, j])

        # Apply local T-matrix of sphere i
        Z_i = np.zeros(2 * nblk_half, dtype=np.complex128)
        an_i = an_array[i]

        for deg in range(1, p + 1):
            idx_start = deg * deg - 1
            idx_end   = idx_start + 2 * deg + 1
            for offset_idx in range(idx_start, idx_end):
                cx_p1 = incoming_i[offset_idx]
                cx_p2 = incoming_i[offset_idx + nblk_half]
                Z_i[offset_idx]              = an_i[0, 0, deg-1] * cx_p1 + an_i[0, 1, deg-1] * cx_p2
                Z_i[offset_idx + nblk_half] = an_i[1, 0, deg-1] * cx_p1 + an_i[1, 1, deg-1] * cx_p2

        # AX_i = X_i - Z_i
        for k in range(2 * nblk_half):
            AX[i, k] -= Z_i[k]

    return AX


@njit(parallel=False, fastmath=True, nogil=True)
def _apply_system_operator_serial_jit(
    n: int,
    p: int,
    nblk_half: int,
    X: np.ndarray,             # shape (N, 2*nblk_half)
    pair_r: np.ndarray,        # shape (N, N)
    pair_phi: np.ndarray,      # shape (N, N)
    pair_d_blocks: np.ndarray, # shape (N, N, d_cache_size)
    an_array: np.ndarray,      # shape (N, 2, 2, p)
    fac_table: np.ndarray,
    fnr_table: np.ndarray,
    tran_coef_arr: np.ndarray, # shape (nblk_half+1, nblk_half+1, 2*p+1)
    dc: np.ndarray,            # precomputed, shape (2*p + 1)
) -> np.ndarray:
    """Fully serial (no prange) version of the system operator.

    Used in Mode A (outer Python ThreadPoolExecutor) where N_cpu Python threads
    each call this function concurrently.  Because there is no Numba threading
    layer involved (parallel=False) and all buffers are allocated locally, this
    is safe to call from multiple Python threads simultaneously.
    """
    AX = X.copy()
    wmax_global = 2 * p

    for i in range(n):          # serial — outer Python threads provide concurrency
        incoming_i = np.zeros(2 * nblk_half, dtype=np.complex128)
        hn_i = np.empty(wmax_global + 1, dtype=np.complex128)

        for j in range(n):
            if i == j:
                continue

            r   = pair_r[i, j]
            phi = pair_phi[i, j]

            x_rot = _rotate_to_z_cached_jit(p, nblk_half, phi, X[j], pair_d_blocks[i, j])

            exp_phase = np.exp(1j * r)
            hn_i[0] = -1j * exp_phase / r
            hn_i[1] = (-1.0/r - 1j/(r**2)) * exp_phase
            for w in range(2, wmax_global + 1):
                hn_i[w] = ((2 * w - 1) / r) * hn_i[w - 1] - hn_i[w - 2]

            y_rot = _apply_coaxial_translation_loop(p, nblk_half, x_rot, hn_i, dc, tran_coef_arr)
            incoming_i += _rotate_from_z_cached_jit(p, nblk_half, phi, y_rot, pair_d_blocks[i, j])

        Z_i = np.zeros(2 * nblk_half, dtype=np.complex128)
        an_i = an_array[i]

        for deg in range(1, p + 1):
            idx_start = deg * deg - 1
            idx_end   = idx_start + 2 * deg + 1
            for offset_idx in range(idx_start, idx_end):
                cx_p1 = incoming_i[offset_idx]
                cx_p2 = incoming_i[offset_idx + nblk_half]
                Z_i[offset_idx]              = an_i[0, 0, deg-1] * cx_p1 + an_i[0, 1, deg-1] * cx_p2
                Z_i[offset_idx + nblk_half] = an_i[1, 0, deg-1] * cx_p1 + an_i[1, 1, deg-1] * cx_p2

        for k in range(2 * nblk_half):
            AX[i, k] -= Z_i[k]

    return AX


@njit(fastmath=True, nogil=True)
def _dot_complex_jit(a: np.ndarray, b: np.ndarray) -> complex:
    val = 0.0 + 0.0j
    for i in range(len(a)):
        val += (a[i].real - 1j * a[i].imag) * b[i]
    return val


@njit(fastmath=True, nogil=True)
def _norm_complex_jit(a: np.ndarray) -> float:
    val = 0.0
    for i in range(len(a)):
        val += a[i].real**2 + a[i].imag**2
    return math.sqrt(val)


@njit(fastmath=True, nogil=True)
def _bicgstab_serial_jit(
    b: np.ndarray,
    n: int,
    p: int,
    nblk_half: int,
    pair_r: np.ndarray,
    pair_phi: np.ndarray,
    pair_d_blocks: np.ndarray,
    an_array: np.ndarray,
    fac_table: np.ndarray,
    fnr_table: np.ndarray,
    tran_coef_arr: np.ndarray,
    dc: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 2000
) -> tuple[np.ndarray, float]:
    X = np.zeros_like(b)
    X_2d = X.reshape((n, 2 * nblk_half))
    b_2d = b.reshape((n, 2 * nblk_half))
    
    AX = _apply_system_operator_serial_jit(
        n, p, nblk_half, X_2d, pair_r, pair_phi, pair_d_blocks,
        an_array, fac_table, fnr_table, tran_coef_arr, dc
    )
    r = b_2d - AX
    r_tilde = r.copy()
    
    rho = 1.0 + 0.0j
    alpha = 1.0 + 0.0j
    omega = 1.0 + 0.0j
    
    v = np.zeros_like(r)
    p_vec = np.zeros_like(r)
    
    b_flat = b.ravel()
    r_flat = r.ravel()
    r_tilde_flat = r_tilde.ravel()
    v_flat = v.ravel()
    p_flat = p_vec.ravel()
    
    norm_b = _norm_complex_jit(b_flat)
    if norm_b < 1e-300:
        return X.ravel(), 0.0
        
    for i in range(max_iter):
        rho_new = _dot_complex_jit(r_tilde_flat, r_flat)
        if abs(rho_new) < 1e-300:
            return X.ravel(), _norm_complex_jit(r_flat) / norm_b
            
        if i == 0:
            p_vec = r.copy()
            p_flat = p_vec.ravel()
        else:
            beta = (rho_new / rho) * (alpha / omega)
            p_vec = r + beta * (p_vec - omega * v)
            p_flat = p_vec.ravel()
            
        rho = rho_new
        
        v = _apply_system_operator_serial_jit(
            n, p, nblk_half, p_vec, pair_r, pair_phi, pair_d_blocks,
            an_array, fac_table, fnr_table, tran_coef_arr, dc
        )
        v_flat = v.ravel()
        
        denom = _dot_complex_jit(r_tilde_flat, v_flat)
        if abs(denom) < 1e-300:
            return X.ravel(), _norm_complex_jit(r_flat) / norm_b
        alpha = rho / denom
        
        s = r - alpha * v
        s_flat = s.ravel()
        if _norm_complex_jit(s_flat) / norm_b < tol:
            X_2d += alpha * p_vec
            return X.ravel(), _norm_complex_jit(s_flat) / norm_b
            
        t = _apply_system_operator_serial_jit(
            n, p, nblk_half, s, pair_r, pair_phi, pair_d_blocks,
            an_array, fac_table, fnr_table, tran_coef_arr, dc
        )
        t_flat = t.ravel()
        
        denom_t = _dot_complex_jit(t_flat, t_flat).real
        if abs(denom_t) < 1e-300:
            omega = 0.0 + 0.0j
        else:
            omega = _dot_complex_jit(t_flat, s_flat) / denom_t
            
        X_2d += alpha * p_vec + omega * s
        r = s - omega * t
        r_flat = r.ravel()
        
        norm_r = _norm_complex_jit(r_flat)
        if norm_r / norm_b < tol:
            return X.ravel(), norm_r / norm_b
            
    return X.ravel(), _norm_complex_jit(r_flat) / norm_b


@njit(fastmath=True)
def _build_rhs_single_col_jit(
    nsph: int,
    p_sphere: int,
    p_agg: int,
    b_vec: np.ndarray,
    sphere_jn_vals: np.ndarray,
    pair_phi: np.ndarray,
    positions_d_blocks: np.ndarray,
    k: int, l: int, q: int,
    an_array: np.ndarray,
    offsets: np.ndarray,
    nblk_full_list: np.ndarray,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
):
    nblk_half_agg = p_agg * (p_agg + 2)
    nblk_half_sph = p_sphere * (p_sphere + 2)
    
    common_rhs = np.zeros(2 * nblk_half_agg, dtype=np.complex128)
    idx_sh = l * (l + 1) + k - 1
    common_rhs[idx_sh] = 0.5
    common_rhs[idx_sh + nblk_half_agg] = -0.5 * ((-1) ** q)
    
    for i in range(nsph):
        phi = pair_phi[i]
        d_blocks_i = positions_d_blocks[i]
        an_i = an_array[i]
        off_i = offsets[i]
        
        # 1. Rotate to z-axis
        x_rot = _rotate_to_z_cached_jit(p_agg, nblk_half_agg, phi, common_rhs, d_blocks_i)
        
        # 2. Coaxial translation
        y_rot = _apply_coaxial_translation_loop(p_agg, nblk_half_agg, x_rot, sphere_jn_vals[i], dc, tran_coef_arr)
        
        # 3. Rotate back
        y_rcr = _rotate_from_z_cached_jit(p_agg, nblk_half_agg, phi, y_rot, d_blocks_i)
        
        # 4. Slice and apply T-matrix
        Z_i = np.zeros(2 * nblk_half_sph, dtype=np.complex128)
        for deg in range(1, p_sphere + 1):
            idx_start = deg * deg - 1
            idx_end = idx_start + 2 * deg + 1
            for offset_idx in range(idx_start, idx_end):
                cx_p1 = y_rcr[offset_idx]
                cx_p2 = y_rcr[offset_idx + nblk_half_agg]
                Z_i[offset_idx]                 = an_i[0, 0, deg-1] * cx_p1 + an_i[0, 1, deg-1] * cx_p2
                Z_i[offset_idx + nblk_half_sph] = an_i[1, 0, deg-1] * cx_p1 + an_i[1, 1, deg-1] * cx_p2
                
        # Write to b_vec
        for k_full in range(2 * nblk_half_sph):
            b_vec[off_i + k_full] = Z_i[k_full]


@njit(fastmath=True)
def _build_rhs_single_col_matrix_write_jit(
    nsph: int,
    p_sphere: int,
    p_agg: int,
    b_matrix: np.ndarray,
    idx: int,
    sphere_jn_vals: np.ndarray,
    pair_phi: np.ndarray,
    positions_d_blocks: np.ndarray,
    k: int, l: int, q: int,
    an_array: np.ndarray,
    offsets: np.ndarray,
    nblk_full_list: np.ndarray,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
):
    nblk_half_agg = p_agg * (p_agg + 2)
    nblk_half_sph = p_sphere * (p_sphere + 2)
    
    common_rhs = np.zeros(2 * nblk_half_agg, dtype=np.complex128)
    idx_sh = l * (l + 1) + k - 1
    common_rhs[idx_sh] = 0.5
    common_rhs[idx_sh + nblk_half_agg] = -0.5 * ((-1) ** q)
    
    for i in range(nsph):
        phi = pair_phi[i]
        d_blocks_i = positions_d_blocks[i]
        an_i = an_array[i]
        off_i = offsets[i]
        
        # 1. Rotate to z-axis
        x_rot = _rotate_to_z_cached_jit(p_agg, nblk_half_agg, phi, common_rhs, d_blocks_i)
        
        # 2. Coaxial translation
        y_rot = _apply_coaxial_translation_loop(p_agg, nblk_half_agg, x_rot, sphere_jn_vals[i], dc, tran_coef_arr)
        
        # 3. Rotate back
        y_rcr = _rotate_from_z_cached_jit(p_agg, nblk_half_agg, phi, y_rot, d_blocks_i)
        
        # 4. Slice and apply T-matrix
        Z_i = np.zeros(2 * nblk_half_sph, dtype=np.complex128)
        for deg in range(1, p_sphere + 1):
            idx_start = deg * deg - 1
            idx_end = idx_start + 2 * deg + 1
            for offset_idx in range(idx_start, idx_end):
                cx_p1 = y_rcr[offset_idx]
                cx_p2 = y_rcr[offset_idx + nblk_half_agg]
                Z_i[offset_idx]                 = an_i[0, 0, deg-1] * cx_p1 + an_i[0, 1, deg-1] * cx_p2
                Z_i[offset_idx + nblk_half_sph] = an_i[1, 0, deg-1] * cx_p1 + an_i[1, 1, deg-1] * cx_p2
                
        # Write directly to b_matrix column
        for k_full in range(2 * nblk_half_sph):
            b_matrix[off_i + k_full, idx] = Z_i[k_full]


@njit(fastmath=True)
def _merge_results_single_col_jit(
    nsph: int,
    p_sphere: int,
    p_agg: int,
    x_vec: np.ndarray,
    sphere_jn_vals: np.ndarray,
    pair_phi_neg: np.ndarray,
    positions_d_blocks_neg: np.ndarray,
    offsets: np.ndarray,
    nblk_full_list: np.ndarray,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
) -> np.ndarray:
    nblk_half_agg = p_agg * (p_agg + 2)
    nblk_half_sph = p_sphere * (p_sphere + 2)
    
    merged_lr = np.zeros(2 * nblk_half_agg, dtype=np.complex128)
    x_padded = np.zeros(2 * nblk_half_agg, dtype=np.complex128)
    
    for i in range(nsph):
        off_i = offsets[i]
        nb_i = nblk_full_list[i]
        
        # Pad monomer vector to aggregate size
        x_padded.fill(0.0)
        x_padded[:nblk_half_sph] = x_vec[off_i : off_i + nblk_half_sph]
        x_padded[nblk_half_agg : nblk_half_agg + nblk_half_sph] = x_vec[off_i + nblk_half_sph : off_i + nb_i]
        
        phi_neg = pair_phi_neg[i]
        d_blocks_neg_i = positions_d_blocks_neg[i]
        
        # 1. Rotate to z-axis
        x_rot = _rotate_to_z_cached_jit(p_agg, nblk_half_agg, phi_neg, x_padded, d_blocks_neg_i)
        
        # 2. Coaxial translation
        y_rot = _apply_coaxial_translation_loop(p_agg, nblk_half_agg, x_rot, sphere_jn_vals[i], dc, tran_coef_arr)
        
        # 3. Rotate back
        y_rcr = _rotate_from_z_cached_jit(p_agg, nblk_half_agg, phi_neg, y_rot, d_blocks_neg_i)
        
        # 4. Accumulate
        merged_lr += y_rcr
        
    lr1 = merged_lr[:nblk_half_agg]
    lr2 = merged_lr[nblk_half_agg:]
    
    em_vec = np.zeros(2 * nblk_half_agg, dtype=np.complex128)
    for deg in range(nblk_half_agg):
        em_vec[deg] = lr1[deg] + lr2[deg]
        em_vec[deg + nblk_half_agg] = lr1[deg] - lr2[deg]
        
    return em_vec


@njit(parallel=True, fastmath=True)
def _precompute_merge_operator_flat_jit(
    nsph: int,
    p_sphere: int,
    p_agg: int,
    sphere_jn_vals: np.ndarray,
    pair_phi_neg: np.ndarray,
    positions_d_blocks_neg: np.ndarray,
    offsets: np.ndarray,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
) -> np.ndarray:
    nblk_half_agg = p_agg * (p_agg + 2)
    nblk_half_sph = p_sphere * (p_sphere + 2)
    c_sphere = 2 * nblk_half_sph
    c_agg = 2 * nblk_half_agg
    
    # M_flat of shape (c_agg, nsph * c_sphere)
    M_flat = np.zeros((c_agg, nsph * c_sphere), dtype=np.complex128)
    
    for i in prange(nsph):
        phi_neg = pair_phi_neg[i]
        d_blocks_neg_i = positions_d_blocks_neg[i]
        sphere_jn_vals_i = sphere_jn_vals[i]
        off_i = offsets[i]
        
        x_padded = np.zeros(c_agg, dtype=np.complex128)
        
        for c in range(c_sphere):
            # Build basis vector e_c
            x_padded.fill(0.0)
            if c < nblk_half_sph:
                x_padded[c] = 1.0
            else:
                x_padded[nblk_half_agg + (c - nblk_half_sph)] = 1.0
                
            # 1. Rotate to z-axis
            x_rot = _rotate_to_z_cached_jit(p_agg, nblk_half_agg, phi_neg, x_padded, d_blocks_neg_i)
            
            # 2. Coaxial translation
            y_rot = _apply_coaxial_translation_loop(p_agg, nblk_half_agg, x_rot, sphere_jn_vals_i, dc, tran_coef_arr)
            
            # 3. Rotate back
            y_rcr = _rotate_from_z_cached_jit(p_agg, nblk_half_agg, phi_neg, y_rot, d_blocks_neg_i)
            
            # 4. Model conversion P
            lr1 = y_rcr[:nblk_half_agg]
            lr2 = y_rcr[nblk_half_agg:]
            
            col_idx = off_i + c
            M_flat[:nblk_half_agg, col_idx] = lr1 + lr2
            M_flat[nblk_half_agg:, col_idx] = lr1 - lr2
            
    return M_flat


@njit(parallel=True, fastmath=True, nogil=True)
def _build_rhs_matrix_jit(
    nsph: int,
    p_sphere: int,
    p_agg: int,
    b_matrix: np.ndarray,
    sphere_jn_vals: np.ndarray,
    pair_phi: np.ndarray,
    positions_d_blocks: np.ndarray,
    col_keys: np.ndarray,
    an_array: np.ndarray,
    offsets: np.ndarray,
    nblk_full_list: np.ndarray,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
):
    B = len(col_keys)
    for idx in prange(B):
        k = col_keys[idx, 0]
        l = col_keys[idx, 1]
        q = col_keys[idx, 2]
        _build_rhs_single_col_matrix_write_jit(
            nsph, p_sphere, p_agg, b_matrix, idx,
            sphere_jn_vals, pair_phi, positions_d_blocks,
            k, l, q, an_array, offsets, nblk_full_list,
            dc, tran_coef_arr
        )


@njit(fastmath=True)
def _merge_results_matrix_jit(
    nsph: int,
    p_sphere: int,
    p_agg: int,
    x_matrix: np.ndarray,
    T_agg: np.ndarray,
    sphere_jn_vals: np.ndarray,
    pair_phi_neg: np.ndarray,
    positions_d_blocks_neg: np.ndarray,
    col_keys: np.ndarray,
    offsets: np.ndarray,
    nblk_full_list: np.ndarray,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
):
    B = len(col_keys)
    for idx in range(B):
        k = col_keys[idx, 0]
        l = col_keys[idx, 1]
        q = col_keys[idx, 2]
        col_idx = packed_mode_index(p_agg, q, l, k)
        
        x_vec = x_matrix[:, idx]
        em_vec = _merge_results_single_col_jit(
            nsph, p_sphere, p_agg, x_vec,
            sphere_jn_vals, pair_phi_neg, positions_d_blocks_neg,
            offsets, nblk_full_list, dc, tran_coef_arr
        )
        T_agg[:, col_idx] = em_vec


class MultisphereRCRSystem:
    """Manages matrix-free multi-sphere system assembly and Krylov solving via RCR."""

    def __init__(self, positions: np.ndarray, nodr_list: list[int], an_list: list[np.ndarray], k_medium: float, backend: str = "auto"):
        """Initialize the multi-sphere RCR solver."""
        self.positions = np.asarray(positions, dtype=np.float64)
        self.nodr_list = list(nodr_list)
        self.an_list = list(an_list)
        self.k_medium = float(k_medium)
        self.nsph = len(self.positions)
        self.backend = backend
        self.profile_data = {}
        
        if self.backend == "direct_gpu_dense":
            p_val = self.nodr_list[0] if len(self.nodr_list) > 0 else 0
            nblk_half_val = p_val * (p_val + 2)
            c_val = 2 * nblk_half_val
            nc_val = self.nsph * c_val
            mem_required = nc_val * nc_val * 16

            # Check available CPU RAM first
            cpu_avail = None
            try:
                import psutil
                cpu_avail = psutil.virtual_memory().available
            except ImportError:
                pass

            if cpu_avail is not None and mem_required > cpu_avail:
                raise MemoryError(
                    f"Required dense matrix size {mem_required / (1024**3):.2f} GiB "
                    f"exceeds available CPU memory {cpu_avail / (1024**3):.2f} GiB."
                )

            # Check available GPU free VRAM
            free_vram = 0
            try:
                import cupy as cp
                if cp.is_available():
                    free_vram, total_vram = cp.cuda.Device().mem_info
            except Exception:
                pass

            if free_vram > 0 and mem_required > free_vram:
                raise MemoryError(
                    f"Required dense matrix size {mem_required / (1024**3):.2f} GiB "
                    f"exceeds available GPU free memory {free_vram / (1024**3):.2f} GiB."
                )

        # Precompute pair distances and angles
        self.pair_r = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        self.pair_theta = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        self.pair_phi = np.zeros((self.nsph, self.nsph), dtype=np.float64)
        
        for i in range(self.nsph):
            for j in range(self.nsph):
                if i != j:
                    r_vec = self.positions[i] - self.positions[j]
                    r_mag = np.linalg.norm(r_vec)
                    self.pair_r[i, j] = r_mag * self.k_medium
                    if r_mag > 1e-14:
                        self.pair_theta[i, j] = np.arccos(r_vec[2] / r_mag)
                        self.pair_phi[i, j] = np.arctan2(r_vec[1], r_vec[0])
                        
        # Precompute ALL pair D-blocks for ultra-fast MVM
        self.max_nodr = max(self.nodr_list) if len(self.nodr_list) > 0 else 0
        if self.nsph > 0 and self.max_nodr > 0:
            d_cache_size = sum((2*n + 1)**2 for n in range(1, self.max_nodr + 1))
            self.pair_d_blocks = np.zeros((self.nsph, self.nsph, d_cache_size), dtype=np.float64)
            for i in range(self.nsph):
                for j in range(self.nsph):
                    if i != j:
                        self.pair_d_blocks[i, j] = _build_cached_d_jit(self.max_nodr, self.pair_theta[i, j], FAC_TABLE)
        else:
            self.pair_d_blocks = np.empty((0, 0, 0), dtype=np.float64)
                        
        # Determine flat sizes per sphere
        self.nblk_half_list = [n * (n + 2) for n in self.nodr_list]
        self.nblk_full_list = [2 * nb for nb in self.nblk_half_list]
        
        # Offsets for concatenating vectors
        self.offsets = [0]
        for nb in self.nblk_full_list:
            self.offsets.append(self.offsets[-1] + nb)
        self.N_total = self.offsets[-1]
            
        # Initialize RCR providers for needed truncation orders
        self.rot_providers = {}
        self.trans_providers = {}
        
        # Precompute pair-wise translation coordinates and providers
        self.pair_data = {}
        for j in range(self.nsph):
            for i in range(self.nsph):
                if i == j:
                    continue
                tvec = self.positions[i] - self.positions[j]
                r = np.linalg.norm(tvec)
                theta = np.arccos(tvec[2] / r) if r > 1e-14 else 0.0
                phi = np.arctan2(tvec[1], tvec[0]) if r > 1e-14 else 0.0
                
                p = max(self.nodr_list[j], self.nodr_list[i])
                
                if p not in self.rot_providers:
                    self.rot_providers[p] = WignerRotationProvider(p)
                    self.trans_providers[p] = CoaxialTranslationProvider(p)
                    
                self.pair_data[(j, i)] = {
                    'r': r,
                    'theta': theta,
                    'phi': phi,
                    'p': p
                }
                
        # Check if we can use JIT parallel fast-path (homogenous truncation order p, N > 1)
        self.use_parallel_fastpath = (self.nsph > 1 and len(set(self.nodr_list)) == 1)
        if self.backend in ("direct_gpu", "direct_multi_gpu", "direct_gpu_dense") and not self.use_parallel_fastpath:
            raise ValueError("GPU direct solver backends only support homogeneous truncation orders.")

        if self.use_parallel_fastpath:
            self.p_uniform = self.nodr_list[0]
            self.nblk_half_uniform = self.nblk_half_list[0]

            self.an_array = np.array(self.an_list)  # shape (N, 2, 2, p)
            self.prov_uniform = self.trans_providers[self.p_uniform]

            # Precompute Legendre dc outside parallel loop
            self.wmax_global = 2 * self.p_uniform
            self.dc = _mstm_normalizedlegendre_zero_jit(1.0, self.wmax_global, self.prov_uniform.fnr)
            # NOTE: hn_buf / incoming_buf removed — now allocated locally inside the JIT
            # prange loop to eliminate race conditions when called from concurrent threads.

            # Initialize GPU direct solver backend if selected
            if self.backend == "direct_gpu":
                from fmm.gpu_direct import GPUDirectRCRSystem
                self._gpu_system = GPUDirectRCRSystem(
                    self.positions,
                    self.nodr_list,
                    self.an_array,
                    self.k_medium,
                    self.dc,
                    self.prov_uniform.tran_coef_arr
                )
            elif self.backend == "direct_gpu_dense":
                from fmm.gpu_dense import GPUDenseRCRSystem
                self._gpu_system = GPUDenseRCRSystem(
                    self.positions,
                    self.nodr_list,
                    self.an_array,
                    self.k_medium,
                    self.dc,
                    self.prov_uniform.tran_coef_arr,
                    self.prov_uniform.fnr
                )
            elif self.backend == "direct_multi_gpu":
                import cupy as cp
                from fmm.gpu_direct import GPUDirectRCRSystem
                num_gpus = cp.cuda.runtime.getDeviceCount() if cp.is_available() else 1
                self._gpu_systems = []
                for dev_id in range(num_gpus):
                    with cp.cuda.Device(dev_id):
                        gpu_sys = GPUDirectRCRSystem(
                            self.positions,
                            self.nodr_list,
                            self.an_array,
                            self.k_medium,
                            self.dc,
                            self.prov_uniform.tran_coef_arr
                        )
                        self._gpu_systems.append(gpu_sys)

    def translate_rcr(self, j: int, i: int, x_j: np.ndarray) -> np.ndarray:
        """Translate coefficient vector x_j of sphere j to the coordinate frame of sphere i using RCR."""
        p_data = self.pair_data[(j, i)]
        p = p_data['p']
        r = p_data['r']
        theta = p_data['theta']
        phi = p_data['phi']
        
        rot_prov = self.rot_providers[p]
        trans_prov = self.trans_providers[p]
        
        nblk_half_j = self.nblk_half_list[j]
        nblk_half_p = p * (p + 2)
        
        # Pad x_j to max order p
        x_p = np.zeros(2 * nblk_half_p, dtype=np.complex128)
        x_p[:nblk_half_j] = x_j[:nblk_half_j]  # polarization 1
        x_p[nblk_half_p : nblk_half_p + nblk_half_j] = x_j[nblk_half_j:]  # polarization 2
        
        # Apply RCR: forward rotation -> coaxial translation -> inverse rotation
        x_rot = rot_prov.rotate_to_z(theta, phi, x_p)
        y_rot = trans_prov.apply_coaxial_translation(r, x_rot)
        y_rcr = rot_prov.rotate_from_z(theta, phi, y_rot)
        
        # Slice back to target order nodr_i
        nblk_half_i = self.nblk_half_list[i]
        y_i = np.zeros(2 * nblk_half_i, dtype=np.complex128)
        y_i[:nblk_half_i] = y_rcr[:nblk_half_i]  # polarization 1
        y_i[nblk_half_i:] = y_rcr[nblk_half_p : nblk_half_p + nblk_half_i]  # polarization 2
        
        return y_i

    def apply_system_operator(self, X) -> np.ndarray:
        """Apply system operator (I - T * M) to the concatenated vector X of all spheres."""
        if self.backend in ("direct_gpu", "direct_gpu_dense"):
            return self._gpu_system.apply_system_operator(X)
        elif self.backend == "direct_multi_gpu":
            import cupy as cp
            import concurrent.futures
            is_numpy = isinstance(X, np.ndarray)
            if is_numpy:
                X_np = X
            else:
                X_np = X.get() if hasattr(X, 'get') else np.asarray(X)
            
            if X_np.ndim == 1 or X_np.shape[1] == 1:
                with cp.cuda.Device(0):
                    res_gpu = self._gpu_systems[0].apply_system_operator(X_np)
                    return res_gpu.get() if hasattr(res_gpu, 'get') else res_gpu
            
            B = X_np.shape[1]
            num_gpus = len(self._gpu_systems)
            chunks = np.array_split(np.arange(B), num_gpus)
            
            results = [None] * num_gpus
            
            def _worker(dev_idx):
                col_idxs = chunks[dev_idx]
                if len(col_idxs) == 0:
                    return None
                with cp.cuda.Device(dev_idx):
                    X_chunk = X_np[:, col_idxs]
                    res_chunk = self._gpu_systems[dev_idx].apply_system_operator(X_chunk)
                    return res_chunk.get() if hasattr(res_chunk, 'get') else res_chunk

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [executor.submit(_worker, i) for i in range(num_gpus)]
                for i, f in enumerate(futures):
                    results[i] = f.result()
                    
            non_empty_results = [r for r in results if r is not None]
            final_res = np.concatenate(non_empty_results, axis=1)
            return final_res

        X = np.asarray(X, dtype=np.complex128)
        
        if self.use_parallel_fastpath:
            # Reshape X from flat 1D to 2D of shape (N, 2*nblk_half)
            X_2d = X.reshape((self.nsph, 2 * self.nblk_half_uniform))

            # Dispatch to the appropriate JIT kernel:
            # - serial kernel: safe to call from multiple Python threads (no Numba threading layer)
            # - parallel kernel: uses Numba prange, not safe for concurrent Python threads
            use_serial = False
            if getattr(self, '_use_serial_operator', False):
                use_serial = True
            elif self.backend == "direct_cpu_serial":
                use_serial = True
            elif self.backend == "direct_cpu_parallel":
                use_serial = False
            else: # auto or other
                use_serial = False

            if use_serial:
                AX = _apply_system_operator_serial_jit(
                    self.nsph,
                    self.p_uniform,
                    self.nblk_half_uniform,
                    X_2d,
                    self.pair_r,
                    self.pair_phi,
                    self.pair_d_blocks,
                    self.an_array,
                    FAC_TABLE,
                    self.prov_uniform.fnr,
                    self.prov_uniform.tran_coef_arr,
                    self.dc,
                )
            else:
                AX = _apply_system_operator_parallel_jit(
                    self.nsph,
                    self.p_uniform,
                    self.nblk_half_uniform,
                    X_2d,
                    self.pair_r,
                    self.pair_phi,
                    self.pair_d_blocks,
                    self.an_array,
                    FAC_TABLE,
                    self.prov_uniform.fnr,
                    self.prov_uniform.tran_coef_arr,
                    self.dc,
                )

            # Flatten back to 1D
            return AX.ravel()
            
        # Fallback to slow Python loop if heterogeneous p
        AX = X.copy()
        
        # Extract per-sphere blocks from X
        X_blocks = []
        for j in range(self.nsph):
            off = self.offsets[j]
            nb = self.nblk_full_list[j]
            X_blocks.append(X[off : off + nb])
            
        # Accumulate translated fields to each target sphere i
        for i in range(self.nsph):
            off_i = self.offsets[i]
            nb_i = self.nblk_full_list[i]
            nodr_i = self.nodr_list[i]
            an_i = self.an_list[i]
            
            # Sum over all source spheres j != i
            incoming_i = np.zeros(nb_i, dtype=np.complex128)
            for j in range(self.nsph):
                if i == j:
                    continue
                incoming_i += self.translate_rcr(j, i, X_blocks[j])
                
            # Apply T-matrix of sphere i: Z_i = T_i * incoming_i
            Z_i = apply_tmatrix_model1(incoming_i, an_i, nodr_i)
            
            # A_i = X_i - Z_i
            AX[off_i : off_i + nb_i] -= Z_i
            
        return AX

    def solve(self, b_vec, tol: float = 1e-12, max_iter: int = 2000) -> tuple[np.ndarray, float]:
        """Solve the linear system (I - T * M) X = b_vec using Matrix-Free BiCGSTAB."""
        if self.backend in ("direct_gpu", "direct_gpu_dense"):
            return self._gpu_system.solve(b_vec, tol=tol, max_iter=max_iter)
        elif self.backend == "direct_multi_gpu":
            import cupy as cp
            import concurrent.futures
            is_numpy = isinstance(b_vec, np.ndarray)
            if is_numpy:
                b_np = b_vec
            else:
                b_np = b_vec.get() if hasattr(b_vec, 'get') else np.asarray(b_vec)
            
            if b_np.ndim == 1 or b_np.shape[1] == 1:
                with cp.cuda.Device(0):
                    res_x, res_norm = self._gpu_systems[0].solve(b_np, tol=tol, max_iter=max_iter)
                    return (res_x.get() if hasattr(res_x, 'get') else res_x), res_norm
            
            B = b_np.shape[1]
            num_gpus = len(self._gpu_systems)
            chunks = np.array_split(np.arange(B), num_gpus)
            
            x_results = [None] * num_gpus
            res_norms = [0.0] * num_gpus
            
            def _worker(dev_idx):
                col_idxs = chunks[dev_idx]
                if len(col_idxs) == 0:
                    return None, 0.0
                with cp.cuda.Device(dev_idx):
                    b_chunk = b_np[:, col_idxs]
                    x_chunk, r_norm = self._gpu_systems[dev_idx].solve(b_chunk, tol=tol, max_iter=max_iter)
                    return (x_chunk.get() if hasattr(x_chunk, 'get') else x_chunk), r_norm

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [executor.submit(_worker, i) for i in range(num_gpus)]
                for i, f in enumerate(futures):
                    x_results[i], res_norms[i] = f.result()
                    
            non_empty_x = [r for r in x_results if r is not None]
            final_x = np.concatenate(non_empty_x, axis=1)
            final_res = max(res_norms)
            return final_x, final_res

        x, res, _iters = bicgstab(self.apply_system_operator, b_vec, tol=tol, max_iter=max_iter)
        return x, res

    def translate_rcr_general(self, tvec: np.ndarray, x_j: np.ndarray, source_order: int, target_order: int, regular: bool = False) -> np.ndarray:
        """Translate coefficient vector x_j over arbitrary displacement tvec using RCR."""
        tvec = np.asarray(tvec, dtype=np.float64)
        r = np.linalg.norm(tvec)
        nblk_half_src = source_order * (source_order + 2)
        nblk_half_tgt = target_order * (target_order + 2)
        
        if r < 1e-14:
            y = np.zeros(2 * nblk_half_tgt, dtype=np.complex128)
            take = min(nblk_half_src, nblk_half_tgt)
            y[:take] = x_j[:take]
            y[nblk_half_tgt : nblk_half_tgt + take] = x_j[nblk_half_src : nblk_half_src + take]
            return y
            
        theta = np.arccos(np.clip(tvec[2] / r, -1.0, 1.0))
        phi = np.arctan2(tvec[1], tvec[0])
        
        p = max(source_order, target_order)
        if p not in self.rot_providers:
            self.rot_providers[p] = WignerRotationProvider(p)
            self.trans_providers[p] = CoaxialTranslationProvider(p)
            
        rot_prov = self.rot_providers[p]
        trans_prov = self.trans_providers[p]
        
        nblk_half_p = p * (p + 2)
        
        # Pad x_j to max order p
        x_p = np.zeros(2 * nblk_half_p, dtype=np.complex128)
        x_p[:nblk_half_src] = x_j[:nblk_half_src]
        x_p[nblk_half_p : nblk_half_p + nblk_half_src] = x_j[nblk_half_src:]
        
        # Apply RCR: forward rotation -> coaxial translation (regular or outgoing) -> inverse rotation
        x_rot = rot_prov.rotate_to_z(theta, phi, x_p)
        y_rot = trans_prov.apply_coaxial_translation(r, x_rot, regular=regular)
        y_rcr = rot_prov.rotate_from_z(theta, phi, y_rot)
        
        # Slice back to target order
        y_tgt = np.zeros(2 * nblk_half_tgt, dtype=np.complex128)
        y_tgt[:nblk_half_tgt] = y_rcr[:nblk_half_tgt]
        y_tgt[nblk_half_tgt:] = y_rcr[nblk_half_p : nblk_half_p + nblk_half_tgt]
        
        return y_tgt

    @staticmethod
    def auto_aggregate_order(positions: np.ndarray, r_dimless: float, k_medium: float = 1.0) -> int:
        """Compute a conservative aggregate T-matrix truncation order following MSTM's criterion.

        MSTM selects nmax = ceil(x + 4.05 * x^(1/3) + 2) where x is the circumscribing
        sphere size parameter of the aggregate.  We use the cluster radius of gyration
        scaled by a factor of sqrt(5/3) to convert to the circumscribed sphere radius.

        Args:
            positions: (N, 3) sphere centres in the same units as k_medium^-1.
            r_dimless:  monomer radius in dimensionless units (k * a).
            k_medium:   medium wavenumber (default 1.0 when positions already dimensionless).

        Returns:
            Recommended aggregate_order integer.
        """
        if len(positions) <= 1:
            return max(3, int(np.ceil(r_dimless + 4.05 * r_dimless**(1/3) + 2)))
        cm = positions.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((positions - cm)**2, axis=1))))
        # Circumscribed sphere radius ≈ max(|r_i - cm|) + r_monomer_dimless
        r_circ = float(np.max(np.linalg.norm(positions - cm, axis=1))) * k_medium + r_dimless
        x_agg = r_circ
        nmax = int(np.ceil(x_agg + 4.05 * x_agg**(1/3) + 2))
        return max(nmax, 3)

    def calculate_aggregate_tmatrix(self, aggregate_order: int, tol: float = 1e-12) -> np.ndarray:
        """Calculate the full aggregate T-matrix of the sphere cluster.

        Parallelism strategy (thread-safe):
          - For direct_gpu: solves all T-matrix columns simultaneously in a single batched BiCGSTAB call.
          - For CPU backends: column problems are solved sequentially or using outer Python threads.
        """
        p_sphere = self.nodr_list[0] if self.nodr_list else 0
        if aggregate_order < p_sphere:
            raise ValueError(
                f"aggregate_order ({aggregate_order}) must be greater than or equal to "
                f"monomer order p_sphere ({p_sphere})."
            )

        nblk_half_agg = aggregate_order * (aggregate_order + 2)
        dim_agg = 2 * nblk_half_agg
        T_agg = np.zeros((dim_agg, dim_agg), dtype=np.complex128)

        col_keys = []
        for l in range(1, aggregate_order + 1):
            for k in range(-l, l + 1):
                for q in (1, 2):
                    col_keys.append((k, l, q))

        # Pre-initialize providers so _process_column never mutates self in parallel
        p_max_needed = aggregate_order
        if self.nodr_list:
            p_max_needed = max(p_max_needed, max(self.nodr_list))
        for p in range(1, p_max_needed + 1):
            if p not in self.rot_providers:
                self.rot_providers[p] = WignerRotationProvider(p)
                self.trans_providers[p] = CoaxialTranslationProvider(p)

        # Precompute polar coordinates and spherical Bessel values for each sphere
        from scipy.special import spherical_jn
        
        # d_cache_size for aggregate_order
        d_cache_size = sum((2*n + 1)**2 for n in range(1, aggregate_order + 1))
        
        sphere_jn_vals = np.zeros((self.nsph, 2 * aggregate_order + 1), dtype=np.complex128)
        pair_phi = np.zeros(self.nsph, dtype=np.float64)
        positions_d_blocks = np.zeros((self.nsph, d_cache_size), dtype=np.float64)
        
        pair_phi_neg = np.zeros(self.nsph, dtype=np.float64)
        positions_d_blocks_neg = np.zeros((self.nsph, d_cache_size), dtype=np.float64)
        
        for i in range(self.nsph):
            pos = self.positions[i]
            r_val = float(np.linalg.norm(pos)) * self.k_medium
            for w in range(2 * aggregate_order + 1):
                sphere_jn_vals[i, w] = spherical_jn(w, r_val)
                
            r_geom = float(np.linalg.norm(pos))
            if r_geom > 1e-14:
                theta = float(np.arccos(np.clip(pos[2] / r_geom, -1.0, 1.0)))
                phi = float(np.arctan2(pos[1], pos[0]))
            else:
                theta = 0.0
                phi = 0.0
            pair_phi[i] = phi
            positions_d_blocks[i] = _build_cached_d_jit(aggregate_order, theta, FAC_TABLE)
            
            # For -pos_i
            pos_neg = -pos
            if r_geom > 1e-14:
                theta_neg = float(np.arccos(np.clip(pos_neg[2] / r_geom, -1.0, 1.0)))
                phi_neg = float(np.arctan2(pos_neg[1], pos_neg[0]))
            else:
                theta_neg = 0.0
                phi_neg = 0.0
            pair_phi_neg[i] = phi_neg
            positions_d_blocks_neg[i] = _build_cached_d_jit(aggregate_order, theta_neg, FAC_TABLE)

        dim_system = sum(2 * nodr_i * (nodr_i + 2) for nodr_i in self.nodr_list)
        offsets_arr = np.array(self.offsets, dtype=np.int64)
        nblk_full_arr = np.array(self.nblk_full_list, dtype=np.int64)
        an_array = np.array(self.an_list, dtype=np.complex128)
        col_keys_arr = np.array(col_keys, dtype=np.int32)
        
        trans_prov = self.trans_providers[aggregate_order]
        tran_coef_arr = trans_prov.tran_coef_arr
        dc = _mstm_normalizedlegendre_zero_jit(1.0, 2 * aggregate_order, trans_prov.fnr)

        # GPU Batched Solve path
        if self.backend in ("direct_gpu", "direct_multi_gpu", "direct_gpu_dense"):
            B = len(col_keys)
            b_matrix = np.zeros((dim_system, B), dtype=np.complex128)
            
            # Build RHS for all columns using parallel JIT helper
            t0_rhs = time.perf_counter()
            _build_rhs_matrix_jit(
                self.nsph, p_sphere, aggregate_order, b_matrix,
                sphere_jn_vals, pair_phi, positions_d_blocks, col_keys_arr,
                an_array, offsets_arr, nblk_full_arr, dc, tran_coef_arr
            )
            self.profile_data['rhs_time'] = time.perf_counter() - t0_rhs

            # Solve all columns simultaneously on the GPU
            t0_solve = time.perf_counter()
            x_matrix, _res = self.solve(b_matrix, tol=tol, max_iter=2000)
            self.profile_data['solve_time'] = time.perf_counter() - t0_solve

            # Merge and transform for all columns using precomputed M_flat and GEMM
            t0_merge = time.perf_counter()
            
            # Precompute/cache M_flat if not present or aggregate_order changed
            if not hasattr(self, 'M_flat') or getattr(self, 'M_flat_agg_order', 0) != aggregate_order:
                t0_pre = time.perf_counter()
                self.M_flat = _precompute_merge_operator_flat_jit(
                    self.nsph, p_sphere, aggregate_order,
                    sphere_jn_vals, pair_phi_neg, positions_d_blocks_neg,
                    offsets_arr, dc, tran_coef_arr
                )
                self.M_flat_agg_order = aggregate_order
                self.profile_data['precompute_merge_time'] = time.perf_counter() - t0_pre
            else:
                self.profile_data['precompute_merge_time'] = 0.0
                
            # Map column keys to indices in T_agg
            col_indices = np.zeros(B, dtype=np.int64)
            nblk_half_agg = aggregate_order * (aggregate_order + 2)
            for idx in range(B):
                k = col_keys_arr[idx, 0]
                l = col_keys_arr[idx, 1]
                q = col_keys_arr[idx, 2]
                col_indices[idx] = packed_mode_index(aggregate_order, q, l, k)
                
            # Perform single highly optimized CPU GEMM
            t0_gemm = time.perf_counter()
            T_agg_flat = self.M_flat @ x_matrix
            T_agg[:, col_indices] = T_agg_flat
            self.profile_data['merge_gemm_time'] = time.perf_counter() - t0_gemm
            
            self.profile_data['merge_time'] = time.perf_counter() - t0_merge

            # Gather GPU sub-system profile data
            if self.backend in ("direct_gpu", "direct_gpu_dense") and hasattr(self, '_gpu_system'):
                if hasattr(self._gpu_system, 'profile_data'):
                    self.profile_data.update(self._gpu_system.profile_data)
            elif self.backend == "direct_multi_gpu" and hasattr(self, '_gpu_systems'):
                # Combine iters from device 0 as a representative
                if len(self._gpu_systems) > 0 and hasattr(self._gpu_systems[0], 'profile_data'):
                    self.profile_data.update(self._gpu_systems[0].profile_data)

            return T_agg

        # CPU / Fallback path
        def _process_column(key):
            k, l, q = key
            col_idx = packed_mode_index(aggregate_order, q, l, k)

            # 1. Build RHS vector for this column using JIT helper
            b_vec = np.zeros(dim_system, dtype=np.complex128)
            _build_rhs_single_col_jit(
                self.nsph, p_sphere, aggregate_order, b_vec,
                sphere_jn_vals, pair_phi, positions_d_blocks,
                k, l, q, an_array, offsets_arr, nblk_full_arr,
                dc, tran_coef_arr
            )

            # 2. Solve the coupled linear system
            if self.use_parallel_fastpath:
                x_m1, _res = _bicgstab_serial_jit(
                    b_vec, self.nsph, p_sphere, self.nblk_half_uniform,
                    self.pair_r, self.pair_phi, self.pair_d_blocks,
                    self.an_array, FAC_TABLE, self.prov_uniform.fnr,
                    self.prov_uniform.tran_coef_arr, self.dc,
                    tol=tol, max_iter=2000
                )
            else:
                x_m1, _res = self.solve(b_vec, tol=tol, max_iter=2000)

            # 3. Merge per-sphere scattered fields using JIT helper
            em_vec = _merge_results_single_col_jit(
                self.nsph, p_sphere, aggregate_order, x_m1,
                sphere_jn_vals, pair_phi_neg, positions_d_blocks_neg,
                offsets_arr, nblk_full_arr, dc, tran_coef_arr
            )
            return col_idx, em_vec

        # ── Parallel dispatch ──────────────────────────────────────────────────
        num_outer_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

        if num_outer_threads > 1:
            # Mode A: outer Python threads × serial inner JIT.
            self._use_serial_operator = True
            try:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_outer_threads
                ) as executor:
                    futures = {
                        executor.submit(_process_column, key): key
                        for key in col_keys
                    }
                    for future in concurrent.futures.as_completed(futures):
                        col_idx, em_vec = future.result()
                        T_agg[:, col_idx] = em_vec
            finally:
                self._use_serial_operator = False
        else:
            # Mode B: inner Numba prange only.
            for key in col_keys:
                col_idx, em_vec = _process_column(key)
                T_agg[:, col_idx] = em_vec

        return T_agg

    def calculate_orientation_averages(self, T_agg: np.ndarray, radii_dimless: list[float]) -> dict[str, float]:
        """Compute analytical orientation averages Qext, Qsca, Qabs and asymmetry parameter g from aggregate T-matrix."""
        t0_ori = time.perf_counter()
        xgeff = sum(r**3 for r in radii_dimless) ** (1.0/3.0)
        
        # Extinction trace sum
        trace_sum = np.trace(T_agg).real
        
        # Scattering Frobenius norm squared
        fro_sq = np.vdot(T_agg, T_agg).real
        
        normalizer = 2.0 / (xgeff ** 2)
        
        Qext = -normalizer * trace_sum
        Qsca = normalizer * fro_sq
        Qabs = Qext - Qsca

        # Compute analytical orientation-averaged asymmetry parameter g
        agg_order = int(round(math.sqrt(T_agg.shape[0] // 2 + 1) - 1.0))
        nbc = 6 * agg_order + 6
        fnr_t, bcof_t = precompute_fnr_bcof(nbc)
        
        # Precompute or retrieve cached Clebsch-Gordan caches
        if not hasattr(self, 'vc_cache_A') or getattr(self, 'vc_cache_agg_order', 0) != agg_order:
            self.vc_cache_A, self.vc_cache_B = _precompute_vc_caches_jit(agg_order, fnr_t, bcof_t)
            self.vc_cache_agg_order = agg_order
            
        sm_coef = calculate_orientation_averaged_phase_matrix_coefficients_jit(
            T_agg, agg_order, 0.0, fnr_t, bcof_t, self.vc_cache_A, self.vc_cache_B
        )
        
        g = 0.0
        if abs(sm_coef[0, 0, 0]) > 1e-300:
            g = float(sm_coef[1, 0, 0] / (3.0 * sm_coef[0, 0, 0]))
        
        self.profile_data['orientation_time'] = time.perf_counter() - t0_ori
        return {
            "Qext": float(Qext),
            "Qsca": float(Qsca),
            "Qabs": float(Qabs),
            "g": float(g),
            "xgeff": float(xgeff)
        }

    def calculate_orientation_averaged_phase_matrix_coefficients(self, T_agg: np.ndarray, cbeam: float = 0.0) -> np.ndarray:
        """Calculate the GSF expansion coefficients sm_coef of shape (2 * agg_order + 1, 4, 4)."""
        agg_order = int(round(math.sqrt(T_agg.shape[0] // 2 + 1) - 1.0))
        nbc = 6 * agg_order + 6
        fnr_t, bcof_t = precompute_fnr_bcof(nbc)
        
        # Precompute or retrieve cached Clebsch-Gordan caches
        if not hasattr(self, 'vc_cache_A') or getattr(self, 'vc_cache_agg_order', 0) != agg_order:
            self.vc_cache_A, self.vc_cache_B = _precompute_vc_caches_jit(agg_order, fnr_t, bcof_t)
            self.vc_cache_agg_order = agg_order
            
        return calculate_orientation_averaged_phase_matrix_coefficients_jit(
            T_agg, agg_order, cbeam, fnr_t, bcof_t, self.vc_cache_A, self.vc_cache_B
        )

    def calculate_phase_matrix_from_coefficients(self, sm_coef: np.ndarray, theta_deg: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Calculate the 4x4 Mueller matrix at given theta angles (degrees) from GSF coefficients.
        
        Returns:
            array of shape (len(theta_deg), 4, 4)
        """
        nodrexp = sm_coef.shape[0] - 1
        nbc = 3 * nodrexp + 6
        fnr_t, bcof_t = precompute_fnr_bcof(nbc)
        
        res = np.zeros((len(theta_deg), 4, 4))
        for i in range(len(theta_deg)):
            ct = math.cos(math.radians(theta_deg[i]))
            res[i] = calculate_phase_matrix_at_angle_jit(ct, sm_coef, nodrexp, fnr_t, bcof_t)
            
        if normalize and abs(sm_coef[0, 0, 0]) > 1e-300:
            res /= (4.0 * sm_coef[0, 0, 0])
            
        return res


@njit(fastmath=True)
def precompute_fnr_bcof(nbc: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the factorial recurrence tables for Wigner small-d recurrence."""
    fnr = np.zeros(2 * nbc + 1)
    for n in range(1, 2 * nbc + 1):
        fnr[n] = np.sqrt(n)
        
    bcof = np.zeros((nbc + 1, nbc + 1))
    bcof[0, 0] = 1.0
    for n in range(1, nbc + 1):
        bcof[n, 0] = 1.0
        bcof[0, n] = 1.0
        
    for n in range(1, nbc):
        for l in range(1, n + 1):
            bcof[n, l] = fnr[n + l] * bcof[n, l - 1] / fnr[l]
            bcof[l, n] = bcof[n, l]
        bcof[n + 1, n + 1] = fnr[n + n + 2] * fnr[n + n + 1] * bcof[n, n] / (fnr[n + 1] * fnr[n + 1])
        
    n = nbc
    for l in range(1, n + 1):
        bcof[n, l] = fnr[n + l] * bcof[n, l - 1] / fnr[l]
        bcof[l, n] = bcof[n, l]
        
    return fnr, bcof


@njit(fastmath=True)
def vcfuncuprec(m: int, n: int, k: int, l: int, wmax: int, vcn: np.ndarray, fnr: np.ndarray, bcof: np.ndarray) -> np.ndarray:
    """Upward recurrence logic for Clebsch-Gordan / linearization coefficients."""
    mk_abs = abs(m + k)
    nl_abs = abs(n - l)
    
    if nl_abs >= mk_abs:
        w = nl_abs
        if n >= l:
            m1, n1, l1, k1 = m, n, l, k
        else:
            m1, n1, l1, k1 = k, l, m, n
        vc1 = ((-1)**(k1 + l1)) * bcof[int(l1 + k1), int(w - m1 - k1)] * bcof[int(l1 - k1), int(w + m1 + k1)] / bcof[int(l1 + l1), int(w + w + 1)]
    else:
        w = mk_abs
        if m + k >= 0:
            vc1 = ((-1)**(n + m)) * bcof[int(n - l + w), int(l - k)] * bcof[int(l - n + w), int(n - m)] / bcof[int(w + w + 1), int(n + l - w)]
        else:
            vc1 = ((-1)**(l + k)) * bcof[int(n - l + w), int(l + k)] * bcof[int(l - n + w), int(n + m)] / bcof[int(w + w + 1), int(n + l - w)]
            
    w1 = int(w)
    vcn[w1] = vc1
    w = w1 + 1
    mk = m + k
    w2 = min(wmax, n + l)
    
    if w2 > w1:
        t1 = 2.0 * w * fnr[2*w + 1] * fnr[2*w - 1] / (
            fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1]
        )
        if w1 == 0:
            t2 = 0.5 * float(m - k)
        else:
            t2 = float((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2.0 * w * (w - 1))
        vcn[w] = t1 * t2 * vcn[w1]
        
    for w in range(w1 + 2, w2 + 1):
        t1 = 2.0 * w * fnr[2*w + 1] * fnr[2*w - 1] / (
            fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1]
        )
        t2 = float((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2.0 * w * (w - 1))
        t3 = (
            fnr[w - mk - 1] * fnr[w + mk - 1] * fnr[l - n + w - 1] * fnr[n - l + w - 1] * fnr[n + l - w + 2] * fnr[n + l + w]
        ) / (2.0 * (w - 1) * fnr[2*w - 3] * fnr[2*w - 1])
        vcn[w] = t1 * (t2 * vcn[w - 1] - t3 * vcn[w - 2])
        
    return vcn


@njit(fastmath=True)
def vcfunc(m: int, n: int, k: int, l: int, fnr: np.ndarray, bcof: np.ndarray) -> np.ndarray:
    """Compute Clebsch-Gordan / linearization coefficients using downwards/upwards recurrence."""
    wmax = n + l
    wmin = max(abs(n - l), abs(m + k))
    vcn = np.zeros(wmax + 1)
    
    vcn[wmax] = bcof[n + m, l + k] * bcof[n - m, l - k] / bcof[n + n, l + l]
    if wmin == wmax:
        return vcn
        
    vcn[wmax - 1] = vcn[wmax] * (l * m - k * n) * fnr[2 * (l + n) - 1] / (
        fnr[l] * fnr[n] * fnr[n + l + m + k] * fnr[n + l - m - k]
    )
    if wmin == wmax - 1:
        return vcn
        
    mk = m + k
    vcmax = abs(vcn[wmax]) + abs(vcn[wmax - 1])
    
    w = wmax
    while w >= wmin + 2:
        t1 = 2.0 * w * fnr[2*w + 1] * fnr[2*w - 1] / (
            fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1]
        )
        t2 = float((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2.0 * w * (w - 1))
        t3 = (
            fnr[w - mk - 1] * fnr[w + mk - 1] * fnr[l - n + w - 1] * fnr[n - l + w - 1] * fnr[n + l - w + 2] * fnr[n + l + w]
        ) / (2.0 * (w - 1) * fnr[2*w - 3] * fnr[2*w - 1])
        
        vcn[w - 2] = (t2 * vcn[w - 1] - vcn[w] / t1) / t3
        
        if (wmax - w) % 2 == 1:
            vctest = abs(vcn[w - 2]) + abs(vcn[w - 1])
            vcmax = max(vcmax, vctest)
            rat = vctest / vcmax
            if rat < 0.01:
                break
        w -= 1
        
    if w - 2 > wmin:
        wmax_up = w - 3
        vcn = vcfuncuprec(m, n, k, l, wmax_up, vcn, fnr, bcof)
        
    return vcn


@njit(fastmath=True)
def rotcoef(cbe: float, kmax: int, nmax: int, fnr: np.ndarray, bcof: np.ndarray) -> np.ndarray:
    """Compute Wigner-D rotation coefficients for scattering matrix expansion."""
    dc = np.zeros((2 * kmax + 1, nmax * (nmax + 2) + 1))
    
    if abs(cbe) >= 1.0:
        sbe = 0.0
    else:
        sbe = np.sqrt(abs((1.0 + cbe) * (1.0 - cbe)))
        
    cbe2 = 0.5 * (1.0 + cbe)
    sbe2 = 0.5 * (1.0 - cbe)
    
    offset_dk = nmax + 1
    dk0 = np.zeros(2 * nmax + 3)
    dk01 = np.zeros(2 * nmax + 3)
    
    sin = 1
    dk0[0 + offset_dk] = 1.0
    sben = 1.0
    dc[0 + kmax, 0] = 1.0
    dk01[0 + offset_dk] = 0.0
    
    for n in range(1, nmax + 1):
        knmax = min(n, kmax)
        nn1 = n * (n + 1)
        sin = -sin
        sben = sben * sbe / 2.0
        if sben < 1e-30:
            sben = 0.0
        
        dk0[n + offset_dk] = sin * sben * bcof[n, n]
        dk0[-n + offset_dk] = sin * dk0[n + offset_dk]
        dk01[n + offset_dk] = 0.0
        dk01[-n + offset_dk] = 0.0
        
        dc[0 + kmax, nn1 + n] = dk0[n + offset_dk]
        dc[0 + kmax, nn1 - n] = dk0[-n + offset_dk]
        
        for k in range(-n + 1, n):
            kn = nn1 + k
            dkt = dk01[k + offset_dk]
            dk01[k + offset_dk] = dk0[k + offset_dk]
            dk0[k + offset_dk] = (cbe * float(2 * n - 1) * dk01[k + offset_dk] - fnr[n - k - 1] * fnr[n + k - 1] * dkt) / (fnr[n + k] * fnr[n - k])
            dc[0 + kmax, kn] = dk0[k + offset_dk]
            
        im = 1
        for m in range(1, knmax + 1):
            im = -im
            fmn = 1.0 / (fnr[n - m + 1] * fnr[n + m])
            m1 = m - 1
            dkm0 = 0.0
            for k in range(-n, n + 1):
                kn = nn1 + k
                dkm1 = dkm0
                dkm0 = dc[m1 + kmax, kn]
                if k == n:
                    dkn1 = 0.0
                else:
                    dkn1 = dc[m1 + kmax, kn + 1]
                
                dc[m + kmax, kn] = (fnr[n + k] * fnr[n - k + 1] * cbe2 * dkm1 - fnr[n - k] * fnr[n + k + 1] * sbe2 * dkn1 - float(k) * sbe * dc[m1 + kmax, kn]) * fmn
                
                sign_val = -1.0 if k % 2 != 0 else 1.0
                dc[-m + kmax, nn1 - k] = dc[m + kmax, kn] * sign_val * im
                
    return dc


@njit(fastmath=True)
def rcr_basis_identity_jit(T_rcr: np.ndarray, nodr: int) -> np.ndarray:
    """Identity helper for the already-transformed two-block RCR basis.

    The aggregate T-matrix is already stored in the transformed Psi^(1)/Psi^(2)
    basis used by Appendix A, so no additional basis permutation is applied here.
    """
    return T_rcr


@njit(parallel=True, fastmath=True, nogil=True)
def _precompute_vc_caches_jit(nodr: int, fnr: np.ndarray, bcof: np.ndarray):
    """Precompute and cache Clebsch-Gordan coefficients for O(1) lookups."""
    # Cache A stores vcfunc(m, n, k, l) for n <= 2*nodr, l <= nodr.
    # Shape: (m_idx, n, k_idx, l, w) where m_idx = m + n, k_idx = k + l
    cache_A = np.zeros((4*nodr + 1, 2*nodr + 1, 2*nodr + 1, nodr + 1, 3*nodr + 1), dtype=np.float64)
    for n in prange(2 * nodr + 1):
        for l in range(1, nodr + 1):
            for m in range(-n, n + 1):
                for k in range(-l, l + 1):
                    vcs = vcfunc(m, n, k, l, fnr, bcof)
                    wmax = n + l
                    for w in range(wmax + 1):
                        cache_A[m + n, n, k + l, l, w] = vcs[w]
                        
    # Cache B stores vcfunc(m, n, k, l) for n <= nodr, l <= 2*nodr, and k in {0, -2}.
    # Shape: (m_idx, n, k_idx, l, w) where k_idx is 0 for k=0, 1 for k=-2
    cache_B = np.zeros((2*nodr + 1, nodr + 1, 2, 2*nodr + 1, 3*nodr + 1), dtype=np.float64)
    for n in prange(1, nodr + 1):
        for l in range(2 * nodr + 1):
            for m in range(-n, n + 1):
                # k = 0
                vcs0 = vcfunc(m, n, 0, l, fnr, bcof)
                wmax0 = n + l
                for w in range(wmax0 + 1):
                    cache_B[m + n, n, 0, l, w] = vcs0[w]
                # k = -2 (only valid if l >= 2)
                if l >= 2:
                    vcs2 = vcfunc(m, n, -2, l, fnr, bcof)
                    wmax2 = n + l
                    for w in range(wmax2 + 1):
                        cache_B[m + n, n, 1, l, w] = vcs2[w]
                        
    return cache_A, cache_B


@njit(parallel=True, fastmath=True, nogil=True)
def calculate_orientation_averaged_phase_matrix_coefficients_jit(
    T_agg: np.ndarray,
    nodr: int,
    cbeam: float,
    fnr: np.ndarray,
    bcof: np.ndarray,
    cache_A: np.ndarray,
    cache_B: np.ndarray
) -> np.ndarray:
    """Implement Mishchenko's analytical random orientation GSF expansion coefficients algorithm."""
    # 1. Precompute helper lookup tables to eliminate inner square roots, divisions, and complex powers
    sqrt_2n_plus_1 = np.zeros(nodr + 1, dtype=np.float64)
    one_over_sqrt_2n_plus_1 = np.zeros(nodr + 1, dtype=np.float64)
    one_over_2n_plus_1 = np.zeros(nodr + 1, dtype=np.float64)
    for n in range(nodr + 1):
        val = 2.0 * n + 1.0
        sqrt_val = np.sqrt(val)
        sqrt_2n_plus_1[n] = sqrt_val
        one_over_sqrt_2n_plus_1[n] = 1.0 / sqrt_val
        one_over_2n_plus_1[n] = 1.0 / val

    neg1_pow = np.zeros(3 * nodr + 2, dtype=np.float64)
    for i in range(3 * nodr + 2):
        neg1_pow[i] = (-1.0)**i

    pow_1j = np.zeros(2 * nodr + 1, dtype=np.complex128)
    for d in range(-nodr, nodr + 1):
        pow_1j[d + nodr] = (1j)**d

        # T_agg is already stored in the transformed two-block RCR basis.
        # No additional polarization-basis conversion is applied here.
        T_rcr = rcr_basis_identity_jit(T_agg, nodr)
        N_agg = nodr * (nodr + 2)
        
        tc = np.zeros((2, N_agg, 2, N_agg), dtype=np.complex128)
        for p_idx in range(2):
            for q_idx in range(2):
                tc[p_idx, :, q_idx, :] = T_rcr[
                    p_idx * N_agg : (p_idx + 1) * N_agg,
                    q_idx * N_agg : (q_idx + 1) * N_agg,
                ]
            
    pp = np.zeros((nodr + 1, 2, 2), dtype=np.complex128)
    for l_val in range(1, nodr + 1):
        gbn = np.exp(-((float(l_val) + 0.5) * cbeam)**2)
        cin = (1j)**(l_val + 1)
        val = -0.5 * cin * np.sqrt(2 * l_val + 1) * gbn
        pp[l_val, 0, 0] = val
        pp[l_val, 1, 0] = -val
        pp[l_val, 0, 1] = -val
        pp[l_val, 1, 1] = -val
        
    l_of_kl = np.zeros(N_agg, dtype=np.int32)
    k_of_kl = np.zeros(N_agg, dtype=np.int32)
    kl_idx = 0
    for l_val in range(1, nodr + 1):
        for k_val in range(-l_val, l_val + 1):
            l_of_kl[kl_idx] = l_val
            k_of_kl[kl_idx] = k_val
            kl_idx += 1
            
    for kl in range(N_agg):
        l_val = l_of_kl[kl]
        p_11 = pp[l_val, 0, 0]
        p_12 = pp[l_val, 0, 1]
        p_21 = pp[l_val, 1, 0]
        p_22 = pp[l_val, 1, 1]
        for p in range(2):
            for mn in range(N_agg):
                a = tc[p, mn, 0, kl]
                tc[p, mn, 0, kl] = a * p_11 + tc[p, mn, 1, kl] * p_12
                tc[p, mn, 1, kl] = a * p_21 + tc[p, mn, 1, kl] * p_22
                
    # Parallel outer loop using dm_all_w to avoid write conflicts
    dm_all_w = np.zeros((2 * nodr + 1, 2 * nodr + 3, 3, nodr + 1, 2, nodr + 1, 2), dtype=np.complex128)
    
    for w in prange(2 * nodr + 1):
        # Preallocate bm and am once per parallel thread (w iteration) to eliminate millions of dynamic allocations
        bm = np.zeros((2, N_agg, 2), dtype=np.complex128)
        am = np.zeros((2, nodr + 1, 2), dtype=np.complex128)
        
        for v in range(-w, w + 1):
            bm.fill(0.0)
            for n in range(1, nodr + 1):
                am.fill(0.0)
                for t in range(-n, n + 1):
                    tn = n * (n + 1) + t - 1
                    lmax = min(nodr, w + n)
                    l_start = max(1, abs(v - t))
                    l_start = max(l_start, abs(n - w))
                    
                    # Clebsch-Gordan lookup from Cache A
                    for l_val in range(l_start, lmax + 1):
                        tvl = l_val * (l_val + 1) + t - v - 1
                        val_vc = cache_A[v + w, w, -t + n, n, l_val]
                        # Unroll k and p loops
                        am[0, l_val, 0] += val_vc * tc[0, tn, 0, tvl]
                        am[0, l_val, 1] += val_vc * tc[1, tn, 0, tvl]
                        am[1, l_val, 0] += val_vc * tc[0, tn, 1, tvl]
                        am[1, l_val, 1] += val_vc * tc[1, tn, 1, tvl]
                                
                for m in range(-n, n + 1):
                    mn = n * (n + 1) + m - 1
                    # k = 0
                    u0 = m + 1
                    if abs(u0) <= w:
                        lmax = min(nodr, w + n)
                        l_start = max(1, abs(w - n))
                        for l_val in range(l_start, lmax + 1):
                            val_vc2 = cache_A[-u0 + w, w, m + n, n, l_val]
                            fl = -neg1_pow[l_val] * val_vc2 * one_over_2n_plus_1[l_val]
                            # Unroll p loop
                            bm[0, mn, 0] += am[0, l_val, 0] * fl
                            bm[0, mn, 1] += am[0, l_val, 1] * fl
                    # k = 1
                    u1 = m - 1
                    if abs(u1) <= w:
                        lmax = min(nodr, w + n)
                        l_start = max(1, abs(w - n))
                        for l_val in range(l_start, lmax + 1):
                            val_vc2 = cache_A[-u1 + w, w, m + n, n, l_val]
                            fl = -neg1_pow[l_val] * val_vc2 * one_over_2n_plus_1[l_val]
                            # Unroll p loop
                            bm[1, mn, 0] += am[1, l_val, 0] * fl
                            bm[1, mn, 1] += am[1, l_val, 1] * fl
                                    
            u_lim = min(w, nodr + 1)
            for u in range(-u_lim, u_lim + 1):
                u_idx = u + nodr + 1
                for ku in range(3):
                    if ku == 0:
                        k_phys, k1_phys = -1, -1
                        ik, ik1 = 0, 0
                    elif ku == 1:
                        k_phys, k1_phys = 1, 1
                        ik, ik1 = 1, 1
                    else:
                        k_phys, k1_phys = 1, -1
                        ik, ik1 = 1, 0
                        
                    m = u + k_phys
                    m1 = u + k1_phys
                    ns = max(1, abs(m))
                    n1s = max(1, abs(m1))
                    
                    for n_val in range(ns, nodr + 1):
                        nu = n_val * (n_val + 1) + m - 1
                        bm_ik_nu_0 = bm[ik, nu, 0]
                        bm_ik_nu_1 = bm[ik, nu, 1]
                        
                        n1s_val = max(n1s, n_val - 2 * nodr)
                        n1e_val = min(nodr, n_val + 2 * nodr)
                        fl_part = -sqrt_2n_plus_1[n_val] * (2.0 * w + 1.0)
                        
                        for n1_val in range(n1s_val, n1e_val + 1):
                            cin = pow_1j[n_val - n1_val + nodr]
                            nu1 = n1_val * (n1_val + 1) + m1 - 1
                            fl = fl_part * sqrt_2n_plus_1[n1_val]
                            
                            conj_bm_ik1_nu1_0 = np.conj(bm[ik1, nu1, 0])
                            conj_bm_ik1_nu1_1 = np.conj(bm[ik1, nu1, 1])
                            
                            cin_fl = cin * fl
                            
                            # Unroll p and p1 loops
                            dm_all_w[w, u_idx, ku, n_val, 0, n1_val, 0] += bm_ik_nu_0 * cin_fl * conj_bm_ik1_nu1_0
                            dm_all_w[w, u_idx, ku, n_val, 0, n1_val, 1] += bm_ik_nu_0 * cin_fl * conj_bm_ik1_nu1_1
                            dm_all_w[w, u_idx, ku, n_val, 1, n1_val, 0] += bm_ik_nu_1 * cin_fl * conj_bm_ik1_nu1_0
                            dm_all_w[w, u_idx, ku, n_val, 1, n1_val, 1] += bm_ik_nu_1 * cin_fl * conj_bm_ik1_nu1_1
                                    
    # Parallel reduction (summation)
    dm = np.zeros((2 * nodr + 3, 3, nodr + 1, 2, nodr + 1, 2), dtype=np.complex128)
    for w in range(2 * nodr + 1):
        dm += dm_all_w[w]
                                    
    aw = np.zeros((3, 3, 2 * nodr + 1), dtype=np.complex128)
    bw = np.zeros((3, 3, 2 * nodr + 1), dtype=np.complex128)
    cw = np.zeros(2 * nodr + 1, dtype=np.complex128)
    dw = np.zeros(2 * nodr + 1, dtype=np.complex128)
    
    for w in prange(2 * nodr + 1):
        fm = np.zeros((3, nodr + 1, 2, nodr + 1, 2), dtype=np.complex128)
        sqrt_2w_plus_1 = np.sqrt(2.0 * w + 1.0)
        
        for u in range(-nodr - 1, nodr + 2):
            u_idx = u + nodr + 1
            for ik in range(2):
                k_phys = -1 if ik == 0 else 1
                m = u + k_phys
                ns = max(1, abs(m))
                for n_val in range(ns, nodr + 1):
                    for n1_val in range(ns, nodr + 1):
                        if (n_val + n1_val < w) or (abs(n_val - n1_val) > w):
                            continue
                        val_vc = cache_B[m + n_val, n_val, 0, w, n1_val]
                        fl = -neg1_pow[n_val] * val_vc * sqrt_2w_plus_1 * one_over_sqrt_2n_plus_1[n1_val]
                        # Unroll p and p1 loops
                        fm[ik, n_val, 0, n1_val, 0] += dm[u_idx, ik, n_val, 0, n1_val, 0] * fl
                        fm[ik, n_val, 0, n1_val, 1] += dm[u_idx, ik, n_val, 0, n1_val, 1] * fl
                        fm[ik, n_val, 1, n1_val, 0] += dm[u_idx, ik, n_val, 1, n1_val, 0] * fl
                        fm[ik, n_val, 1, n1_val, 1] += dm[u_idx, ik, n_val, 1, n1_val, 1] * fl
                                
            if w >= 2:
                m = u + 1
                m1 = u - 1
                ns = max(1, abs(m))
                n1s = max(1, abs(m1))
                for n_val in range(ns, nodr + 1):
                    for n1_val in range(n1s, nodr + 1):
                        if (n_val + n1_val < w) or (abs(n_val - n1_val) > w):
                            continue
                        val_vc = cache_B[m + n_val, n_val, 1, w, n1_val]
                        fl = -neg1_pow[n_val] * val_vc * sqrt_2w_plus_1 * one_over_sqrt_2n_plus_1[n1_val]
                        # Unroll p and p1 loops
                        fm[2, n_val, 0, n1_val, 0] += dm[u_idx, 2, n_val, 0, n1_val, 0] * fl
                        fm[2, n_val, 0, n1_val, 1] += dm[u_idx, 2, n_val, 0, n1_val, 1] * fl
                        fm[2, n_val, 1, n1_val, 0] += dm[u_idx, 2, n_val, 1, n1_val, 0] * fl
                        fm[2, n_val, 1, n1_val, 1] += dm[u_idx, 2, n_val, 1, n1_val, 1] * fl
                                
        for n_val in range(1, nodr + 1):
            n1s_val = max(1, abs(n_val - w))
            n1e_val = min(nodr, n_val + w)
            in_sign = neg1_pow[n_val]
            
            for n1_val in range(n1s_val, n1e_val + 1):
                val_vc1 = cache_B[1 + n_val, n_val, 0, w, n1_val]
                fl = 2.0 * in_sign * val_vc1 * sqrt_2w_plus_1 * one_over_sqrt_2n_plus_1[n1_val]
                
                # Unroll p loop
                # p = 0
                p1_0 = 0 if (n_val + n1_val - w) % 2 == 0 else 1
                p1_prime_0 = 1 - p1_0
                aw[0, 0, w] += fm[0, n_val, 0, n1_val, p1_0] * fl
                bw[0, 0, w] += fm[0, n_val, 0, n1_val, p1_prime_0] * fl
                aw[0, 2, w] += fm[1, n_val, 0, n1_val, p1_0] * fl
                bw[0, 2, w] += fm[1, n_val, 0, n1_val, p1_prime_0] * fl
                bw[2, 1, w] += fm[2, n_val, 0, n1_val, p1_prime_0] * fl
                aw[2, 1, w] += fm[2, n_val, 0, n1_val, p1_0] * fl
                
                # p = 1
                p1_1 = 1 if (n_val + n1_val - w) % 2 == 0 else 0
                p1_prime_1 = 1 - p1_1
                aw[0, 0, w] += fm[0, n_val, 1, n1_val, p1_1] * fl
                bw[0, 0, w] += fm[0, n_val, 1, n1_val, p1_prime_1] * fl
                aw[0, 2, w] += fm[1, n_val, 1, n1_val, p1_1] * fl
                bw[0, 2, w] += fm[1, n_val, 1, n1_val, p1_prime_1] * fl
                bw[2, 1, w] += fm[2, n_val, 1, n1_val, p1_prime_1] * fl
                aw[2, 1, w] += fm[2, n_val, 1, n1_val, p1_1] * fl
                    
            if w < 2:
                continue
                
            for n1_val in range(n1s_val, n1e_val + 1):
                val_vc2 = cache_B[1 + n_val, n_val, 1, w, n1_val]
                fl = 2.0 * in_sign * val_vc2 * sqrt_2w_plus_1 * one_over_sqrt_2n_plus_1[n1_val]
                
                # Unroll p loop
                # p = 0
                p1_0 = 0 if (n_val + n1_val - w) % 2 == 0 else 1
                p1_prime_0 = 1 - p1_0
                sign_p1_0 = -1.0 if p1_0 == 0 else 1.0
                sign_p1_prime_0 = -1.0 if p1_prime_0 == 0 else 1.0
                
                aw[2, 0, w] += fm[0, n_val, 0, n1_val, p1_0] * fl * sign_p1_0
                bw[2, 0, w] += fm[0, n_val, 0, n1_val, p1_prime_0] * fl * sign_p1_prime_0
                aw[2, 2, w] += fm[1, n_val, 0, n1_val, p1_0] * fl * sign_p1_0
                bw[2, 2, w] += fm[1, n_val, 0, n1_val, p1_prime_0] * fl * sign_p1_prime_0
                
                # p = 1
                p1_1 = 1 if (n_val + n1_val - w) % 2 == 0 else 0
                p1_prime_1 = 1 - p1_1
                sign_p1_1 = -1.0 if p1_1 == 0 else 1.0
                sign_p1_prime_1 = -1.0 if p1_prime_1 == 0 else 1.0
                
                aw[2, 0, w] += fm[0, n_val, 1, n1_val, p1_1] * fl * sign_p1_1
                bw[2, 0, w] += fm[0, n_val, 1, n1_val, p1_prime_1] * fl * sign_p1_prime_1
                aw[2, 2, w] += fm[1, n_val, 1, n1_val, p1_1] * fl * sign_p1_1
                bw[2, 2, w] += fm[1, n_val, 1, n1_val, p1_prime_1] * fl * sign_p1_prime_1
                        
                fl2 = 2.0 * neg1_pow[n1_val + w] * val_vc2 * sqrt_2w_plus_1 * one_over_sqrt_2n_plus_1[n1_val]
                # Unroll p and p1 loops for cw and dw
                # p = 0, p1 = 0
                cw[w] += fm[2, n_val, 0, n1_val, 0] * fl * -1.0
                dw[w] += fm[2, n_val, 0, n1_val, 0] * fl2 * -1.0
                # p = 0, p1 = 1
                cw[w] += fm[2, n_val, 0, n1_val, 1] * fl * 1.0
                dw[w] += fm[2, n_val, 0, n1_val, 1] * fl2 * -1.0
                # p = 1, p1 = 0
                cw[w] += fm[2, n_val, 1, n1_val, 0] * fl * -1.0
                dw[w] += fm[2, n_val, 1, n1_val, 0] * fl2 * 1.0
                # p = 1, p1 = 1
                cw[w] += fm[2, n_val, 1, n1_val, 1] * fl * 1.0
                dw[w] += fm[2, n_val, 1, n1_val, 1] * fl2 * 1.0
                
    sm = np.zeros((2 * nodr + 1, 4, 4))
    for n in range(2 * nodr + 1):
        sm[n, 0, 0] = np.real(aw[0, 0, n] + aw[0, 2, n])
        sm[n, 0, 1] = np.real(aw[2, 0, n] + aw[2, 2, n])
        sm[n, 0, 2] = 2.0 * np.imag(aw[2, 1, n])
        sm[n, 0, 3] = np.real(aw[0, 2, n] - aw[0, 0, n])
        
        sm[n, 1, 1] = np.real(dw[n])
        sm[n, 1, 2] = np.imag(dw[n])
        sm[n, 1, 3] = np.real(aw[2, 2, n] - aw[2, 0, n])
        
        sm[n, 2, 0] = -np.imag(bw[2, 0, n] + bw[2, 2, n])
        sm[n, 2, 1] = np.imag(cw[n])
        sm[n, 2, 2] = np.real(cw[n])
        sm[n, 2, 3] = np.imag(bw[2, 0, n] - bw[2, 2, n])
        
        sm[n, 3, 0] = np.real(bw[0, 0, n] + bw[0, 2, n])
        sm[n, 3, 1] = 2.0 * np.real(bw[2, 1, n])
        sm[n, 3, 3] = np.real(bw[0, 2, n] - bw[0, 0, n])
        
        sm[n, 1, 0] = sm[n, 0, 1]
        sm[n, 3, 2] = -sm[n, 2, 3]
        
    return sm


@njit(fastmath=True)
def calculate_phase_matrix_at_angle_jit(
    ct: float,
    smc: np.ndarray,
    nodrexp: int,
    fnr: np.ndarray,
    bcof: np.ndarray
) -> np.ndarray:
    """Accumulate the 4x4 Mueller matrix at an angle ct=cos(theta) from GSF coefficients."""
    dc = rotcoef(ct, 2, nodrexp, fnr, bcof)
    sm = np.zeros((4, 4))
    for n in range(nodrexp + 1):
        nn0 = n * (n + 1)
        nnp2 = nn0 + 2
        nnm2 = nn0 - 2
        
        sm[0, 0] += dc[0 + 2, nn0] * smc[n, 0, 0]
        sm[0, 3] += dc[0 + 2, nn0] * smc[n, 0, 3]
        sm[3, 0] += dc[0 + 2, nn0] * smc[n, 3, 0]
        sm[3, 3] += dc[0 + 2, nn0] * smc[n, 3, 3]
        
        if n >= 2:
            sm[0, 1] += dc[2 + 2, nn0] * smc[n, 0, 1]
            sm[1, 3] += dc[2 + 2, nn0] * smc[n, 1, 3]
            sm[2, 3] += dc[2 + 2, nn0] * smc[n, 2, 3]
            sm[0, 2] += dc[2 + 2, nn0] * smc[n, 0, 2]
            sm[2, 0] = sm[0, 2] + dc[2 + 2, nn0] * smc[n, 2, 0]
            sm[3, 1] += dc[2 + 2, nn0] * smc[n, 3, 1]
            sm[1, 1] += dc[2 + 2, nnm2] * smc[n, 1, 1] + dc[2 + 2, nnp2] * smc[n, 2, 2]
            sm[1, 2] += dc[2 + 2, nnp2] * smc[n, 1, 2] + dc[2 + 2, nnp2] * smc[n, 2, 1]
            sm[2, 2] += -dc[2 + 2, nnm2] * smc[n, 1, 1] + dc[2 + 2, nnp2] * smc[n, 2, 2]
            sm[2, 1] += dc[2 + 2, nnp2] * smc[n, 1, 2] - dc[2 + 2, nnp2] * smc[n, 2, 1]
            
    sm[1, 0] = sm[0, 1]
    sm[3, 2] = -sm[2, 3]
    return sm
