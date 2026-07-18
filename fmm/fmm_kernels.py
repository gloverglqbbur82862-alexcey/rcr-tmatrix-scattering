#!/usr/bin/env python3
"""FMM Kernels: Numba JIT-compiled hot paths for near-field and horizontal M2L passes."""

import numpy as np
from numba import njit
from rcr_engine.wigner_rotation import (
    _build_cached_d_jit,
    _rotate_to_z_cached_jit,
    _rotate_from_z_cached_jit,
)
from rcr_engine.coaxial_translation import _apply_coaxial_translation_loop


@njit(fastmath=True)
def _pad_slice_VSWF_accumulate_jit(
    src: np.ndarray, 
    tgt: np.ndarray, 
    src_order: int, 
    tgt_order: int
):
    nblk_src = src_order * (src_order + 2)
    nblk_tgt = tgt_order * (tgt_order + 2)
    copy_len = min(nblk_src, nblk_tgt)
    tgt[:copy_len] += src[:copy_len]
    tgt[nblk_tgt : nblk_tgt + copy_len] += src[nblk_src : nblk_src + copy_len]


@njit(fastmath=True)
def _rotate_to_z_cached_phase_jit(
    p: int, 
    nblk_half: int, 
    x: np.ndarray, 
    phase_pos: np.ndarray, 
    phase_offset: int, 
    d_blocks: np.ndarray, 
    d_offset_start: int
) -> np.ndarray:
    """Rotate coefficients x to Z-axis using cached phase and Wigner-D blocks."""
    y = np.zeros_like(x)
    d_offset = d_offset_start
    
    for n in range(1, p + 1):
        block_dim = 2 * n + 1
        for pol in range(2):
            offset = pol * nblk_half
            idx_start = offset + n * n - 1
            
            block_x_phased = np.empty(block_dim, dtype=np.complex128)
            for j in range(block_dim):
                m = j - n
                block_x_phased[j] = x[idx_start + j] * phase_pos[phase_offset + (m + p)]
                
            for i in range(block_dim):
                val = 0.0 + 0.0j
                for j in range(block_dim):
                    val += d_blocks[d_offset + i * block_dim + j] * block_x_phased[j]
                y[idx_start + i] = val
                
        d_offset += block_dim ** 2
    return y


@njit(fastmath=True)
def _rotate_from_z_cached_phase_jit(
    p: int, 
    nblk_half: int, 
    y: np.ndarray, 
    phase_neg: np.ndarray, 
    phase_offset: int, 
    d_blocks: np.ndarray, 
    d_offset_start: int
) -> np.ndarray:
    """Rotate coefficients y back from Z-axis using cached phase and Wigner-D transpose blocks."""
    x_out = np.zeros_like(y)
    d_offset = d_offset_start
    
    for n in range(1, p + 1):
        block_dim = 2 * n + 1
        for pol in range(2):
            offset = pol * nblk_half
            idx_start = offset + n * n - 1
            
            block_x_temp = np.empty(block_dim, dtype=np.complex128)
            for j in range(block_dim):
                val = 0.0 + 0.0j
                for i in range(block_dim):
                    val += d_blocks[d_offset + i * block_dim + j] * y[idx_start + i]
                block_x_temp[j] = val
                
            for j in range(block_dim):
                m = j - n
                x_out[idx_start + j] = block_x_temp[j] * phase_neg[phase_offset + (m + p)]
                
        d_offset += block_dim ** 2
    return x_out


@njit(fastmath=True)
def evaluate_recurrence_hankel_scalar_jit(L_max: int, r: float) -> np.ndarray:
    """Evaluate spherical Hankel functions h_L^(1)(r) using stable forward recurrence for scalar r."""
    h = np.empty(L_max + 1, dtype=np.complex128)
    exp_phase = np.exp(1j * r)
    h[0] = -1j * exp_phase / r
    h[1] = (-1.0 / r - 1j / (r ** 2)) * exp_phase
    for n in range(2, L_max + 1):
        h[n] = ((2 * n - 1) / r) * h[n - 1] - h[n - 2]
    return h


@njit(fastmath=True)
def near_field_direct_pass_batch_jit(
    near_pairs: np.ndarray,      # (num_near_pairs, 2)
    X_padded: np.ndarray,        # (N, coeff_size, B)
    kpos: np.ndarray,            # (N, 3)
    p_fmm: int,
    nblk_half: int,
    fac_table: np.ndarray,
    tran_coef_arr: np.ndarray,
    fnr: np.ndarray
) -> np.ndarray:                 # returns (N, coeff_size, B)
    """Perform near-field pairwise singular translations across B batch columns."""
    N = X_padded.shape[0]
    coeff_size = X_padded.shape[1]
    B = X_padded.shape[2]
    
    Y = np.zeros((N, coeff_size, B), dtype=np.complex128)
    wmax_global = 2 * p_fmm
    
    # Precompute Legendre function at ct = 1.0 (constant for Z-axis translation)
    dc = np.zeros(wmax_global + 1, dtype=np.float64)
    dc[0] = 1.0
    if wmax_global >= 1:
        dc[1] = fnr[1] * dc[0]  # ct = 1.0
    for n in range(1, wmax_global):
        nm = fnr[n] * fnr[n]
        np1m = fnr[n + 1] * fnr[n + 1]
        if abs(np1m) < 1e-300:
            dc[n + 1] = 0.0
        else:
            dc[n + 1] = (-(nm) * dc[n - 1] + float(2 * n + 1) * dc[n]) / np1m
            
    num_near_pairs = near_pairs.shape[0]
    for idx in range(num_near_pairs):
        j = near_pairs[idx, 0]  # source
        i = near_pairs[idx, 1]  # target
        
        tvec = kpos[i] - kpos[j]
        r = np.linalg.norm(tvec)
        if r < 1e-14:
            for b in range(B):
                Y[i, :, b] += X_padded[j, :, b]
            continue
            
        val = tvec[2] / r
        val = max(-1.0, min(1.0, val))
        theta = np.arccos(val)
        phi = np.arctan2(tvec[1], tvec[0])
        
        # Build cached Wigner-d rotation flat blocks once for the pair
        d_blocks_flat = _build_cached_d_jit(p_fmm, theta, fac_table)
        
        # Evaluate Hankel functions once for the pair
        hn = evaluate_recurrence_hankel_scalar_jit(wmax_global, r)
        
        # Loop over batch columns
        for b in range(B):
            x = X_padded[j, :, b]
            x_rot = _rotate_to_z_cached_jit(p_fmm, nblk_half, phi, x, d_blocks_flat)
            y_rot = _apply_coaxial_translation_loop(p_fmm, nblk_half, x_rot, hn, dc, tran_coef_arr)
            y = _rotate_from_z_cached_jit(p_fmm, nblk_half, phi, y_rot, d_blocks_flat)
            Y[i, :, b] += y
            
    return Y


@njit(fastmath=True)
def horizontal_pass_level_batch_jit(
    m2l_pairs: np.ndarray,        # (num_pairs, 2)
    multipole_coeffs: np.ndarray, # (total_multipole_size, B)
    local_coeffs: np.ndarray,     # (total_local_size, B)
    cell_offsets: np.ndarray,     # (ncells,)
    cell_centers: np.ndarray,     # (ncells, 3)
    P_l: int,
    tran_coef_arr: np.ndarray,
    fac_table: np.ndarray,
    fnr: np.ndarray
):
    """Perform horizontal M2L cell translations at a specific level across B batch columns."""
    B = multipole_coeffs.shape[1]
    wmax_global = 2 * P_l
    nblk_half = P_l * (P_l + 2)
    coeff_size_l = 2 * nblk_half
    
    # Precompute Legendre function at ct = 1.0
    dc = np.zeros(wmax_global + 1, dtype=np.float64)
    dc[0] = 1.0
    if wmax_global >= 1:
        dc[1] = fnr[1] * dc[0]
    for n in range(1, wmax_global):
        nm = fnr[n] * fnr[n]
        np1m = fnr[n + 1] * fnr[n + 1]
        if abs(np1m) < 1e-300:
            dc[n + 1] = 0.0
        else:
            dc[n + 1] = (-(nm) * dc[n - 1] + float(2 * n + 1) * dc[n]) / np1m
            
    num_pairs = m2l_pairs.shape[0]
    for idx in range(num_pairs):
        src = m2l_pairs[idx, 0]
        tgt = m2l_pairs[idx, 1]
        
        tvec = cell_centers[tgt] - cell_centers[src]
        r = np.linalg.norm(tvec)
        if r < 1e-14:
            src_offset = cell_offsets[src]
            tgt_offset = cell_offsets[tgt]
            for b in range(B):
                local_coeffs[tgt_offset : tgt_offset + coeff_size_l, b] += multipole_coeffs[src_offset : src_offset + coeff_size_l, b]
            continue
            
        val = tvec[2] / r
        val = max(-1.0, min(1.0, val))
        theta = np.arccos(val)
        phi = np.arctan2(tvec[1], tvec[0])
        
        d_blocks_flat = _build_cached_d_jit(P_l, theta, fac_table)
        hn = evaluate_recurrence_hankel_scalar_jit(wmax_global, r)
        
        src_offset = cell_offsets[src]
        tgt_offset = cell_offsets[tgt]
        
        for b in range(B):
            x = multipole_coeffs[src_offset : src_offset + coeff_size_l, b]
            x_rot = _rotate_to_z_cached_jit(P_l, nblk_half, phi, x, d_blocks_flat)
            y_rot = _apply_coaxial_translation_loop(P_l, nblk_half, x_rot, hn, dc, tran_coef_arr)
            y = _rotate_from_z_cached_jit(P_l, nblk_half, phi, y_rot, d_blocks_flat)
            local_coeffs[tgt_offset : tgt_offset + coeff_size_l, b] += y


@njit(fastmath=True)
def m2l_pass_group_jit(
    edges_int: np.ndarray,          # (num_edges, 8)
    edges_geom: np.ndarray,         # (num_edges, 3)
    d_offsets: np.ndarray,          # (num_edges + 1,)
    d_blocks_flat: np.ndarray,      # (total_d_size,)
    radial_offsets: np.ndarray,     # (num_edges + 1,)
    radial_flat: np.ndarray,        # (total_radial_size,)
    phase_offsets: np.ndarray,      # (num_edges + 1,)
    phase_pos_flat: np.ndarray,     # (total_phase_pos_size,)
    phase_neg_flat: np.ndarray,     # (total_phase_neg_size,)
    multipole_coeffs: np.ndarray,   # (total_multipole_size, B) - Input
    local_coeffs: np.ndarray,       # (total_local_size, B) - Output
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
    has_d_cache: bool,
    fac_table: np.ndarray
):
    num_edges = edges_int.shape[0]
    B = multipole_coeffs.shape[1]
    
    src_order = edges_int[0, 4]
    tgt_order = edges_int[0, 5]
    work_order = edges_int[0, 6]
    
    nblk_half_work = work_order * (work_order + 2)
    coeff_size_work = 2 * nblk_half_work
    
    for e in range(num_edges):
        src_offset = edges_int[e, 2]
        tgt_offset = edges_int[e, 3]
        flags = edges_int[e, 7]
        
        # Zero translation case
        if flags == 1:
            for b in range(B):
                src_slice = multipole_coeffs[src_offset : src_offset + coeff_size_work, b]
                tgt_slice = local_coeffs[tgt_offset : tgt_offset + coeff_size_work, b]
                _pad_slice_VSWF_accumulate_jit(src_slice, tgt_slice, src_order, tgt_order)
            continue
            
        # Normal translation case
        theta = edges_geom[e, 1]
        d_offset = d_offsets[e]
        if has_d_cache:
            d_blocks = d_blocks_flat
        else:
            d_blocks = _build_cached_d_jit(work_order, theta, fac_table)
            d_offset = 0
            
        radial_offset = radial_offsets[e]
        phase_offset = phase_offsets[e]
        
        # Bounded slice for radial functions
        r_slice = radial_flat[radial_offsets[e] : radial_offsets[e + 1]]
        
        for b in range(B):
            src_slice = multipole_coeffs[src_offset : src_offset + coeff_size_work, b]
            
            x_rot = _rotate_to_z_cached_phase_jit(
                work_order, nblk_half_work, src_slice, phase_pos_flat, phase_offset, d_blocks, d_offset
            )
            
            y_rot = _apply_coaxial_translation_loop(
                work_order, nblk_half_work, x_rot, r_slice, dc, tran_coef_arr
            )
            
            y = _rotate_from_z_cached_phase_jit(
                work_order, nblk_half_work, y_rot, phase_neg_flat, phase_offset, d_blocks, d_offset
            )
            
            tgt_slice = local_coeffs[tgt_offset : tgt_offset + coeff_size_work, b]
            _pad_slice_VSWF_accumulate_jit(y, tgt_slice, work_order, tgt_order)


@njit(fastmath=True)
def p2m_pass_group_jit(
    edges_int: np.ndarray,          # (num_edges, 8)
    edges_geom: np.ndarray,         # (num_edges, 3)
    d_offsets: np.ndarray,          # (num_edges + 1,)
    d_blocks_flat: np.ndarray,      # (total_d_size,)
    radial_offsets: np.ndarray,     # (num_edges + 1,)
    radial_flat: np.ndarray,        # (total_radial_size,)
    phase_offsets: np.ndarray,      # (num_edges + 1,)
    phase_pos_flat: np.ndarray,     # (total_phase_pos_size,)
    phase_neg_flat: np.ndarray,     # (total_phase_neg_size,)
    multipole_coeffs: np.ndarray,   # (total_multipole_size, B) - Output
    X_padded: np.ndarray,           # (nsph, coeff_size, B) - Input
    p_fmm: int,
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
    has_d_cache: bool,
    fac_table: np.ndarray
):
    num_edges = edges_int.shape[0]
    B = X_padded.shape[2]
    
    src_order = edges_int[0, 4]
    tgt_order = edges_int[0, 5]
    work_order = edges_int[0, 6]
    
    nblk_half_work = work_order * (work_order + 2)
    coeff_size_work = 2 * nblk_half_work
    
    for e in range(num_edges):
        src_idx = edges_int[e, 0]
        tgt_offset = edges_int[e, 3]
        flags = edges_int[e, 7]
        
        # Zero translation case
        if flags == 1:
            for b in range(B):
                x_sphere = X_padded[src_idx, :, b]
                tgt_slice = multipole_coeffs[tgt_offset : tgt_offset + coeff_size_work, b]
                _pad_slice_VSWF_accumulate_jit(x_sphere, tgt_slice, src_order, tgt_order)
            continue
            
        # Normal translation case
        theta = edges_geom[e, 1]
        d_offset = d_offsets[e]
        if has_d_cache:
            d_blocks = d_blocks_flat
        else:
            d_blocks = _build_cached_d_jit(work_order, theta, fac_table)
            d_offset = 0
            
        radial_offset = radial_offsets[e]
        phase_offset = phase_offsets[e]
        
        # Bounded slice for radial functions
        r_slice = radial_flat[radial_offsets[e] : radial_offsets[e + 1]]
        
        for b in range(B):
            x_sphere = X_padded[src_idx, :, b]
            
            x_work = np.zeros(coeff_size_work, dtype=np.complex128)
            _pad_slice_VSWF_accumulate_jit(x_sphere, x_work, src_order, work_order)
            
            x_rot = _rotate_to_z_cached_phase_jit(
                work_order, nblk_half_work, x_work, phase_pos_flat, phase_offset, d_blocks, d_offset
            )
            
            y_rot = _apply_coaxial_translation_loop(
                work_order, nblk_half_work, x_rot, r_slice, dc, tran_coef_arr
            )
            
            y_work = _rotate_from_z_cached_phase_jit(
                work_order, nblk_half_work, y_rot, phase_neg_flat, phase_offset, d_blocks, d_offset
            )
            
            tgt_slice = multipole_coeffs[tgt_offset : tgt_offset + coeff_size_work, b]
            _pad_slice_VSWF_accumulate_jit(y_work, tgt_slice, work_order, tgt_order)


@njit(fastmath=True)
def m2m_pass_group_jit(
    edges_int: np.ndarray,          # (num_edges, 8)
    edges_geom: np.ndarray,         # (num_edges, 3)
    d_offsets: np.ndarray,          # (num_edges + 1,)
    d_blocks_flat: np.ndarray,      # (total_d_size,)
    radial_offsets: np.ndarray,     # (num_edges + 1,)
    radial_flat: np.ndarray,        # (total_radial_size,)
    phase_offsets: np.ndarray,      # (num_edges + 1,)
    phase_pos_flat: np.ndarray,     # (total_phase_pos_size,)
    phase_neg_flat: np.ndarray,     # (total_phase_neg_size,)
    multipole_coeffs: np.ndarray,   # (total_multipole_size, B) - Input and Output
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
    has_d_cache: bool,
    fac_table: np.ndarray
):
    num_edges = edges_int.shape[0]
    B = multipole_coeffs.shape[1]
    
    src_order = edges_int[0, 4]
    tgt_order = edges_int[0, 5]
    work_order = edges_int[0, 6]
    
    nblk_half_src = src_order * (src_order + 2)
    nblk_half_work = work_order * (work_order + 2)
    coeff_size_work = 2 * nblk_half_work
    coeff_size_tgt = 2 * tgt_order * (tgt_order + 2)
    
    for e in range(num_edges):
        src_offset = edges_int[e, 2]
        tgt_offset = edges_int[e, 3]
        flags = edges_int[e, 7]
        
        # Zero translation case
        if flags == 1:
            for b in range(B):
                src_slice = multipole_coeffs[src_offset : src_offset + 2 * nblk_half_src, b]
                tgt_slice = multipole_coeffs[tgt_offset : tgt_offset + coeff_size_tgt, b]
                _pad_slice_VSWF_accumulate_jit(src_slice, tgt_slice, src_order, tgt_order)
            continue
            
        # Normal translation case
        theta = edges_geom[e, 1]
        d_offset = d_offsets[e]
        if has_d_cache:
            d_blocks = d_blocks_flat
        else:
            d_blocks = _build_cached_d_jit(work_order, theta, fac_table)
            d_offset = 0
            
        radial_offset = radial_offsets[e]
        phase_offset = phase_offsets[e]
        
        # Bounded slice for radial functions
        r_slice = radial_flat[radial_offsets[e] : radial_offsets[e + 1]]
        
        for b in range(B):
            src_slice = multipole_coeffs[src_offset : src_offset + 2 * nblk_half_src, b]
            
            x_work = np.zeros(coeff_size_work, dtype=np.complex128)
            _pad_slice_VSWF_accumulate_jit(src_slice, x_work, src_order, work_order)
            
            x_rot = _rotate_to_z_cached_phase_jit(
                work_order, nblk_half_work, x_work, phase_pos_flat, phase_offset, d_blocks, d_offset
            )
            
            y_rot = _apply_coaxial_translation_loop(
                work_order, nblk_half_work, x_rot, r_slice, dc, tran_coef_arr
            )
            
            y_work = _rotate_from_z_cached_phase_jit(
                work_order, nblk_half_work, y_rot, phase_neg_flat, phase_offset, d_blocks, d_offset
            )
            
            tgt_slice = multipole_coeffs[tgt_offset : tgt_offset + coeff_size_tgt, b]
            _pad_slice_VSWF_accumulate_jit(y_work, tgt_slice, work_order, tgt_order)


@njit(fastmath=True)
def l2l_pass_group_jit(
    edges_int: np.ndarray,          # (num_edges, 8)
    edges_geom: np.ndarray,         # (num_edges, 3)
    d_offsets: np.ndarray,          # (num_edges + 1,)
    d_blocks_flat: np.ndarray,      # (total_d_size,)
    radial_offsets: np.ndarray,     # (num_edges + 1,)
    radial_flat: np.ndarray,        # (total_radial_size,)
    phase_offsets: np.ndarray,      # (num_edges + 1,)
    phase_pos_flat: np.ndarray,     # (total_phase_pos_size,)
    phase_neg_flat: np.ndarray,     # (total_phase_neg_size,)
    local_coeffs: np.ndarray,       # (total_local_size, B) - Input and Output
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
    has_d_cache: bool,
    fac_table: np.ndarray
):
    num_edges = edges_int.shape[0]
    B = local_coeffs.shape[1]
    
    src_order = edges_int[0, 4]
    tgt_order = edges_int[0, 5]
    work_order = edges_int[0, 6]
    
    nblk_half_src = src_order * (src_order + 2)
    nblk_half_work = work_order * (work_order + 2)
    nblk_half_tgt = tgt_order * (tgt_order + 2)
    
    coeff_size_work = 2 * nblk_half_work
    coeff_size_tgt = 2 * nblk_half_tgt
    
    for e in range(num_edges):
        src_offset = edges_int[e, 2]
        tgt_offset = edges_int[e, 3]
        flags = edges_int[e, 7]
        
        # Zero translation case
        if flags == 1:
            for b in range(B):
                src_slice = local_coeffs[src_offset : src_offset + 2 * nblk_half_src, b]
                tgt_slice = local_coeffs[tgt_offset : tgt_offset + coeff_size_tgt, b]
                _pad_slice_VSWF_accumulate_jit(src_slice, tgt_slice, src_order, tgt_order)
            continue
            
        # Normal translation case
        theta = edges_geom[e, 1]
        d_offset = d_offsets[e]
        if has_d_cache:
            d_blocks = d_blocks_flat
        else:
            d_blocks = _build_cached_d_jit(work_order, theta, fac_table)
            d_offset = 0
            
        radial_offset = radial_offsets[e]
        phase_offset = phase_offsets[e]
        
        # Bounded slice for radial functions
        r_slice = radial_flat[radial_offsets[e] : radial_offsets[e + 1]]
        
        for b in range(B):
            src_slice = local_coeffs[src_offset : src_offset + 2 * nblk_half_src, b]
            
            x_work = np.zeros(coeff_size_work, dtype=np.complex128)
            _pad_slice_VSWF_accumulate_jit(src_slice, x_work, src_order, work_order)
            
            x_rot = _rotate_to_z_cached_phase_jit(
                work_order, nblk_half_work, x_work, phase_pos_flat, phase_offset, d_blocks, d_offset
            )
            
            y_rot = _apply_coaxial_translation_loop(
                work_order, nblk_half_work, x_rot, r_slice, dc, tran_coef_arr
            )
            
            y_work = _rotate_from_z_cached_phase_jit(
                work_order, nblk_half_work, y_rot, phase_neg_flat, phase_offset, d_blocks, d_offset
            )
            
            tgt_slice = local_coeffs[tgt_offset : tgt_offset + coeff_size_tgt, b]
            _pad_slice_VSWF_accumulate_jit(y_work, tgt_slice, work_order, tgt_order)


@njit(fastmath=True)
def l2p_pass_group_jit(
    edges_int: np.ndarray,          # (num_edges, 8)
    edges_geom: np.ndarray,         # (num_edges, 3)
    d_offsets: np.ndarray,          # (num_edges + 1,)
    d_blocks_flat: np.ndarray,      # (total_d_size,)
    radial_offsets: np.ndarray,     # (num_edges + 1,)
    radial_flat: np.ndarray,        # (total_radial_size,)
    phase_offsets: np.ndarray,      # (num_edges + 1,)
    phase_pos_flat: np.ndarray,     # (total_phase_pos_size,)
    phase_neg_flat: np.ndarray,     # (total_phase_neg_size,)
    local_coeffs: np.ndarray,       # (total_local_size, B) - Input
    Y_far_padded: np.ndarray,       # (nsph, coeff_size, B) - Output
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
    has_d_cache: bool,
    fac_table: np.ndarray
):
    num_edges = edges_int.shape[0]
    B = local_coeffs.shape[1]
    
    src_order = edges_int[0, 4]
    tgt_order = edges_int[0, 5]
    work_order = edges_int[0, 6]
    
    nblk_half_src = src_order * (src_order + 2)
    nblk_half_work = work_order * (work_order + 2)
    coeff_size_work = 2 * nblk_half_work
    
    for e in range(num_edges):
        src_offset = edges_int[e, 2]
        tgt_idx = edges_int[e, 1]
        flags = edges_int[e, 7]
        
        # Zero translation case
        if flags == 1:
            for b in range(B):
                src_slice = local_coeffs[src_offset : src_offset + 2 * nblk_half_src, b]
                tgt_slice = Y_far_padded[tgt_idx, :, b]
                _pad_slice_VSWF_accumulate_jit(src_slice, tgt_slice, src_order, tgt_order)
            continue
            
        # Normal translation case
        theta = edges_geom[e, 1]
        d_offset = d_offsets[e]
        if has_d_cache:
            d_blocks = d_blocks_flat
        else:
            d_blocks = _build_cached_d_jit(work_order, theta, fac_table)
            d_offset = 0
            
        radial_offset = radial_offsets[e]
        phase_offset = phase_offsets[e]
        
        # Bounded slice for radial functions
        r_slice = radial_flat[radial_offsets[e] : radial_offsets[e + 1]]
        
        for b in range(B):
            src_slice = local_coeffs[src_offset : src_offset + 2 * nblk_half_src, b]
            
            x_work = np.zeros(coeff_size_work, dtype=np.complex128)
            _pad_slice_VSWF_accumulate_jit(src_slice, x_work, src_order, work_order)
            
            x_rot = _rotate_to_z_cached_phase_jit(
                work_order, nblk_half_work, x_work, phase_pos_flat, phase_offset, d_blocks, d_offset
            )
            
            y_rot = _apply_coaxial_translation_loop(
                work_order, nblk_half_work, x_rot, r_slice, dc, tran_coef_arr
            )
            
            y_work = _rotate_from_z_cached_phase_jit(
                work_order, nblk_half_work, y_rot, phase_neg_flat, phase_offset, d_blocks, d_offset
            )
            
            tgt_slice = Y_far_padded[tgt_idx, :, b]
            _pad_slice_VSWF_accumulate_jit(y_work, tgt_slice, work_order, tgt_order)


@njit(fastmath=True)
def near_field_pass_group_jit(
    edges_int: np.ndarray,          # (num_edges, 8)
    edges_geom: np.ndarray,         # (num_edges, 3)
    d_offsets: np.ndarray,          # (num_edges + 1,)
    d_blocks_flat: np.ndarray,      # (total_d_size,)
    radial_offsets: np.ndarray,     # (num_edges + 1,)
    radial_flat: np.ndarray,        # (total_radial_size,)
    phase_offsets: np.ndarray,      # (num_edges + 1,)
    phase_pos_flat: np.ndarray,     # (total_phase_pos_size,)
    phase_neg_flat: np.ndarray,     # (total_phase_neg_size,)
    X_padded: np.ndarray,           # (nsph, coeff_size, B) - Input
    Y_near: np.ndarray,             # (nsph, coeff_size, B) - Output
    dc: np.ndarray,
    tran_coef_arr: np.ndarray,
    has_d_cache: bool,
    fac_table: np.ndarray
):
    num_edges = edges_int.shape[0]
    B = X_padded.shape[2]
    
    src_order = edges_int[0, 4]
    tgt_order = edges_int[0, 5]
    work_order = edges_int[0, 6]
    
    nblk_half_work = work_order * (work_order + 2)
    coeff_size_work = 2 * nblk_half_work
    
    for e in range(num_edges):
        src_idx = edges_int[e, 0]
        tgt_idx = edges_int[e, 1]
        flags = edges_int[e, 7]
        
        # Zero translation case
        if flags == 1:
            for b in range(B):
                src_slice = X_padded[src_idx, :, b]
                tgt_slice = Y_near[tgt_idx, :, b]
                _pad_slice_VSWF_accumulate_jit(src_slice, tgt_slice, src_order, tgt_order)
            continue
            
        # Normal translation case
        theta = edges_geom[e, 1]
        d_offset = d_offsets[e]
        if has_d_cache:
            d_blocks = d_blocks_flat
        else:
            d_blocks = _build_cached_d_jit(work_order, theta, fac_table)
            d_offset = 0
            
        radial_offset = radial_offsets[e]
        phase_offset = phase_offsets[e]
        
        # Bounded slice for radial functions
        r_slice = radial_flat[radial_offsets[e] : radial_offsets[e + 1]]
        
        for b in range(B):
            src_slice = X_padded[src_idx, :, b]
            
            x_rot = _rotate_to_z_cached_phase_jit(
                work_order, nblk_half_work, src_slice, phase_pos_flat, phase_offset, d_blocks, d_offset
            )
            
            y_rot = _apply_coaxial_translation_loop(
                work_order, nblk_half_work, x_rot, r_slice, dc, tran_coef_arr
            )
            
            y = _rotate_from_z_cached_phase_jit(
                work_order, nblk_half_work, y_rot, phase_neg_flat, phase_offset, d_blocks, d_offset
            )
            
            tgt_slice = Y_near[tgt_idx, :, b]
            _pad_slice_VSWF_accumulate_jit(y, tgt_slice, work_order, tgt_order)

