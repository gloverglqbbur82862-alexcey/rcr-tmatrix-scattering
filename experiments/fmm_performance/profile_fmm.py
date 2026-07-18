#!/usr/bin/env python3
"""Profiling script to inspect FMM apply_batch execution breakdown."""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fmm.fmm_solver import FMMMultisphereSystem
from fmm.fmm_operator import _rcr_translate
from experiments.fmm_performance.benchmark import make_mie_coefficients, build_random_cluster
from fmm.fmm_kernels import (
    p2m_pass_group_jit,
    m2m_pass_group_jit,
    m2l_pass_group_jit,
    l2l_pass_group_jit,
    l2p_pass_group_jit,
    horizontal_pass_level_batch_jit,
    near_field_direct_pass_batch_jit,
)
from rcr_engine.wigner_rotation import FAC_TABLE

def profile_mode(fmm_sys, X, X_blocks, use_precomputed, runs=20):
    op = fmm_sys.fmm_op
    op.use_precomputed_passes = use_precomputed
    
    # Warmup
    fmm_sys.apply_system_operator(X)
    
    times = {
        "padding": 0.0,
        "upward": 0.0,
        "horizontal": 0.0,
        "downward": 0.0,
        "l2p": 0.0,
        "near": 0.0,
        "slicing": 0.0,
        "total": 0.0
    }
    
    max_order = max(op.p_levels[2:] + [op.p_fmm])
    fnr = op._cache.trans(max_order).fnr
    
    for _ in range(runs):
        t_start = time.perf_counter()
        
        # Padding
        t0 = time.perf_counter()
        X_padded = np.zeros((op.nsph, op.coeff_size, 1), dtype=np.complex128)
        for j, xj in enumerate(X_blocks):
            nblk_j = op.nodr_list[j] * (op.nodr_list[j] + 2)
            X_padded[j, :nblk_j, 0] = xj[:nblk_j]
            X_padded[j, op.nblk_half : op.nblk_half + nblk_j, 0] = xj[nblk_j:]
        times["padding"] += time.perf_counter() - t0
        
        # Multipole & local arrays
        multipole_coeffs = np.zeros((op.total_coeff_size, 1), dtype=np.complex128)
        local_coeffs = np.zeros((op.total_coeff_size, 1), dtype=np.complex128)
        
        # Upward Pass
        t0 = time.perf_counter()
        if use_precomputed:
            for group in op.p2m_cache:
                p2m_pass_group_jit(
                    group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                    group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                    group.phase_neg_flat, multipole_coeffs, X_padded, op.p_fmm, group.dc, group.tran_coef_arr,
                    group.has_d_cache, FAC_TABLE
                )
            max_level = len(op.tree.levels) - 1
            for level in range(max_level, 2, -1):
                for group in op.m2m_cache:
                    if group.level == level:
                        m2m_pass_group_jit(
                            group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                            group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                            group.phase_neg_flat, multipole_coeffs, group.dc, group.tran_coef_arr,
                            group.has_d_cache, FAC_TABLE
                        )
        else:
            for level in range(len(op.tree.levels) - 1, 1, -1):
                P_l = op.p_levels[level]
                coeff_size_l = 2 * P_l * (P_l + 2)
                for cell in op.tree.levels[level]:
                    cell_idx = op.cell_to_idx[id(cell)]
                    cell_offset = op.cell_offsets[cell_idx]
                    if not cell.children:
                        for idx in cell.sphere_indices:
                            tvec = cell.center - op.kpos[idx]
                            x_sphere_padded = op._pad_coefficients(X_padded[idx, :, 0], op.p_fmm, P_l)
                            translated = _rcr_translate(tvec, x_sphere_padded, P_l, op._cache, regular=True)
                            multipole_coeffs[cell_offset : cell_offset + coeff_size_l, 0] += translated
                if level > 2:
                    parent_level = level - 1
                    P_parent = op.p_levels[parent_level]
                    coeff_size_parent = 2 * P_parent * (P_parent + 2)
                    for parent_cell in op.tree.levels[parent_level]:
                        parent_idx = op.cell_to_idx[id(parent_cell)]
                        parent_offset = op.cell_offsets[parent_idx]
                        for child in parent_cell.children:
                            child_idx = op.cell_to_idx[id(child)]
                            child_offset = op.cell_offsets[child_idx]
                            child_size = 2 * op.p_levels[level] * (op.p_levels[level] + 2)
                            tvec = parent_cell.center - child.center
                            child_mp = multipole_coeffs[child_offset : child_offset + child_size, 0]
                            child_mp_padded = op._pad_coefficients(child_mp, op.p_levels[level], P_parent)
                            translated = _rcr_translate(tvec, child_mp_padded, P_parent, op._cache, regular=True)
                            multipole_coeffs[parent_offset : parent_offset + coeff_size_parent, 0] += translated
        times["upward"] += time.perf_counter() - t0
        
        # Horizontal Pass
        t0 = time.perf_counter()
        if use_precomputed:
            for level in range(2, len(op.tree.levels)):
                for group in op.m2l_cache:
                    if group.level == level:
                        m2l_pass_group_jit(
                            group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                            group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                            group.phase_neg_flat, multipole_coeffs, local_coeffs, group.dc, group.tran_coef_arr,
                            group.has_d_cache, FAC_TABLE
                        )
        else:
            for level in range(2, len(op.tree.levels)):
                P_l = op.p_levels[level]
                m2l_pairs = op.m2l_pairs_level[level]
                if m2l_pairs.shape[0] > 0:
                    tran_coef_arr = op.tran_coef_arr_by_order[P_l]
                    horizontal_pass_level_batch_jit(
                        m2l_pairs, multipole_coeffs, local_coeffs,
                        op.cell_offsets, op.cell_centers, P_l,
                        tran_coef_arr, FAC_TABLE, fnr
                    )
        times["horizontal"] += time.perf_counter() - t0
        
        # Downward Pass
        t0 = time.perf_counter()
        if use_precomputed:
            max_level = len(op.tree.levels) - 1
            for level in range(2, max_level):
                for group in op.l2l_cache:
                    if group.level == level:
                        l2l_pass_group_jit(
                            group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                            group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                            group.phase_neg_flat, local_coeffs, group.dc, group.tran_coef_arr,
                            group.has_d_cache, FAC_TABLE
                        )
        else:
            for level in range(2, len(op.tree.levels)):
                P_l = op.p_levels[level]
                child_level = level + 1
                if child_level < len(op.tree.levels):
                    P_child = op.p_levels[child_level]
                    coeff_size_child = 2 * P_child * (P_child + 2)
                    for parent_cell in op.tree.levels[level]:
                        parent_idx = op.cell_to_idx[id(parent_cell)]
                        parent_offset = op.cell_offsets[parent_idx]
                        for child in parent_cell.children:
                            child_idx = op.cell_to_idx[id(child)]
                            child_offset = op.cell_offsets[child_idx]
                            tvec = child.center - parent_cell.center
                            parent_local = local_coeffs[parent_offset : parent_offset + 2 * P_l * (P_l + 2), 0]
                            translated = _rcr_translate(tvec, parent_local, P_l, op._cache, regular=True)
                            local_child = op._slice_coefficients(translated, P_l, P_child)
                            local_coeffs[child_offset : child_offset + coeff_size_child, 0] += local_child
        times["downward"] += time.perf_counter() - t0
        
        # Leaf to Spheres (L2P)
        t0 = time.perf_counter()
        Y_far_padded = np.zeros((op.nsph, op.coeff_size, 1), dtype=np.complex128)
        if use_precomputed:
            for group in op.l2p_cache:
                l2p_pass_group_jit(
                    group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                    group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                    group.phase_neg_flat, local_coeffs, Y_far_padded, group.dc, group.tran_coef_arr,
                    group.has_d_cache, FAC_TABLE
                )
        else:
            for cell in op.tree.leaves:
                cell_idx = op.cell_to_idx[id(cell)]
                cell_offset = op.cell_offsets[cell_idx]
                P_l = op.p_levels[cell.level]
                coeff_size_l = 2 * P_l * (P_l + 2)
                for idx in cell.sphere_indices:
                    tvec = op.kpos[idx] - cell.center
                    local_cell = local_coeffs[cell_offset : cell_offset + coeff_size_l, 0]
                    translated = _rcr_translate(tvec, local_cell, P_l, op._cache, regular=True)
                    sliced = op._slice_coefficients(translated, P_l, op.p_fmm)
                    Y_far_padded[idx, :, 0] += sliced
        times["l2p"] += time.perf_counter() - t0
        
        # Near-field
        t0 = time.perf_counter()
        tran_coef_arr_fmm = op.tran_coef_arr_by_order[op.p_fmm]
        Y_near_padded = near_field_direct_pass_batch_jit(
            op.near_pairs, X_padded, op.kpos, op.p_fmm, op.nblk_half,
            FAC_TABLE, tran_coef_arr_fmm, fnr
        )
        times["near"] += time.perf_counter() - t0
        
        # Slicing
        t0 = time.perf_counter()
        Y_combined_padded = Y_near_padded + Y_far_padded
        Y_blocks_batch = []
        for i in range(op.nsph):
            nblk_i = op.nodr_list[i] * (op.nodr_list[i] + 2)
            y_i = np.empty(2 * nblk_i, dtype=np.complex128)
            y_i[:nblk_i] = Y_combined_padded[i, :nblk_i, 0]
            y_i[nblk_i:] = Y_combined_padded[i, op.nblk_half : op.nblk_half + nblk_i, 0]
            Y_blocks_batch.append(y_i)
        times["slicing"] += time.perf_counter() - t0
        
        times["total"] += time.perf_counter() - t_start
        
    return {k: v / runs for k, v in times.items()}

def profile_fmm():
    print("=========================================")
    print("Profiling FMM Operator Breakdown (N=512, p=3)")
    print("=========================================")
    N = 512
    p = 3
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm

    positions = build_random_cluster(N, radius=r_monomer_nm, seed=42)
    positions_dimless = positions * k_medium
    nodr_list = [p] * N
    an_list = [make_mie_coefficients(p)] * N

    # Instantiation
    fmm_sys = FMMMultisphereSystem(
        positions_dimless, nodr_list, an_list, k_medium=1.0,
        p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
        fmm_mode="speed", fmm_tol=1e-6, theta=0.55
    )

    N_total = fmm_sys.N_total
    rng = np.random.default_rng(42)
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
    
    # Split X into blocks
    X_blocks = [
        X[fmm_sys.offsets[j]: fmm_sys.offsets[j] + fmm_sys.nblk_full_list[j]]
        for j in range(fmm_sys.nsph)
    ]
    
    # Run profiles
    print("\nRunning legacy loop breakdown...")
    times_legacy = profile_mode(fmm_sys, X, X_blocks, use_precomputed=False, runs=20)
    
    print("\nRunning precomputed JIT breakdown...")
    times_precomputed = profile_mode(fmm_sys, X, X_blocks, use_precomputed=True, runs=20)
    
    print("\n--- Speedup Breakdown (Average of 20 runs) ---")
    print(f"{'Stage':<12} | {'Legacy':<10} | {'Precomputed':<11} | {'Speedup':<8}")
    print("-" * 50)
    for stage in ["padding", "upward", "horizontal", "downward", "l2p", "near", "slicing", "total"]:
        t_l = times_legacy[stage]
        t_p = times_precomputed[stage]
        speedup = t_l / t_p if t_p > 0 else 0.0
        print(f"{stage:<12} | {t_l:.4f}s     | {t_p:.4f}s      | {speedup:.2f}x")

if __name__ == "__main__":
    profile_fmm()
