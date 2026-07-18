#!/usr/bin/env python3
"""Pass-Level Profiling Script.

Benchmarks and breaks down the execution time of Direct JIT vs. precomputed
FMM (Speed and Reference) solvers for N = [128, 256, 512, 1024].
Outputs results in JSON and Markdown formats.
"""

import os
import sys
import time
import json
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solver.bicgstab_solver import MultisphereRCRSystem
from fmm.fmm_solver import FMMMultisphereSystem
from fmm.fmm_kernels import (
    p2m_pass_group_jit,
    m2m_pass_group_jit,
    m2l_pass_group_jit,
    l2l_pass_group_jit,
    l2p_pass_group_jit,
    near_field_pass_group_jit,
)
from rcr_engine.wigner_rotation import FAC_TABLE
from experiments.fmm_performance.benchmark import make_mie_coefficients, build_random_cluster


def profile_fmm_passes(fmm_sys, X_blocks, runs=20):
    """Profiles the individual passes of the FMM operator."""
    op = fmm_sys.fmm_op
    B = X_blocks[0].shape[1] if len(X_blocks) > 0 and X_blocks[0].ndim > 1 else 1
    nsph = op.nsph

    # Warmup runs to ensure Numba is compiled
    for _ in range(3):
        fmm_sys.apply_system_operator(np.concatenate(X_blocks))

    totals = {
        "allocation/zero-fill": 0.0,
        "input padding": 0.0,
        "P2M": 0.0,
        "M2M": 0.0,
        "M2L": 0.0,
        "L2L": 0.0,
        "L2P": 0.0,
        "near field": 0.0,
        "output assembly": 0.0
    }

    for _ in range(runs):
        # 1. Allocation / Zero-fill
        t0 = time.perf_counter()
        if (op._multipole_coeffs_cache is None or 
            op._multipole_coeffs_cache.shape[1] != B):
            op._X_padded_cache = np.zeros((nsph, op.coeff_size, B), dtype=np.complex128)
            op._multipole_coeffs_cache = np.zeros((op.total_coeff_size, B), dtype=np.complex128)
            op._local_coeffs_cache = np.zeros((op.total_coeff_size, B), dtype=np.complex128)
            op._Y_far_padded_cache = np.zeros((nsph, op.coeff_size, B), dtype=np.complex128)
        else:
            op._X_padded_cache.fill(0.0)
            op._multipole_coeffs_cache.fill(0.0)
            op._local_coeffs_cache.fill(0.0)
            op._Y_far_padded_cache.fill(0.0)
        totals["allocation/zero-fill"] += time.perf_counter() - t0

        # 2. Input Padding
        t0 = time.perf_counter()
        X_padded = op._X_padded_cache
        multipole_coeffs = op._multipole_coeffs_cache
        local_coeffs = op._local_coeffs_cache
        Y_far_padded = op._Y_far_padded_cache

        for j, xj in enumerate(X_blocks):
            nblk_j = op.nodr_list[j] * (op.nodr_list[j] + 2)
            if xj.ndim > 1:
                X_padded[j, :nblk_j, :] = xj[:nblk_j, :]
                X_padded[j, op.nblk_half : op.nblk_half + nblk_j, :] = xj[nblk_j:, :]
            else:
                X_padded[j, :nblk_j, 0] = xj[:nblk_j]
                X_padded[j, op.nblk_half : op.nblk_half + nblk_j, 0] = xj[nblk_j:]
        totals["input padding"] += time.perf_counter() - t0

        # 3. P2M
        t0 = time.perf_counter()
        for group in op.p2m_cache:
            p2m_pass_group_jit(
                group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                group.phase_neg_flat, multipole_coeffs, X_padded, op.p_fmm, group.dc, group.tran_coef_arr,
                group.has_d_cache, FAC_TABLE
            )
        totals["P2M"] += time.perf_counter() - t0

        # 4. M2M
        t0 = time.perf_counter()
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
        totals["M2M"] += time.perf_counter() - t0

        # 5. M2L
        t0 = time.perf_counter()
        for level in range(2, len(op.tree.levels)):
            for group in op.m2l_cache:
                if group.level == level:
                    m2l_pass_group_jit(
                        group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                        group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                        group.phase_neg_flat, multipole_coeffs, local_coeffs, group.dc, group.tran_coef_arr,
                        group.has_d_cache, FAC_TABLE
                    )
        totals["M2L"] += time.perf_counter() - t0

        # 6. L2L
        t0 = time.perf_counter()
        for level in range(2, max_level):
            for group in op.l2l_cache:
                if group.level == level:
                    l2l_pass_group_jit(
                        group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                        group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                        group.phase_neg_flat, local_coeffs, group.dc, group.tran_coef_arr,
                        group.has_d_cache, FAC_TABLE
                    )
        totals["L2L"] += time.perf_counter() - t0

        # 7. L2P
        t0 = time.perf_counter()
        for group in op.l2p_cache:
            l2p_pass_group_jit(
                group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                group.phase_neg_flat, local_coeffs, Y_far_padded, group.dc, group.tran_coef_arr,
                group.has_d_cache, FAC_TABLE
            )
        totals["L2P"] += time.perf_counter() - t0

        # 8. Near-field
        t0 = time.perf_counter()
        Y_near_padded = np.zeros((nsph, op.coeff_size, B), dtype=np.complex128)
        if len(op.near_cache) > 0:
            for group in op.near_cache:
                near_field_pass_group_jit(
                    group.edges_int, group.edges_geom, group.d_offsets, group.d_blocks_flat,
                    group.radial_offsets, group.radial_flat, group.phase_offsets, group.phase_pos_flat,
                    group.phase_neg_flat, X_padded, Y_near_padded, group.dc, group.tran_coef_arr,
                    group.has_d_cache, FAC_TABLE
                )
        totals["near field"] += time.perf_counter() - t0

        # 9. Output assembly / slicing
        t0 = time.perf_counter()
        Y_combined_padded = Y_near_padded + Y_far_padded
        Y_blocks_batch = []
        for i in range(nsph):
            nblk_i = op.nodr_list[i] * (op.nodr_list[i] + 2)
            y_i = np.empty((2 * nblk_i, B), dtype=np.complex128)
            y_i[:nblk_i, :] = Y_combined_padded[i, :nblk_i, :]
            y_i[nblk_i:, :] = Y_combined_padded[i, op.nblk_half : op.nblk_half + nblk_i, :]
            Y_blocks_batch.append(y_i)
        totals["output assembly"] += time.perf_counter() - t0

    # Calculate average times
    avg_passes = {k: v / runs for k, v in totals.items()}
    return avg_passes


def run_breakdown():
    N_list = [128, 256, 512, 1024]
    p = 3
    runs = 10
    
    r_monomer_nm = 25.0
    wavelength_nm = 550.0
    k_medium = 2.0 * np.pi / wavelength_nm

    print("=========================================================")
    print(f"Starting Pass-Level Timing Breakdown (p={p}, runs={runs})")
    print("=========================================================")

    # Global warm-up
    print("Performing JIT warm-up...")
    pos_warm = build_random_cluster(32, radius=r_monomer_nm, seed=42)
    pos_warm_dimless = pos_warm * k_medium
    nodr_warm = [p] * 32
    an_warm = [make_mie_coefficients(p)] * 32
    direct_sys = MultisphereRCRSystem(pos_warm_dimless, nodr_warm, an_warm, k_medium=1.0)
    fmm_sys = FMMMultisphereSystem(
        pos_warm_dimless, nodr_warm, an_warm, k_medium=1.0,
        p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
        fmm_mode="speed", fmm_tol=1e-6, theta=0.55,
        use_precomputed_passes=True, d_cache_budget_mb=500.0,
        verbose=False
    )
    X_warm = np.random.default_rng(42).standard_normal(direct_sys.N_total) + 1j * np.random.default_rng(42).standard_normal(direct_sys.N_total)
    direct_sys.apply_system_operator(X_warm)
    fmm_sys.apply_system_operator(X_warm)
    print("Warm-up complete.\n")

    breakdown_data = {}

    for N in N_list:
        print(f"Profiling N={N}...")
        breakdown_data[N] = {}
        
        # Build cluster
        positions = build_random_cluster(N, radius=r_monomer_nm, seed=42)
        positions_dimless = positions * k_medium
        nodr_list = [p] * N
        an_list = [make_mie_coefficients(p)] * N
        
        nblk_half = p * (p + 2)
        N_total = 2 * nblk_half * N
        rng = np.random.default_rng(42)
        X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
        X_blocks = [
            X[j * 2 * nblk_half : (j + 1) * 2 * nblk_half]
            for j in range(N)
        ]

        # 1. Profile Direct JIT
        print("  Profiling Direct JIT...")
        direct_sys = MultisphereRCRSystem(positions_dimless, nodr_list, an_list, k_medium=1.0)
        # warm up
        direct_sys.apply_system_operator(X)
        direct_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            direct_sys.apply_system_operator(X)
            direct_times.append(time.perf_counter() - t0)
        direct_avg = float(np.mean(direct_times))
        
        breakdown_data[N]["direct"] = {
            "Direct JIT kernel/matvec": direct_avg,
            "total": direct_avg
        }

        # 2. Profile FMM Speed
        print("  Profiling FMM Speed...")
        fmm_speed_sys = FMMMultisphereSystem(
            positions_dimless, nodr_list, an_list, k_medium=1.0,
            p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
            fmm_mode="speed", fmm_tol=1e-6, theta=0.55,
            use_precomputed_passes=True, d_cache_budget_mb=500.0,
            verbose=False
        )
        speed_passes = profile_fmm_passes(fmm_speed_sys, X_blocks, runs=runs)
        speed_passes["total"] = sum(speed_passes.values())
        breakdown_data[N]["fmm_speed"] = speed_passes

        # 3. Profile FMM Reference
        print("  Profiling FMM Reference...")
        fmm_ref_sys = FMMMultisphereSystem(
            positions_dimless, nodr_list, an_list, k_medium=1.0,
            p_fmm=p, max_per_leaf=4, max_level=8, use_fmm=True,
            fmm_mode="reference", fmm_tol=1e-12, theta=0.55,
            use_precomputed_passes=True, d_cache_budget_mb=500.0,
            verbose=False
        )
        ref_passes = profile_fmm_passes(fmm_ref_sys, X_blocks, runs=runs)
        ref_passes["total"] = sum(ref_passes.values())
        breakdown_data[N]["fmm_ref"] = ref_passes

    # Output to JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "pass_breakdown.json")
    with open(json_path, "w") as f:
        json.dump(breakdown_data, f, indent=4)
    print(f"JSON data saved to: {json_path}")

    # Generate Markdown Table
    md_lines = []
    md_lines.append("# Pass-Level Execution Time Breakdown (p=3)")
    md_lines.append("")
    md_lines.append("| N | Solver / Pass | Time (ms) | Fraction of Total |")
    md_lines.append("|---|---------------|-----------|-------------------|")

    for N in N_list:
        md_lines.append(f"| **{N}** | | | |")
        
        # Direct
        d_val = breakdown_data[N]["direct"]["Direct JIT kernel/matvec"] * 1000.0
        md_lines.append(f"| | Direct JIT kernel/matvec | {d_val:.3f} ms | 100.0% |")
        
        # FMM Speed
        s_total = breakdown_data[N]["fmm_speed"]["total"]
        md_lines.append(f"| | **FMM Speed (Total)** | **{s_total * 1000.0:.3f} ms** | 100.0% |")
        for k, v in breakdown_data[N]["fmm_speed"].items():
            if k != "total":
                frac = (v / s_total) * 100.0
                md_lines.append(f"| | &nbsp;&nbsp;&nbsp;&nbsp;- {k} | {v * 1000.0:.3f} ms | {frac:.1f}% |")

        # FMM Ref
        r_total = breakdown_data[N]["fmm_ref"]["total"]
        md_lines.append(f"| | **FMM Reference (Total)** | **{r_total * 1000.0:.3f} ms** | 100.0% |")
        for k, v in breakdown_data[N]["fmm_ref"].items():
            if k != "total":
                frac = (v / r_total) * 100.0
                md_lines.append(f"| | &nbsp;&nbsp;&nbsp;&nbsp;- {k} | {v * 1000.0:.3f} ms | {frac:.1f}% |")

    md_content = "\n".join(md_lines)
    print("\n" + md_content + "\n")

    md_path = os.path.join(out_dir, "pass_breakdown.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown table saved to: {md_path}")


if __name__ == "__main__":
    run_breakdown()
