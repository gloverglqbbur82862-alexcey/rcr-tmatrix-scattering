#!/usr/bin/env python3
"""Tests: Verify GPU direct solver backend correctness against CPU baselines."""

import numpy as np
import pytest

try:
    import cupy as cp
    cupy_available = True
    # Check if a CUDA GPU is actually available and usable
    try:
        cuda_available = cp.is_available()
    except Exception:
        cuda_available = False
except ImportError:
    cp = None
    cupy_available = False
    cuda_available = False

from solver.bicgstab_solver import MultisphereRCRSystem

def _make_mie_coefficients(p: int) -> np.ndarray:
    an = np.zeros((2, 2, p), dtype=np.complex128)
    for n in range(p):
        an[0, 0, n] = -0.5 + 0.1j * n
        an[1, 1, n] = -0.3 + 0.05j * n
    return an

@pytest.mark.skipif(not cuda_available, reason="CUDA/CuPy is not available on this system.")
def test_gpu_direct_operator():
    """Verify that apply_system_operator on GPU matches CPU serial version exactly."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])
    p = 3
    nodr_list = [p] * 4
    an_list = [_make_mie_coefficients(p)] * 4

    # Instantiate CPU baseline
    sys_cpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_cpu_serial"
    )

    # Instantiate GPU system
    sys_gpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_gpu"
    )

    # Generate a random test input vector X (single RHS)
    rng = np.random.default_rng(12345)
    N_total = sys_cpu.N_total
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)

    # Apply operator on CPU
    AX_cpu = sys_cpu.apply_system_operator(X)

    # Apply operator on GPU
    AX_gpu = sys_gpu.apply_system_operator(X)

    # Calculate relative L2 error
    diff = np.linalg.norm(AX_gpu - AX_cpu) / np.linalg.norm(AX_cpu)
    assert diff < 1e-12, f"Operator mismatch on single RHS. Error: {diff}"

    # Test multiple RHS (batched operator application)
    B = 5
    X_batch = rng.standard_normal((N_total, B)) + 1j * rng.standard_normal((N_total, B))
    
    # CPU sequential execution
    AX_cpu_batch = np.zeros_like(X_batch)
    for b in range(B):
        AX_cpu_batch[:, b] = sys_cpu.apply_system_operator(X_batch[:, b])

    # GPU batched execution
    AX_gpu_batch = sys_gpu.apply_system_operator(X_batch)

    # Compare columns
    for b in range(B):
        diff_col = np.linalg.norm(AX_gpu_batch[:, b] - AX_cpu_batch[:, b]) / np.linalg.norm(AX_cpu_batch[:, b])
        assert diff_col < 1e-12, f"Operator mismatch on column {b}. Error: {diff_col}"


@pytest.mark.skipif(not cuda_available, reason="CUDA/CuPy is not available on this system.")
def test_gpu_direct_solve():
    """Verify that solve on GPU matches CPU solve exactly for single and batched RHS."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])
    p = 3
    nodr_list = [p] * 4
    an_list = [_make_mie_coefficients(p)] * 4

    sys_cpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_cpu_serial"
    )
    sys_gpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_gpu"
    )

    rng = np.random.default_rng(54321)
    N_total = sys_cpu.N_total
    
    # 1. Solve single RHS
    b_vec = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
    
    x_cpu, _ = sys_cpu.solve(b_vec, tol=1e-12)
    x_gpu, _ = sys_gpu.solve(b_vec, tol=1e-12)

    diff = np.linalg.norm(x_gpu - x_cpu) / np.linalg.norm(x_cpu)
    assert diff < 1e-11, f"Solve mismatch on single RHS. Error: {diff}"

    # 2. Solve batched RHS
    B = 3
    b_batch = rng.standard_normal((N_total, B)) + 1j * rng.standard_normal((N_total, B))
    
    x_cpu_batch = np.zeros_like(b_batch)
    for b in range(B):
        x_cpu_batch[:, b], _ = sys_cpu.solve(b_batch[:, b], tol=1e-12)

    x_gpu_batch, _ = sys_gpu.solve(b_batch, tol=1e-12)

    for b in range(B):
        diff_col = np.linalg.norm(x_gpu_batch[:, b] - x_cpu_batch[:, b]) / np.linalg.norm(x_cpu_batch[:, b])
        assert diff_col < 1e-11, f"Solve mismatch on column {b}. Error: {diff_col}"


@pytest.mark.skipif(not cuda_available, reason="CUDA/CuPy is not available on this system.")
def test_multi_gpu_solve():
    """Verify that solve on direct_multi_gpu matches CPU solve exactly for batched RHS."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])
    p = 3
    nodr_list = [p] * 4
    an_list = [_make_mie_coefficients(p)] * 4

    sys_cpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_cpu_serial"
    )
    sys_gpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_multi_gpu"
    )

    rng = np.random.default_rng(99999)
    N_total = sys_cpu.N_total

    # Test batched RHS with 6 columns to verify partitioning across GPUs
    B = 6
    b_batch = rng.standard_normal((N_total, B)) + 1j * rng.standard_normal((N_total, B))
    
    x_cpu_batch = np.zeros_like(b_batch)
    for b in range(B):
        x_cpu_batch[:, b], _ = sys_cpu.solve(b_batch[:, b], tol=1e-12)

    # Solve on multi-GPU (it will partition and run on ThreadPool)
    x_gpu_batch, _ = sys_gpu.solve(b_batch, tol=1e-12)

    for b in range(B):
        diff_col = np.linalg.norm(x_gpu_batch[:, b] - x_cpu_batch[:, b]) / np.linalg.norm(x_cpu_batch[:, b])
        assert diff_col < 1e-11, f"Multi-GPU solve mismatch on column {b}. Error: {diff_col}"

    # Verify apply_system_operator batched matches CPU
    AX_cpu_batch = np.zeros_like(b_batch)
    for b in range(B):
        AX_cpu_batch[:, b] = sys_cpu.apply_system_operator(b_batch[:, b])

    AX_gpu_batch = sys_gpu.apply_system_operator(b_batch)

    for b in range(B):
        diff_col = np.linalg.norm(AX_gpu_batch[:, b] - AX_cpu_batch[:, b]) / np.linalg.norm(AX_cpu_batch[:, b])
        assert diff_col < 1e-12, f"Multi-GPU operator mismatch on column {b}. Error: {diff_col}"


@pytest.mark.skipif(not cuda_available, reason="CUDA/CuPy is not available on this system.")
def test_gpu_dense_operator_and_solve():
    """Verify that apply_system_operator and solve on direct_gpu_dense matches CPU serial version exactly."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])
    p = 3
    nodr_list = [p] * 4
    an_list = [_make_mie_coefficients(p)] * 4

    # Instantiate CPU baseline
    sys_cpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_cpu_serial"
    )

    # Instantiate GPU dense system
    sys_gpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_gpu_dense"
    )

    # 1. Verify operator matches on single RHS
    rng = np.random.default_rng(12345)
    N_total = sys_cpu.N_total
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)

    AX_cpu = sys_cpu.apply_system_operator(X)
    AX_gpu = sys_gpu.apply_system_operator(X)

    diff = np.linalg.norm(AX_gpu - AX_cpu) / np.linalg.norm(AX_cpu)
    assert diff < 1e-12, f"Dense operator mismatch. Error: {diff}"

    # 2. Verify solve matches on single RHS
    b_vec = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
    x_cpu, _ = sys_cpu.solve(b_vec, tol=1e-12)
    x_gpu, _ = sys_gpu.solve(b_vec, tol=1e-12)

    diff_solve = np.linalg.norm(x_gpu - x_cpu) / np.linalg.norm(x_cpu)
    assert diff_solve < 1e-11, f"Dense solve mismatch. Error: {diff_solve}"

    # 3. Verify solve matches on batched RHS (3 columns)
    B = 3
    b_batch = rng.standard_normal((N_total, B)) + 1j * rng.standard_normal((N_total, B))
    
    x_cpu_batch = np.zeros_like(b_batch)
    for b in range(B):
        x_cpu_batch[:, b], _ = sys_cpu.solve(b_batch[:, b], tol=1e-12)
        
    x_gpu_batch, _ = sys_gpu.solve(b_batch, tol=1e-12)
    for b in range(B):
        diff_col = np.linalg.norm(x_gpu_batch[:, b] - x_cpu_batch[:, b]) / np.linalg.norm(x_cpu_batch[:, b])
        assert diff_col < 1e-11, f"Dense solve mismatch on column {b}. Error: {diff_col}"


@pytest.mark.skipif(not cuda_available, reason="CUDA/CuPy is not available on this system.")
def test_gpu_dense_calculate_aggregate_tmatrix():
    """Verify that calculate_aggregate_tmatrix works on direct_gpu_dense and matches CPU serial."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])
    p = 3
    nodr_list = [p] * 4
    an_list = [_make_mie_coefficients(p)] * 4

    sys_cpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_cpu_serial"
    )
    sys_gpu = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_gpu_dense"
    )

    # Calculate aggregate T-matrix of order 3
    T_agg_cpu = sys_cpu.calculate_aggregate_tmatrix(aggregate_order=3, tol=1e-12)
    T_agg_gpu = sys_gpu.calculate_aggregate_tmatrix(aggregate_order=3, tol=1e-12)

    diff = np.linalg.norm(T_agg_gpu - T_agg_cpu) / np.linalg.norm(T_agg_cpu)
    assert diff < 1e-11, f"T_agg calculation mismatch. Error: {diff}"


@pytest.mark.skipif(not cuda_available, reason="CUDA/CuPy is not available on this system.")
def test_gpu_dense_memory_guard():
    """Verify that GPUDenseRCRSystem raises MemoryError on extremely large inputs."""
    k_medium = 1.0
    # Create 10,000 spheres to trigger memory guard (~1340 GB required for p=3)
    nsph = 10000
    positions = np.zeros((nsph, 3))
    positions[:, 0] = np.arange(nsph) * 10.0
    
    p = 3
    nodr_list = [p] * nsph
    an_list = [_make_mie_coefficients(p)] * nsph

    with pytest.raises(MemoryError, match="Required dense matrix size .* exceeds available"):
        MultisphereRCRSystem(
            positions, nodr_list, an_list, k_medium, backend="direct_gpu_dense"
        )


def test_precomputed_merge_correctness():
    """Verify that M_flat @ x_matrix matches column-by-column _merge_results_single_col_jit."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])
    p = 3
    nodr_list = [p] * 4
    an_list = [_make_mie_coefficients(p)] * 4

    sys = MultisphereRCRSystem(
        positions, nodr_list, an_list, k_medium, backend="direct_cpu_serial"
    )
    
    aggregate_order = 3
    nblk_half_agg = aggregate_order * (aggregate_order + 2)
    dim_agg = 2 * nblk_half_agg
    p_sphere = p
    
    from scipy.special import spherical_jn
    d_cache_size = sum((2*n + 1)**2 for n in range(1, aggregate_order + 1))
    sphere_jn_vals = np.zeros((sys.nsph, 2 * aggregate_order + 1), dtype=np.complex128)
    pair_phi_neg = np.zeros(sys.nsph, dtype=np.float64)
    positions_d_blocks_neg = np.zeros((sys.nsph, d_cache_size), dtype=np.float64)
    
    for i in range(sys.nsph):
        pos = sys.positions[i]
        r_val = float(np.linalg.norm(pos)) * sys.k_medium
        for w in range(2 * aggregate_order + 1):
            sphere_jn_vals[i, w] = spherical_jn(w, r_val)
            
        r_geom = float(np.linalg.norm(pos))
        if r_geom > 1e-14:
            theta_neg = float(np.arccos(np.clip(-pos[2] / r_geom, -1.0, 1.0)))
            phi_neg = float(np.arctan2(-pos[1], -pos[0]))
        else:
            theta_neg = 0.0
            phi_neg = 0.0
        pair_phi_neg[i] = phi_neg
        from solver.bicgstab_solver import _build_cached_d_jit, FAC_TABLE, _mstm_normalizedlegendre_zero_jit
        positions_d_blocks_neg[i] = _build_cached_d_jit(aggregate_order, theta_neg, FAC_TABLE)
        
    offsets_arr = np.array(sys.offsets, dtype=np.int64)
    nblk_full_arr = np.array(sys.nblk_full_list, dtype=np.int64)
    
    trans_prov = sys.trans_providers[aggregate_order]
    tran_coef_arr = trans_prov.tran_coef_arr
    dc = _mstm_normalizedlegendre_zero_jit(1.0, 2 * aggregate_order, trans_prov.fnr)
    
    B = 5
    rng = np.random.default_rng(98765)
    c_sphere = 2 * p_sphere * (p_sphere + 2)
    x_matrix = rng.standard_normal((sys.nsph * c_sphere, B)) + 1j * rng.standard_normal((sys.nsph * c_sphere, B))
    
    from solver.bicgstab_solver import _merge_results_single_col_jit, _precompute_merge_operator_flat_jit
    T_agg_old = np.zeros((dim_agg, B), dtype=np.complex128)
    for idx in range(B):
        x_vec = x_matrix[:, idx]
        T_agg_old[:, idx] = _merge_results_single_col_jit(
            sys.nsph, p_sphere, aggregate_order, x_vec,
            sphere_jn_vals, pair_phi_neg, positions_d_blocks_neg,
            offsets_arr, nblk_full_arr, dc, tran_coef_arr
        )
        
    M_flat = _precompute_merge_operator_flat_jit(
        sys.nsph, p_sphere, aggregate_order,
        sphere_jn_vals, pair_phi_neg, positions_d_blocks_neg,
        offsets_arr, dc, tran_coef_arr
    )
    T_agg_new = M_flat @ x_matrix
    
    diff = np.linalg.norm(T_agg_new - T_agg_old) / np.linalg.norm(T_agg_old)
    assert diff < 1e-12, f"Precomputed merge operator mismatch. Error: {diff}"

