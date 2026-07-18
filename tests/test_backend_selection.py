#!/usr/bin/env python3
"""Tests: Verify explicit backend selection and correctness."""

import numpy as np
import pytest
import warnings

from fmm.fmm_solver import FMMMultisphereSystem
from solver.bicgstab_solver import MultisphereRCRSystem

def _make_mie_coefficients(p: int) -> np.ndarray:
    an = np.zeros((2, 2, p), dtype=np.complex128)
    for n in range(p):
        an[0, 0, n] = -0.5 + 0.1j * n
        an[1, 1, n] = -0.3 + 0.05j * n
    return an

def test_backend_selection_cpu():
    """Verify that all CPU backends initialize, run, and produce identical results."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0]
    ])
    nodr_list = [3, 3, 3]
    an_list = [_make_mie_coefficients(3)] * 3
    
    # 1. direct_cpu_serial
    sys_serial = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        backend="direct_cpu_serial"
    )
    assert sys_serial.use_fmm is False
    assert sys_serial._direct_system.backend == "direct_cpu_serial"
    
    # 2. direct_cpu_parallel
    sys_parallel = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        backend="direct_cpu_parallel"
    )
    assert sys_parallel.use_fmm is False
    assert sys_parallel._direct_system.backend == "direct_cpu_parallel"
    
    # 3. fmm_cpu_serial
    sys_fmm = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        backend="fmm_cpu_serial", fmm_mode="speed", fmm_tol=1e-6
    )
    assert sys_fmm.use_fmm is True
    assert sys_fmm.resolved_backend == "fmm_cpu_serial"
    
    # 4. auto
    sys_auto = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        backend="auto", fmm_mode="auto", fmm_tol=1e-6, N_switch=512
    )
    # Since N=3 < 512, auto should resolve to direct_gpu if CUDA is available, else direct_cpu_parallel
    cuda_available = False
    try:
        import cupy as cp
        cuda_available = cp.is_available()
    except Exception:
        pass
    if not cuda_available:
        try:
            from numba import cuda
            cuda_available = cuda.is_available()
        except Exception:
            pass
    expected_auto = "direct_gpu" if cuda_available else "direct_cpu_parallel"
    assert sys_auto.use_fmm is False
    assert sys_auto.resolved_backend == expected_auto
    
    # Compare outputs on a random input
    N_total = sys_serial.N_total
    rng = np.random.default_rng(1234)
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
    
    AX_serial = sys_serial.apply_system_operator(X)
    AX_parallel = sys_parallel.apply_system_operator(X)
    AX_fmm = sys_fmm.apply_system_operator(X)
    AX_auto = sys_auto.apply_system_operator(X)
    
    # Serial vs Parallel should be identical to machine precision
    diff_sp = np.linalg.norm(AX_parallel - AX_serial) / np.linalg.norm(AX_serial)
    assert diff_sp < 1e-14
    
    # Auto vs Parallel should be identical
    diff_ap = np.linalg.norm(AX_auto - AX_parallel) / np.linalg.norm(AX_parallel)
    assert diff_ap < 1e-14
    
    # FMM vs Direct should be identical within FMM tolerance
    diff_fmm = np.linalg.norm(AX_fmm - AX_serial) / np.linalg.norm(AX_serial)
    assert diff_fmm < 1e-5

def test_backend_selection_unimplemented_and_errors():
    """Verify handling of unimplemented backends and missing CUDA error cases."""
    k_medium = 1.0
    positions = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    nodr_list = [3, 3]
    an_list = [_make_mie_coefficients(3)] * 2
    
    # fmm_cpu_parallel should raise a warning and fall back to fmm_cpu_serial
    with pytest.warns(UserWarning, match="fmm_cpu_parallel is not yet implemented"):
        sys_fmm_parallel = FMMMultisphereSystem(
            positions, nodr_list, an_list, k_medium,
            backend="fmm_cpu_parallel"
        )
        assert sys_fmm_parallel.resolved_backend == "fmm_cpu_serial"
        assert sys_fmm_parallel.use_fmm is True

    # If CUDA is unavailable, requesting GPU backends should raise ValueError
    cuda_available = False
    try:
        import cupy as cp
        cuda_available = cp.is_available()
    except Exception:
        pass
    if not cuda_available:
        try:
            from numba import cuda
            cuda_available = cuda.is_available()
        except Exception:
            pass
        
    if not cuda_available:
        with pytest.raises(ValueError, match="CUDA is not available on this system"):
            FMMMultisphereSystem(
                positions, nodr_list, an_list, k_medium,
                backend="direct_gpu"
            )
            
        with pytest.raises(ValueError, match="CUDA is not available on this system"):
            FMMMultisphereSystem(
                positions, nodr_list, an_list, k_medium,
                backend="direct_multi_gpu"
            )
            
        with pytest.raises(ValueError, match="CUDA is not available on this system"):
            FMMMultisphereSystem(
                positions, nodr_list, an_list, k_medium,
                backend="fmm_gpu"
            )


def test_cpu_direct_batch_rhs():
    """Verify that FMMMultisphereSystem solve_batch works for CPU direct backends with 2D RHS."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0]
    ])
    nodr_list = [3, 3]
    an_list = [_make_mie_coefficients(3)] * 2
    
    # Instantiate with direct_cpu_serial
    sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        backend="direct_cpu_serial"
    )
    
    # 2D batch of RHS (3 columns)
    B_cols = 3
    rng = np.random.default_rng(42)
    B_mat = rng.standard_normal((sys.N_total, B_cols)) + 1j * rng.standard_normal((sys.N_total, B_cols))
    
    # Run batched solve (which delegates to direct CPU column-by-column loop)
    X, res_norms = sys.solve_batch(B_mat, tol=1e-12)
    
    assert X.shape == (sys.N_total, B_cols)
    assert len(res_norms) == B_cols
    assert np.all(res_norms < 1e-11)
    
    # Run solve with 2D array to verify it correctly delegates to solve_batch
    X_solve, res_solve = sys.solve(B_mat, tol=1e-12)
    assert X_solve.shape == (sys.N_total, B_cols)
    assert np.linalg.norm(X_solve - X) / np.linalg.norm(X) < 1e-12
    
    # Verify against sequential manual column solve
    for col in range(B_cols):
        x_col, res_col = sys.solve(B_mat[:, col], tol=1e-12)
        assert np.linalg.norm(X[:, col] - x_col) / np.linalg.norm(x_col) < 1e-12


def test_aggregate_order_boundary_validation():
    """Verify that calculate_aggregate_tmatrix raises ValueError if aggregate_order < monomer_order."""
    k_medium = 1.0
    positions = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    nodr_list = [3, 3]
    an_list = [_make_mie_coefficients(3)] * 2
    
    sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        backend="direct_cpu_serial"
    )
    
    with pytest.raises(ValueError, match="aggregate_order .* must be greater than or equal to monomer order"):
        sys.calculate_aggregate_tmatrix(aggregate_order=2)
