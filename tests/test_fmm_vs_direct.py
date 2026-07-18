#!/usr/bin/env python3
"""Tests: Verify FMM Matvec against direct O(N^2) RCR solver.

Convention (same as validated tests in Phase 4):
  - All positions are in DIMENSIONLESS kR units.
  - k_medium = 1.0 (positions have been pre-multiplied by the physical k).
  - This matches MultisphereRCRSystem which passes raw positions directly
    as the separation argument to apply_coaxial_translation.

Three test levels:
  1. test_octree_* — sanity-check the octree structure.
  2. test_fmm_matvec_2sphere — 2-sphere exact Matvec comparison.
  3. test_fmm_matvec_8sphere — 8-sphere cluster Matvec comparison.
  4. test_fmm_vs_direct_n16 — Random N=16 cluster system-operator comparison.

All comparisons against the validated direct O(N^2) MultisphereRCRSystem.
Tolerance: rel_err <= 1e-12 (machine precision, same as Phase 4).
"""

import numpy as np
import pytest

from fmm.octree import Octree
from fmm.fmm_operator import FMMOperator
from fmm.fmm_solver import FMMMultisphereSystem
from solver.bicgstab_solver import MultisphereRCRSystem, apply_tmatrix_model1


# ================================================================== #
#  Shared physical parameters                                          #
# ================================================================== #

def _make_mie_coefficients(nodr: int) -> np.ndarray:
    """Build a simple diagonal Mie coefficient array for testing.

    Returns an (2, 2, nodr) array with TM=-0.5, TE=-0.3 for all orders.
    These are not physical but exercise all code paths.
    """
    an = np.zeros((2, 2, nodr), dtype=np.complex128)
    for n in range(nodr):
        an[0, 0, n] = -0.5 + 0.1j * n  # TM diagonal
        an[1, 1, n] = -0.3 + 0.05j * n  # TE diagonal
    return an


# ================================================================== #
#  Test 1: Octree structure                                           #
# ================================================================== #

def test_octree_build_2sphere():
    """Verify octree builds correctly for 2 spheres."""
    # Dimensionless kR positions
    positions = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    tree = Octree(positions, max_per_leaf=1)

    print(tree.summary())

    # All spheres should be distributed to leaves
    leaf_sphere_count = sum(len(leaf.sphere_indices) for leaf in tree.leaves)
    assert leaf_sphere_count == 2, f"Expected 2 spheres in leaves, got {leaf_sphere_count}"
    assert len(tree.levels) >= 2


def test_octree_build_random_n16():
    """Verify octree correctly partitions 16 random spheres."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 30, size=(16, 3))  # dimensionless kR coords
    tree = Octree(positions, max_per_leaf=3)

    print(tree.summary())

    leaf_idx_set = []
    for leaf in tree.leaves:
        leaf_idx_set.extend(leaf.sphere_indices)

    assert len(leaf_idx_set) == 16, f"Expected 16 total, got {len(leaf_idx_set)}"
    assert set(leaf_idx_set) == set(range(16))


def test_octree_interaction_list_no_overlap():
    """Near-field and interaction-list cells must not overlap."""
    rng = np.random.default_rng(7)
    positions = rng.uniform(0, 40, size=(12, 3))
    tree = Octree(positions, max_per_leaf=2, max_level=5)

    print(tree.summary())

    for lv_idx, lv_cells in enumerate(tree.levels):
        if lv_idx < 2:
            continue
        for cell in lv_cells:
            adj_set = {id(c) for c in cell.near_field}
            adj_set.add(id(cell))
            il_set = {id(c) for c in cell.interaction_list}
            overlap = adj_set & il_set
            assert len(overlap) == 0, (
                f"Level {lv_idx}: near_field and interaction_list overlap"
            )


# ================================================================== #
#  Test 2: FMM Matvec vs Direct — 2-sphere                           #
# ================================================================== #

def test_fmm_matvec_vs_direct_2sphere():
    """Compare FMM M-operator with direct O(N^2) sum for a 2-sphere system.

    Positions in dimensionless kR units; k_medium=1.0.
    """
    k_medium = 1.0
    # kR positions: separation of 8 (dimensionless)
    positions = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    nodr_list = [5, 5]
    an_list = [_make_mie_coefficients(5), _make_mie_coefficients(5)]

    direct_sys = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    fmm_sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=1, max_level=4, use_fmm=True
    )

    print(fmm_sys.tree_summary())
    print("Pair stats:", fmm_sys.fmm_op.pair_stats())

    N_total = direct_sys.N_total
    rng = np.random.default_rng(123)
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)

    AX_direct = direct_sys.apply_system_operator(X)
    AX_fmm = fmm_sys.apply_system_operator(X)

    rel_err = np.linalg.norm(AX_fmm - AX_direct) / np.linalg.norm(AX_direct)
    print(f"2-sphere FMM vs Direct rel_err = {rel_err:.4e}")

    assert rel_err < 1e-12, (
        f"FMM 2-sphere Matvec rel_err {rel_err:.2e} exceeds 1e-12"
    )


# ================================================================== #
#  Test 3: FMM Matvec vs Direct — 8-sphere random cluster            #
# ================================================================== #

def test_fmm_matvec_vs_direct_8sphere():
    """Compare FMM M-operator with direct sum for an 8-sphere cluster.

    All positions in dimensionless kR units.
    """
    k_medium = 1.0
    rng = np.random.default_rng(42)
    # kR positions: random in a 30x30x30 box
    positions = rng.uniform(0, 30, size=(8, 3))
    nodr_list = [4] * 8
    an_list = [_make_mie_coefficients(4)] * 8

    direct_sys = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    fmm_sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=2, max_level=5, use_fmm=True
    )

    print(fmm_sys.tree_summary())
    print("Pair stats:", fmm_sys.fmm_op.pair_stats())

    N_total = direct_sys.N_total
    rng2 = np.random.default_rng(77)
    X = rng2.standard_normal(N_total) + 1j * rng2.standard_normal(N_total)

    AX_direct = direct_sys.apply_system_operator(X)
    AX_fmm = fmm_sys.apply_system_operator(X)

    rel_err = np.linalg.norm(AX_fmm - AX_direct) / np.linalg.norm(AX_direct)
    print(f"8-sphere FMM vs Direct rel_err = {rel_err:.4e}")

    assert rel_err < 1e-12, (
        f"FMM 8-sphere Matvec rel_err {rel_err:.2e} exceeds 1e-12"
    )


# ================================================================== #
#  Test 4: FMM Matvec vs Direct — N=16 random                        #
# ================================================================== #

def test_fmm_matvec_vs_direct_n16():
    """Compare FMM vs Direct for N=16 randomly distributed spheres.

    All positions in dimensionless kR units.
    """
    k_medium = 1.0
    rng = np.random.default_rng(99)
    positions = rng.uniform(0, 50, size=(16, 3))
    nodr_list = [3] * 16
    an_list = [_make_mie_coefficients(3)] * 16

    direct_sys = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    fmm_sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=3, max_level=6, use_fmm=True
    )

    print(fmm_sys.tree_summary())
    print("Pair stats:", fmm_sys.fmm_op.pair_stats())

    N_total = direct_sys.N_total
    rng2 = np.random.default_rng(55)
    X = rng2.standard_normal(N_total) + 1j * rng2.standard_normal(N_total)

    AX_direct = direct_sys.apply_system_operator(X)
    AX_fmm = fmm_sys.apply_system_operator(X)

    rel_err = np.linalg.norm(AX_fmm - AX_direct) / np.linalg.norm(AX_direct)
    print(f"N=16 FMM vs Direct rel_err = {rel_err:.4e}")

    assert rel_err < 1e-12, (
        f"FMM N=16 Matvec rel_err {rel_err:.2e} exceeds 1e-12"
    )


# ================================================================== #
#  Test 5: FMM disabled == direct solver                              #
# ================================================================== #

def test_fmm_disabled_equals_direct():
    """When use_fmm=False, FMMMultisphereSystem should exactly match direct solver."""
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
    ])
    nodr_list = [4, 4, 4]
    an_list = [_make_mie_coefficients(4)] * 3

    direct_sys = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    fmm_fallback = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        use_fmm=False
    )

    N_total = direct_sys.N_total
    rng = np.random.default_rng(321)
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)

    AX_direct = direct_sys.apply_system_operator(X)
    AX_fmm_off = fmm_fallback.apply_system_operator(X)

    rel_err = np.linalg.norm(AX_fmm_off - AX_direct) / np.linalg.norm(AX_direct)
    print(f"FMM disabled vs direct rel_err = {rel_err:.4e}")

    assert rel_err < 1e-14, (
        f"FMM disabled mode doesn't match direct solver: rel_err={rel_err:.2e}"
    )


# ================================================================== #
#  Test 6: Pair completeness                                          #
# ================================================================== #

def test_all_pairs_classified():
    """Every sphere pair should be classified (near or far) — none missed."""
    k_medium = 1.0
    rng = np.random.default_rng(13)
    positions = rng.uniform(0, 40, size=(12, 3))
    nodr_list = [3] * 12
    an_list = [_make_mie_coefficients(3)] * 12

    fmm_sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=3, max_level=5, use_fmm=True
    )

    stats = fmm_sys.fmm_op.pair_stats()
    print("Pair stats:", stats)

    assert stats["all_classified"], (
        f"Not all pairs classified: {stats}"
    )
    assert stats["N_classified"] == stats["N_pairs_total"], (
        f"Missing {stats['N_pairs_total'] - stats['N_classified']} pairs"
    )


def test_fmm_vs_direct_convergence_n32_n64():
    """Verify convergence parity for N=32 and N=64 systems.
    Asserts that the relative L2 error of the FMM system operator compared to the direct operator
    is strictly < 1e-11 when using fmm_tol=1e-12.
    """
    k_medium = 1.0
    
    for N in [32, 64]:
        # Generate random cluster
        rng = np.random.default_rng(N * 7)
        # Positions in a 40x40x40 box to ensure a mix of near and far cells
        positions = rng.uniform(0, 40, size=(N, 3))
        
        # Truncation order p=3 for all spheres
        p_order = 3
        nodr_list = [p_order] * N
        an_list = [_make_mie_coefficients(p_order)] * N
        
        direct_sys = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
        fmm_sys = FMMMultisphereSystem(
            positions, nodr_list, an_list, k_medium,
            max_per_leaf=4, max_level=6, use_fmm=True,
            fmm_mode="reference", fmm_tol=1e-12, theta=0.55
        )
        
        N_total = direct_sys.N_total
        X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)
        
        AX_direct = direct_sys.apply_system_operator(X)
        AX_fmm = fmm_sys.apply_system_operator(X)
        
        rel_err = np.linalg.norm(AX_fmm - AX_direct) / np.linalg.norm(AX_direct)
        print(f"Convergence test N={N}, fmm_tol=1e-12, rel_err = {rel_err:.4e}")
        
        assert rel_err < 1e-11, (
            f"FMM Matvec for N={N} rel_err {rel_err:.2e} exceeds 1e-11"
        )


def test_fmm_speed_mode_accuracy_and_orders():
    """Verify that speed mode behaves correctly, dynamic cell orders are small,
    and relative error satisfies the accuracy bounds (e.g. < 1e-5).
    """
    k_medium = 1.0
    N = 32
    rng = np.random.default_rng(12345)
    positions = rng.uniform(0, 40, size=(N, 3))
    p_order = 3
    nodr_list = [p_order] * N
    an_list = [_make_mie_coefficients(p_order)] * N

    direct_sys = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    
    # 1. Instantiate under speed mode (fmm_tol=1e-6)
    fmm_sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=4, max_level=6, use_fmm=True,
        fmm_mode="speed", fmm_tol=1e-6, theta=0.55
    )

    # Check cell order bounds
    p_levels = fmm_sys.fmm_op.p_levels
    print("p_levels in speed mode:", p_levels)
    # The cell orders at level >= 2 should be in the target 8-20 range
    for l in range(2, len(p_levels)):
        assert 8 <= p_levels[l] <= 20, f"Expected speed mode cell order at level {l} to be between 8 and 20, got {p_levels[l]}"

    # Check accuracy
    N_total = direct_sys.N_total
    X = rng.standard_normal(N_total) + 1j * rng.standard_normal(N_total)

    AX_direct = direct_sys.apply_system_operator(X)
    AX_fmm = fmm_sys.apply_system_operator(X)

    rel_err = np.linalg.norm(AX_fmm - AX_direct) / np.linalg.norm(AX_direct)
    print(f"Speed mode rel_err = {rel_err:.4e}")

    assert rel_err < 1e-5, f"Speed mode rel_err {rel_err:.2e} exceeds 1e-5"


def test_fmm_auto_mode_switching():
    """Verify that auto mode correctly disables FMM by default (falling back to direct solver)."""
    k_medium = 1.0
    rng = np.random.default_rng(999)
    p_order = 3

    # Under auto mode, FMM is disabled by default to safeguard baseline speed
    positions = rng.uniform(0, 40, size=(32, 3))
    sys_auto = FMMMultisphereSystem(
        positions, [p_order]*32, [_make_mie_coefficients(p_order)]*32, k_medium,
        use_fmm=True, fmm_mode="auto", N_switch=32
    )
    assert sys_auto.use_fmm is False
    assert sys_auto._direct_system is not None


def test_fmm_batched_solve():
    """Verify that batched RHS FMM solve yields identical results to sequential column-by-column FMM solve."""
    k_medium = 1.0
    rng = np.random.default_rng(2468)
    N = 8
    positions = rng.uniform(0, 30, size=(N, 3))
    p_order = 3
    nodr_list = [p_order] * N
    an_list = [_make_mie_coefficients(p_order)] * N

    fmm_sys = FMMMultisphereSystem(
        positions, nodr_list, an_list, k_medium,
        max_per_leaf=2, max_level=4, use_fmm=True,
        fmm_mode="reference", fmm_tol=1e-12
    )

    N_total = fmm_sys.N_total
    B_cols = 3
    # Create 3 independent RHS vectors
    B_mat = rng.standard_normal((N_total, B_cols)) + 1j * rng.standard_normal((N_total, B_cols))

    # 1. Batched solve
    X_batch, res_batch = fmm_sys.solve_batch(B_mat, tol=1e-12, max_iter=200)

    # 2. Sequential solve
    X_seq = np.zeros_like(B_mat)
    res_seq = []
    for col in range(B_cols):
        x_col, r_col = fmm_sys.solve(B_mat[:, col], tol=1e-12, max_iter=200)
        X_seq[:, col] = x_col
        res_seq.append(r_col)

    # Compare results
    max_diff = np.max(np.abs(X_batch - X_seq))
    print(f"Batched vs Sequential solve max difference: {max_diff:.4e}")
    assert max_diff < 1e-11, f"Batched solve differs from sequential solve by {max_diff:.2e}"


# ================================================================== #
#  Test 8: Precomputed Cache Initialization                          #
# ================================================================== #

def test_precomputed_cache_initialization():
    """Verify that precomputed cache structures are initialized, grouped,
    and memory-budgeted correctly for the four passes.
    """
    k_medium = 1.0
    positions = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
    ])
    nodr_list = [3, 3, 3]
    
    # Instantiate FMMOperator directly
    op = FMMOperator(
        positions, nodr_list, k_medium,
        max_per_leaf=1, max_level=4,
        use_precomputed_passes=True,
        d_cache_budget_mb=100.0
    )
    
    # Check that caches are lists of PassGroupCache objects
    for name, cache in [("p2m", op.p2m_cache), ("m2m", op.m2m_cache), ("l2l", op.l2l_cache), ("l2p", op.l2p_cache)]:
        assert isinstance(cache, list), f"{name} cache is not a list"
        for group in cache:
            assert isinstance(group.edges_int, np.ndarray)
            assert isinstance(group.edges_geom, np.ndarray)
            assert isinstance(group.d_offsets, np.ndarray)
            assert isinstance(group.d_blocks_flat, np.ndarray)
            assert isinstance(group.radial_offsets, np.ndarray)
            assert isinstance(group.radial_flat, np.ndarray)
            assert isinstance(group.phase_offsets, np.ndarray)
            assert isinstance(group.phase_pos_flat, np.ndarray)
            assert isinstance(group.phase_neg_flat, np.ndarray)
            
            num_edges = group.edges_int.shape[0]
            work_order = group.work_order
            
            # Check array shapes/sizes consistency
            assert group.edges_int.shape[1] == 8
            assert group.edges_geom.shape == (num_edges, 3)
            assert group.phase_offsets.shape == (num_edges + 1,)
            assert group.radial_offsets.shape == (num_edges + 1,)
            assert group.phase_pos_flat.shape == (num_edges * (2 * work_order + 1),)
            assert group.phase_neg_flat.shape == (num_edges * (2 * work_order + 1),)
            assert group.radial_flat.shape == (num_edges * (2 * work_order + 1),)
            
            # If d_cache is enabled within budget, verify blocks shape, else empty
            block_sum = sum((2 * n + 1) ** 2 for n in range(1, work_order + 1))
            assert group.d_offsets.shape == (num_edges + 1,)
            if group.has_d_cache:
                assert group.d_blocks_flat.shape == (num_edges * block_sum,)
            else:
                assert group.d_blocks_flat.shape == (0,)

    # Test with 0.0 MB budget to force fallback (has_d_cache = False)
    op_no_d = FMMOperator(
        positions, nodr_list, k_medium,
        max_per_leaf=1, max_level=4,
        use_precomputed_passes=True,
        d_cache_budget_mb=0.0
    )
    for cache in [op_no_d.p2m_cache, op_no_d.m2m_cache, op_no_d.l2l_cache, op_no_d.l2p_cache]:
        for group in cache:
            assert group.has_d_cache is False
            assert group.d_blocks_flat.shape == (0,)


def test_fmm_precomputed_passes_against_legacy():
    """Verify that JIT precomputed passes match legacy python loop passes to machine precision."""
    k_medium = 1.0
    N = 32
    rng = np.random.default_rng(12345)
    
    # Generate random N=32 cluster
    positions = rng.uniform(0, 40, size=(N, 3))
    
    # Random orders p in [1, 5]
    nodr_list = rng.integers(1, 6, size=N).tolist()
    
    # Instantiate two FMMOperators on the same geometry
    op_legacy = FMMOperator(
        positions, nodr_list, k_medium,
        max_per_leaf=4, max_level=6,
        use_precomputed_passes=False
    )
    
    op_precomputed = FMMOperator(
        positions, nodr_list, k_medium,
        max_per_leaf=4, max_level=6,
        use_precomputed_passes=True
    )
    
    # Apply a random coefficient vector X to both and obtain Y_legacy and Y_precomputed
    X_blocks = []
    for p in nodr_list:
        nblk = p * (p + 2)
        X_blocks.append(
            rng.standard_normal(2 * nblk) + 1j * rng.standard_normal(2 * nblk)
        )
        
    Y_legacy_blocks = op_legacy.apply(X_blocks)
    Y_precomputed_blocks = op_precomputed.apply(X_blocks)
    
    Y_legacy = np.concatenate(Y_legacy_blocks)
    Y_precomputed = np.concatenate(Y_precomputed_blocks)
    
    # Assert identical to machine precision (relative difference < 10^-12)
    rel_diff = np.linalg.norm(Y_precomputed - Y_legacy) / np.linalg.norm(Y_legacy)
    print(f"Precomputed vs Legacy rel_diff: {rel_diff:.4e}")
    assert rel_diff < 1e-12, f"Relative difference {rel_diff:.2e} exceeds 1e-12"
    
    # Toggle use_precomputed_passes = False dynamically and verify it matches
    op_precomputed.use_precomputed_passes = False
    Y_toggled_blocks = op_precomputed.apply(X_blocks)
    Y_toggled = np.concatenate(Y_toggled_blocks)
    
    rel_diff_toggled = np.linalg.norm(Y_toggled - Y_legacy) / np.linalg.norm(Y_legacy)
    print(f"Toggled Precomputed vs Legacy rel_diff: {rel_diff_toggled:.4e}")
    assert rel_diff_toggled < 1e-12, f"Toggled relative difference {rel_diff_toggled:.2e} exceeds 1e-12"


