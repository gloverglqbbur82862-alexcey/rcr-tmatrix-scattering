#!/usr/bin/env python3
"""Tests for Exact Core Solver: verifying MultisphereRCRSystem against dense reference solver."""

import sys
import numpy as np
import pytest

# Add reference paths
sys.path.append("/Users/kaer/doc/fast_mtp_scattering/experiments/exp19/code")
try:
    from cp13c_r5_mstm_compatible_dense_solver import (
        solve_multisphere,
        _precompute_fnr,
        _precompute_bcof,
        _tran_coef_table,
        model1_to_model2_permutation,
        load_mie_coefficients,
        reproduce_incident_coefficients
    )
    import os
    DUMP_DIR = "/Users/kaer/doc/fast_mtp_scattering/experiments/exp19/mstm_debug"
    HAS_REFERENCE = os.path.exists(DUMP_DIR)
except ImportError:
    HAS_REFERENCE = False

from solver.bicgstab_solver import MultisphereRCRSystem
from tests.test_rcr_translation import vector_model1_to_model2, vector_model2_to_model1


@pytest.mark.skipif(not HAS_REFERENCE, reason="MSTM reference dumps not found")
def test_two_sphere_exact_solve():
    """Verify that the Matrix-Free RCR solver matches the direct dense MSTM-compatible solver
    to double precision limits across a 2-sphere scattering system.
    """
    # ── 1. Setup Physical System & load Mie Data ──────────────────────────────
    k_medium = 1.0
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([8.0, 0.0, 0.0])
    positions = np.array([pos1, pos2])
    
    mie_data = load_mie_coefficients(os.path.join(DUMP_DIR, 'two_sphere_mie_coefficients.txt'))
    nodr1 = mie_data[1]['nodr']
    nodr2 = mie_data[2]['nodr']
    an1 = mie_data[1]['an']
    an2 = mie_data[2]['an']
    
    nodr_list = [nodr1, nodr2]
    an_list = [an1, an2]
    
    # Precompute MSTM incident coefficients (model-2)
    pmnp0_1 = reproduce_incident_coefficients(nodr1, alpha=0.0, beta=0.0)
    pmnp0_2 = reproduce_incident_coefficients(nodr2, alpha=0.0, beta=0.0)
    
    # ── 2. Run Direct Dense Reference Solver (model-2) ────────────────────────
    # Shape: list of (nblk_full_i, 2)
    amnp_ref_m2 = solve_multisphere(positions, nodr_list, an_list, [pmnp0_1, pmnp0_2], k_medium)
    
    # ── 3. Run RCR Matrix-Free Solver (model-1) ───────────────────────────────
    # We convert incident coefficients to model-1 to feed into our RCR solver
    pmnp0_1_m1 = vector_model2_to_model1(pmnp0_1[:, 0], nodr1)
    pmnp0_2_m1 = vector_model2_to_model1(pmnp0_2[:, 0], nodr2)
    
    # To test polarization 2 as well:
    pmnp0_1_m1_p2 = vector_model2_to_model1(pmnp0_1[:, 1], nodr1)
    pmnp0_2_m1_p2 = vector_model2_to_model1(pmnp0_2[:, 1], nodr2)
    
    # Initialize Multisphere RCR System
    rcr_system = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    
    from solver.bicgstab_solver import apply_tmatrix_model1
    
    # We solve both polarization columns
    for pol_idx, (p1, p2) in enumerate([(pmnp0_1_m1, pmnp0_2_m1), (pmnp0_1_m1_p2, pmnp0_2_m1_p2)]):
        # Concatenate RHS vector b in model-1
        b_m1 = np.concatenate([
            # b_i = T_i * p0_i
            # T_i * p0_i in model-1:
            apply_tmatrix_model1(p1, an1, nodr1),
            apply_tmatrix_model1(p2, an2, nodr2)
        ])
        
        # Solve the system using BiCGSTAB
        x_m1, res = rcr_system.solve(b_m1, tol=1e-12, max_iter=200)
        
        print(f"Polarization {pol_idx + 1} solved. Relative residual: {res:.6e}")
        assert res < 1e-11, f"Krylov solver failed to converge below tolerance for pol={pol_idx+1}"
        
        # Split solved x_m1 back into per-sphere blocks
        off_1 = 0
        nb_1 = 2 * nodr1 * (nodr1 + 2)
        x1_m1 = x_m1[off_1 : off_1 + nb_1]
        x2_m1 = x_m1[off_1 + nb_1 :]
        
        # Convert back to model-2 to compare with reference
        x1_m2 = vector_model1_to_model2(x1_m1, nodr1)
        x2_m2 = vector_model1_to_model2(x2_m1, nodr2)
        
        # Compare sphere 1
        ref_x1 = amnp_ref_m2[0][:, pol_idx]
        diff_1 = np.linalg.norm(x1_m2 - ref_x1)
        rel_err_1 = diff_1 / np.linalg.norm(ref_x1)
        print(f"  Sphere 1 | rel_err={rel_err_1:.6e}")
        assert rel_err_1 < 1.0e-11, f"Sphere 1 mismatch for pol={pol_idx+1} with rel_err={rel_err_1:.2e}"
        
        # Compare sphere 2
        ref_x2 = amnp_ref_m2[1][:, pol_idx]
        diff_2 = np.linalg.norm(x2_m2 - ref_x2)
        rel_err_2 = diff_2 / np.linalg.norm(ref_x2)
        print(f"  Sphere 2 | rel_err={rel_err_2:.6e}")
        assert rel_err_2 < 1.0e-11, f"Sphere 2 mismatch for pol={pol_idx+1} with rel_err={rel_err_2:.2e}"


@pytest.mark.skipif(not HAS_REFERENCE, reason="MSTM reference dumps not found")
def test_eight_sphere_exact_solve():
    """Verify that the Matrix-Free RCR solver matches the direct dense MSTM-compatible solver
    to double precision limits across an 8-sphere FCC soot-like cluster.
    """
    # ── 1. Setup Physical System & load FCC Geometry ──────────────────────────
    k_medium = 1.0
    import json
    geo_path = "/Users/kaer/doc/fast_mtp_scattering/experiments/exp19/logs/cp13b_geometry.json"
    with open(geo_path) as f:
        geo = json.load(f)
    positions = np.array(geo['positions_nm'])
    nsph = len(positions)
    
    # Load monomer Mie coefficients (all 8 spheres are identical, r=350nm)
    mie_data = load_mie_coefficients(os.path.join(DUMP_DIR, 'two_sphere_mie_coefficients.txt'))
    an = mie_data[1]['an']
    nodr = mie_data[1]['nodr']
    
    nodr_list = [nodr] * nsph
    an_list = [an] * nsph
    
    # Precompute MSTM incident coefficients (model-2)
    pmnp0 = reproduce_incident_coefficients(nodr, alpha=0.0, beta=0.0)
    pmnp0_list = [pmnp0] * nsph
    
    # ── 2. Run Direct Dense Reference Solver (model-2) ────────────────────────
    amnp_ref_m2 = solve_multisphere(positions, nodr_list, an_list, pmnp0_list, k_medium)
    
    # ── 3. Run RCR Matrix-Free Solver (model-1) ───────────────────────────────
    # Initialize Multisphere RCR System
    rcr_system = MultisphereRCRSystem(positions, nodr_list, an_list, k_medium)
    
    from solver.bicgstab_solver import apply_tmatrix_model1
    
    # We solve both polarization columns
    for pol_idx, p_col in enumerate([pmnp0[:, 0], pmnp0[:, 1]]):
        # Build concatenated model-1 RHS b
        b_m1_blocks = []
        for i in range(nsph):
            p_m1 = vector_model2_to_model1(p_col, nodr)
            b_m1_blocks.append(apply_tmatrix_model1(p_m1, an, nodr))
        b_m1 = np.concatenate(b_m1_blocks)
        
        # Solve the system using BiCGSTAB
        x_m1, res = rcr_system.solve(b_m1, tol=1e-12, max_iter=200)
        
        print(f"8-Sphere | Polarization {pol_idx + 1} solved. Relative residual: {res:.6e}")
        assert res < 1e-11, f"Krylov solver failed to converge for pol={pol_idx+1}"
        
        # Split and compare each sphere
        nblk_full_i = 2 * nodr * (nodr + 2)
        for i in range(nsph):
            off = i * nblk_full_i
            xi_m1 = x_m1[off : off + nblk_full_i]
            xi_m2 = vector_model1_to_model2(xi_m1, nodr)
            
            ref_xi = amnp_ref_m2[i][:, pol_idx]
            diff = np.linalg.norm(xi_m2 - ref_xi)
            rel_err = diff / np.linalg.norm(ref_xi)
            print(f"  Sphere {i+1} | rel_err={rel_err:.6e}")
            assert rel_err < 1.0e-11, f"Sphere {i+1} mismatch for pol={pol_idx+1} with rel_err={rel_err:.2e}"

