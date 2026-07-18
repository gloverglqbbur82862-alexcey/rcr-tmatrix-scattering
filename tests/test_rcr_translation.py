#!/usr/bin/env python3
"""Tests for RCR Translation Engine: verifying RCR vs MSTM-compatible dense reference."""

import sys
import os
import numpy as np
import pytest
import warnings

# Suppress warnings from external references or deprecated numpy/numba features
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Optional external reference solver path. These legacy reference tests are
# skipped unless RCR_LEGACY_REFERENCE_PATH points to the reference code.
REFERENCE_PATH = os.environ.get("RCR_LEGACY_REFERENCE_PATH")
if REFERENCE_PATH:
    sys.path.append(REFERENCE_PATH)
try:
    from cp13c_r5_mstm_compatible_dense_solver import (
        translation_matrix_model2,
        _precompute_fnr,
        _precompute_bcof,
        _tran_coef_table,
        model1_to_model2_permutation
    )
    HAS_REFERENCE = True
except ImportError:
    HAS_REFERENCE = False

from rcr_engine.wigner_rotation import WignerRotationProvider
from rcr_engine.coaxial_translation import CoaxialTranslationProvider


def vector_model1_to_model2(x_m1: np.ndarray, p: int) -> np.ndarray:
    nblk_half = p * (p + 2)
    perm_1to2, _ = model1_to_model2_permutation(p)
    x_m2 = np.zeros_like(x_m1)
    # p=1 block
    for idx_m1 in range(nblk_half):
        idx_m2 = perm_1to2[idx_m1]
        x_m2[idx_m2] = x_m1[idx_m1]
    # p=2 block
    for idx_m1 in range(nblk_half):
        idx_m2 = perm_1to2[idx_m1]
        x_m2[idx_m2 + nblk_half] = x_m1[idx_m1 + nblk_half]
    return x_m2


def vector_model2_to_model1(x_m2: np.ndarray, p: int) -> np.ndarray:
    nblk_half = p * (p + 2)
    _, perm_2to1 = model1_to_model2_permutation(p)
    x_m1 = np.zeros_like(x_m2)
    # p=1 block
    for idx_m2 in range(nblk_half):
        idx_m1 = perm_2to1[idx_m2]
        x_m1[idx_m1] = x_m2[idx_m2]
    # p=2 block
    for idx_m2 in range(nblk_half):
        idx_m1 = perm_2to1[idx_m2]
        x_m1[idx_m1 + nblk_half] = x_m2[idx_m2 + nblk_half]
    return x_m1


@pytest.mark.skipif(not HAS_REFERENCE, reason="Old fast_mtp_scattering workspace reference not found")
def test_rcr_matches_dense_reference():
    """Verify that Wigner rotation + coaxial translation + Wigner inverse rotation
    matches the general MSTM-compatible translation matrix to machine precision.
    """
    p = 13  # truncation order
    k_medium = 1.0  # wavenumber
    
    # Initialize providers
    rot_provider = WignerRotationProvider(p)
    trans_provider = CoaxialTranslationProvider(p)
    
    # Create a random input coefficient vector x of size 2*p*(p+2) = 390 in model-1 format
    np.random.seed(42)
    nblk_full = 2 * p * (p + 2)
    x_m1 = np.random.randn(nblk_full) + 1j * np.random.randn(nblk_full)
    
    # Test multiple arbitrary translation vectors (near, mid, and far displacements)
    tvecs = [
        np.array([-2.5, 1.2, 0.8]),   # Near-field arbitrary direction
        np.array([5.0, -3.0, 4.0]),    # Mid-field
        np.array([-12.0, 8.0, -15.0])  # Far-field
    ]
    
    # Precompute reference tables
    fnr = _precompute_fnr(2 * p + 4)
    bcof = _precompute_bcof(2 * p + 4, fnr)
    tran_coef_dict = _tran_coef_table(p, p, fnr, bcof)
    
    for tvec in tvecs:
        # Convert input x to model-2 to apply reference translation matrix
        x_m2 = vector_model1_to_model2(x_m1, p)
        
        # 1. Compute reference translation matrix using the mature CP13C MSTM-compatible code
        # Shape: (390, 390) in model-2 indexing
        A_ref = translation_matrix_model2(tvec, k_medium, p, p, fnr, bcof, tran_coef_dict)
        y_ref_m2 = A_ref @ x_m2
        
        # Convert y_ref back to model-1 to compare with RCR output
        y_ref_m1 = vector_model2_to_model1(y_ref_m2, p)
        
        # 2. Compute translation using RCR decomposition
        r = max(np.linalg.norm(tvec), 1e-300)
        theta = np.arccos(np.clip(tvec[2] / r, -1.0, 1.0))
        phi = np.arctan2(tvec[1], tvec[0])
        
        # RCR Step A: Rotate coordinate system to z-axis (model-1)
        x_rot = rot_provider.rotate_to_z(theta, phi, x_m1)
        
        # RCR Step B: Apply coaxial translation along z-axis (model-1)
        y_rot = trans_provider.apply_coaxial_translation(r, x_rot)
        
        # RCR Step C: Rotate coordinate system back to original frame (model-1)
        y_rcr = rot_provider.rotate_from_z(theta, phi, y_rot)
        
        # 3. Compare outputs in model-1
        norm_ref = np.linalg.norm(y_ref_m1)
        diff = np.linalg.norm(y_rcr - y_ref_m1)
        rel_err = diff / norm_ref
        
        print(f"tvec={tvec} | |y_ref_m1|={norm_ref:.6e} | rel_err={rel_err:.6e}")
        
        # The tolerance is set to 1.0e-12, demonstrating exact double-precision match!
        assert rel_err < 1.0e-12, f"RCR mismatch for tvec={tvec} with rel_err={rel_err:.2e}"
